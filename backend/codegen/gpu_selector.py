"""
GPU Selector - Agent 自主决定每次训练用哪种 GPU。

决策优先级（从高到低）：
    1. 用户 override：调用 train(..., gpu_type="A100") 或 ENV `MODAL_GPU_OVERRIDE`
    2. LLM 推荐：把 intent + data_schema + constraints 喂给 LLM，让它在 catalog 里挑
    3. 启发式回退：基于 modality + 数据规模 + 用户 cost/memory 约束的规则表
    4. 环境兜底：ENV `MODAL_GPU` 作为最终默认值
    5. 硬编码默认：DEFAULT_GPU = "A10G"

设计目标：
- LLM 不可用 / 网络抖动 / 解析失败时，启发式必须给出合理 GPU，绝不抛错
- 决策结果带 `source` 字段（user/llm/heuristic/env/default），方便日志和 UI 展示
- catalog 是单一事实源；新增 GPU 类型只需改这里
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------- GPU Catalog（与 backend/codegen/modal_functions.py 中部署的函数一一对应）----------

GPU_CATALOG: dict[str, dict[str, Any]] = {
    "T4": {
        "vram_gb": 16,
        "price_usd_per_hour": 0.35,
        "good_for": [
            "small image classification (≤32K samples, ≤224px)",
            "tabular gradient boosting",
            "small NLP (BERT-base inference)",
            "rapid prototyping",
        ],
    },
    "A10G": {
        "vram_gb": 24,
        "price_usd_per_hour": 0.50,
        "good_for": [
            "medium vision tasks",
            "audio / speech models",
            "small-to-medium transformer fine-tune",
            "balanced cost / performance default",
        ],
    },
    "A100": {  # A100-40GB on Modal
        "vram_gb": 40,
        "price_usd_per_hour": 0.60,
        "good_for": [
            "large vision dataset (≥50K samples)",
            "transformer fine-tune (≤7B params)",
            "diffusion model training",
            "multi-task / large batch training",
        ],
    },
    "H100": {
        "vram_gb": 80,
        "price_usd_per_hour": 1.20,
        "good_for": [
            "LLM fine-tune (7B-70B params)",
            "large diffusion / generative models",
            "max throughput / time-critical jobs",
        ],
    },
}

DEFAULT_GPU = "A10G"

_GPU_ALIASES = {
    "A100-40GB": "A100",
    "A100_40GB": "A100",
    "A10": "A10G",
}


# ---------- Public API ----------


@dataclass
class GPURecommendation:
    """GPU 选型结果。"""

    gpu_type: str
    reason: str
    source: str  # "user" | "llm" | "heuristic" | "env" | "default"
    estimated_cost_per_hour: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "reason": self.reason,
            "source": self.source,
            "estimated_cost_per_hour": self.estimated_cost_per_hour,
        }


def recommend_gpu(
    *,
    intent: str,
    data_schema: dict | None = None,
    constraints: dict | None = None,
    llm_client: Any = None,
    user_override: str | None = None,
) -> GPURecommendation:
    """根据训练任务上下文推荐一个 GPU 类型。

    永不抛错；任何分支失败都会回退到下一层。
    """
    # 1) 用户显式指定
    override = (user_override or os.environ.get("MODAL_GPU_OVERRIDE") or "").strip()
    if override:
        gpu = _normalize(override)
        if gpu:
            return GPURecommendation(
                gpu_type=gpu,
                reason="user explicit override",
                source="user",
                estimated_cost_per_hour=GPU_CATALOG[gpu]["price_usd_per_hour"],
            )
        logger.warning("Ignoring unknown GPU override %r", override)

    # 2) LLM 推荐
    if llm_client is not None and hasattr(llm_client, "chat_completion"):
        try:
            gpu, reason = _llm_decide(intent, data_schema, constraints, llm_client)
            if gpu:
                return GPURecommendation(
                    gpu_type=gpu,
                    reason=reason or "LLM recommendation",
                    source="llm",
                    estimated_cost_per_hour=GPU_CATALOG[gpu]["price_usd_per_hour"],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM GPU decision failed (%s); falling back to heuristic", exc)

    # 3) 启发式回退
    gpu, reason = _heuristic_decide(intent, data_schema, constraints)
    return GPURecommendation(
        gpu_type=gpu,
        reason=reason,
        source="heuristic",
        estimated_cost_per_hour=GPU_CATALOG[gpu]["price_usd_per_hour"],
    )


def env_default_gpu() -> str:
    """读 ENV `MODAL_GPU` 作为最终兜底；用于 sandbox 构造时的默认值。"""
    raw = (os.environ.get("MODAL_GPU") or "").strip()
    return _normalize(raw) or DEFAULT_GPU


# ---------- Internals ----------


def _normalize(name: str | None) -> str | None:
    if not name:
        return None
    s = name.strip().upper()
    s = _GPU_ALIASES.get(s, s)
    return s if s in GPU_CATALOG else None


def _llm_decide(
    intent: str,
    data_schema: dict | None,
    constraints: dict | None,
    llm_client: Any,
) -> tuple[str | None, str]:
    """让 LLM 在 catalog 里挑 GPU，返回 (gpu, reason)。"""

    catalog_summary = "\n".join(
        f"- {name}: {info['vram_gb']}GB VRAM, ${info['price_usd_per_hour']:.2f}/hr — "
        f"good for {', '.join(info['good_for'])}"
        for name, info in GPU_CATALOG.items()
    )

    system_prompt = (
        "You are a senior ML systems engineer choosing GPUs on Modal Cloud. "
        "Given a training job description, pick exactly ONE GPU type from the catalog "
        "below that gives the best cost / capability trade-off for this job. "
        "Prefer cheaper GPUs unless the job clearly needs more VRAM or throughput.\n\n"
        f"CATALOG:\n{catalog_summary}\n\n"
        "Reply with strict JSON only, in the form: "
        '{"gpu": "T4|A10G|A100|H100", "reason": "<one short sentence>"}.'
    )

    user_payload = {
        "intent": (intent or "").strip(),
        "data_schema": _shrink(data_schema or {}, max_chars=1200),
        "constraints": constraints or {},
    }
    user_msg = json.dumps(user_payload, ensure_ascii=False, default=str)

    raw = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )

    obj = _parse_json_loose(raw)
    if not isinstance(obj, dict):
        return None, ""
    gpu = _normalize(str(obj.get("gpu", "")))
    reason = str(obj.get("reason", "")).strip()[:240]
    return gpu, reason


def _heuristic_decide(
    intent: str,
    data_schema: dict | None,
    constraints: dict | None,
) -> tuple[str, str]:
    """规则化决策 —— LLM 兜底用，必须覆盖所有输入组合。"""

    constraints = constraints or {}
    data_schema = data_schema or {}
    intent_lc = (intent or "").lower()

    max_cost = _as_float(constraints.get("max_cost_per_hour"))
    max_vram = _as_float(constraints.get("max_memory_gb") or constraints.get("min_vram_gb"))

    # 用户给了硬性 cost cap → 选满足要求的最贵那个（性能尽量好），都不满足就选最便宜的
    if max_cost is not None:
        affordable = [
            (g, info) for g, info in GPU_CATALOG.items()
            if info["price_usd_per_hour"] <= max_cost
        ]
        if affordable:
            best = max(affordable, key=lambda kv: kv[1]["vram_gb"])
            return best[0], f"highest-VRAM GPU within ${max_cost:.2f}/hr budget"
        cheapest = min(GPU_CATALOG.items(), key=lambda kv: kv[1]["price_usd_per_hour"])
        return cheapest[0], (
            f"no GPU fits ${max_cost:.2f}/hr cap; using cheapest ({cheapest[0]})"
        )

    # 显存硬性下限 → 选满足要求的最便宜的
    if max_vram is not None and max_vram > 0:
        eligible = [
            (g, info) for g, info in GPU_CATALOG.items()
            if info["vram_gb"] >= max_vram
        ]
        if eligible:
            best = min(eligible, key=lambda kv: kv[1]["price_usd_per_hour"])
            return best[0], f"cheapest GPU with ≥{max_vram:g}GB VRAM"

    # 数据规模/模态推断
    modality = _infer_modality(data_schema, intent_lc)
    sample_count = _estimate_sample_count(data_schema)
    size_mb = _estimate_size_mb(data_schema)

    # 强信号：fine-tune 大模型
    if any(k in intent_lc for k in [
        "llama", "qwen", "gpt-", "lora", "qlora", "fine-tune", "finetune",
        "fine tune", "微调", "大模型", "large language model", " llm ",
    ]):
        return "A100", "LLM/transformer fine-tuning needs ≥40GB VRAM"

    if "diffusion" in intent_lc or "stable diffusion" in intent_lc:
        return "A100", "diffusion training is VRAM-heavy"

    if modality == "image":
        if sample_count and sample_count >= 50_000:
            return "A100", "large vision dataset (≥50K samples)"
        if (size_mb and size_mb >= 5_000) or (sample_count and sample_count >= 5_000):
            return "A10G", "medium vision dataset"
        return "T4", "small vision dataset → cheap T4 is enough"

    if modality == "text":
        if sample_count and sample_count >= 100_000:
            return "A100", "large NLP dataset (≥100K samples)"
        if sample_count and sample_count >= 5_000:
            return "A10G", "medium NLP dataset"
        return "T4", "small NLP dataset → cheap T4"

    if modality == "audio":
        return "A10G", "audio/speech models default to A10G"

    if modality == "time_series":
        return "T4", "time-series models are typically small"

    if modality == "tabular":
        return "T4", "tabular / gradient-boosting workloads → cheap T4"

    # 兜底
    return DEFAULT_GPU, "unknown modality → balanced default"


def _infer_modality(data_schema: dict, intent_lc: str) -> str:
    """从 data_schema 或意图里猜 modality。返回 image/text/audio/time_series/tabular/unknown。"""
    direct = (data_schema.get("data_type") or data_schema.get("modality") or "").lower()
    mapping = {
        "image": "image", "vision": "image", "img": "image",
        "text": "text", "nlp": "text",
        "audio": "audio", "speech": "audio", "voice": "audio",
        "time_series": "time_series", "timeseries": "time_series", "ts": "time_series",
        "tabular": "tabular", "table": "tabular", "csv": "tabular",
    }
    if direct in mapping:
        return mapping[direct]
    for k, v in mapping.items():
        if k in intent_lc:
            return v
    if any(w in intent_lc for w in ["图像", "图片", "视觉", "分类图"]):
        return "image"
    if any(w in intent_lc for w in ["文本", "语言模型", "情感", "翻译"]):
        return "text"
    if any(w in intent_lc for w in ["音频", "语音"]):
        return "audio"
    if any(w in intent_lc for w in ["时序", "时间序列", "预测时间"]):
        return "time_series"
    if any(w in intent_lc for w in ["表格", "结构化", "csv"]):
        return "tabular"
    return "unknown"


def _estimate_sample_count(data_schema: dict) -> int | None:
    candidates = [
        data_schema.get("sample_count"),
        data_schema.get("num_samples"),
        data_schema.get("rows"),
        data_schema.get("n_samples"),
    ]
    file_scan = data_schema.get("file_scan") or {}
    if isinstance(file_scan, dict):
        candidates.append(file_scan.get("file_count"))
        candidates.append(file_scan.get("num_files"))
    exploration = data_schema.get("exploration") or {}
    if isinstance(exploration, dict):
        candidates.append(exploration.get("sample_count"))
        candidates.append(exploration.get("rows"))
    for x in candidates:
        if isinstance(x, (int, float)) and x > 0:
            return int(x)
    return None


def _estimate_size_mb(data_schema: dict) -> float | None:
    candidates = [
        data_schema.get("size_mb"),
        data_schema.get("dataset_size_mb"),
    ]
    file_scan = data_schema.get("file_scan") or {}
    if isinstance(file_scan, dict):
        candidates.append(file_scan.get("total_size_mb"))
        bytes_ = file_scan.get("total_bytes") or file_scan.get("size_bytes")
        if isinstance(bytes_, (int, float)) and bytes_ > 0:
            candidates.append(bytes_ / 1024 / 1024)
    for x in candidates:
        if isinstance(x, (int, float)) and x > 0:
            return float(x)
    return None


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _shrink(obj: Any, max_chars: int) -> Any:
    """把 data_schema 里的大字段（如 file_scan 里的全文件列表）截断，避免 prompt 爆。"""
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)[:max_chars]
    if len(text) <= max_chars:
        return obj
    return {"_truncated_schema_preview": text[:max_chars] + "…"}


def _parse_json_loose(raw: str) -> Any:
    """容忍 LLM 输出 JSON 前后带 ``` 包裹或前缀解释文字。"""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None
