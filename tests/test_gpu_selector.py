"""Unit tests for backend.codegen.gpu_selector.

覆盖优先级：user override > LLM > heuristic，以及 catalog/容错路径。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from backend.codegen.gpu_selector import (
    DEFAULT_GPU,
    GPU_CATALOG,
    GPURecommendation,
    env_default_gpu,
    recommend_gpu,
)


# ---------- override ----------

def test_user_override_wins_over_everything():
    llm = MagicMock()
    llm.chat_completion = MagicMock(return_value='{"gpu": "T4", "reason": "small"}')

    rec = recommend_gpu(
        intent="train llama-7b on 1M samples",  # 启发式会推 A100/H100
        data_schema={"data_type": "text", "sample_count": 1_000_000},
        llm_client=llm,
        user_override="H100",
    )

    assert rec.gpu_type == "H100"
    assert rec.source == "user"
    llm.chat_completion.assert_not_called()


def test_user_override_normalizes_alias():
    rec = recommend_gpu(intent="x", user_override="A100-40GB")
    assert rec.gpu_type == "A100"
    assert rec.source == "user"


def test_unknown_override_falls_through_to_heuristic(caplog):
    rec = recommend_gpu(
        intent="train tabular gradient boost",
        data_schema={"data_type": "tabular", "sample_count": 100},
        user_override="V100-bogus",
    )
    assert rec.gpu_type == "T4"
    assert rec.source == "heuristic"


def test_env_override_acts_as_user_override(monkeypatch):
    monkeypatch.setenv("MODAL_GPU_OVERRIDE", "H100")
    rec = recommend_gpu(intent="train tiny tabular model")
    assert rec.gpu_type == "H100"
    assert rec.source == "user"


# ---------- LLM ----------

def test_llm_recommendation_used_when_valid():
    llm = MagicMock()
    llm.chat_completion = MagicMock(
        return_value='{"gpu": "A10G", "reason": "balanced for medium NLP"}'
    )

    rec = recommend_gpu(
        intent="fine-tune a small NLP classifier on 10k samples",
        data_schema={"data_type": "text", "sample_count": 10_000},
        llm_client=llm,
    )

    assert rec.gpu_type == "A10G"
    assert rec.source == "llm"
    assert "balanced" in rec.reason.lower()


def test_llm_response_with_codefence_is_parsed():
    llm = MagicMock()
    llm.chat_completion = MagicMock(
        return_value="```json\n{\"gpu\": \"T4\", \"reason\": \"small dataset\"}\n```"
    )
    rec = recommend_gpu(
        intent="image classification",
        data_schema={"data_type": "image", "sample_count": 200},
        llm_client=llm,
    )
    assert rec.gpu_type == "T4"
    assert rec.source == "llm"


def test_llm_invalid_gpu_falls_back_to_heuristic():
    llm = MagicMock()
    llm.chat_completion = MagicMock(
        return_value='{"gpu": "RTX-4090", "reason": "consumer card"}'
    )
    rec = recommend_gpu(
        intent="image classification on 200 imgs",
        data_schema={"data_type": "image", "sample_count": 200},
        llm_client=llm,
    )
    assert rec.gpu_type == "T4"
    assert rec.source == "heuristic"


def test_llm_exception_does_not_propagate():
    llm = MagicMock()
    llm.chat_completion = MagicMock(side_effect=RuntimeError("LLM down"))
    rec = recommend_gpu(
        intent="image classification",
        data_schema={"data_type": "image", "sample_count": 200},
        llm_client=llm,
    )
    assert rec.gpu_type in GPU_CATALOG
    assert rec.source == "heuristic"


def test_llm_garbage_falls_back_to_heuristic():
    llm = MagicMock()
    llm.chat_completion = MagicMock(return_value="not even json at all")
    rec = recommend_gpu(
        intent="image classification on 200 imgs",
        data_schema={"data_type": "image", "sample_count": 200},
        llm_client=llm,
    )
    assert rec.source == "heuristic"


def test_llm_called_with_strict_json_format():
    llm = MagicMock()
    llm.chat_completion = MagicMock(
        return_value='{"gpu": "A10G", "reason": "x"}'
    )
    recommend_gpu(intent="x", data_schema={"data_type": "image"}, llm_client=llm)

    # response_format=json_object，temperature=0.0 是关键
    kwargs = llm.chat_completion.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs.get("response_format") == {"type": "json_object"}
    msgs = kwargs["messages"]
    assert any("CATALOG" in m["content"] for m in msgs)


# ---------- heuristic ----------

@pytest.mark.parametrize("intent_keyword", [
    "fine-tune llama-7b",
    "QLoRA on Qwen-7B",
    "微调大模型",
    "fine tune transformer",
])
def test_heuristic_picks_a100_for_llm_finetune(intent_keyword):
    rec = recommend_gpu(intent=intent_keyword, llm_client=None)
    assert rec.gpu_type == "A100"
    assert rec.source == "heuristic"


def test_heuristic_picks_a100_for_diffusion():
    rec = recommend_gpu(intent="train a stable diffusion variant on art images")
    assert rec.gpu_type == "A100"


def test_heuristic_image_size_buckets():
    # 大数据集 → A100
    rec = recommend_gpu(
        intent="image classifier",
        data_schema={"data_type": "image", "sample_count": 80_000},
    )
    assert rec.gpu_type == "A100"

    # 中等 → A10G
    rec = recommend_gpu(
        intent="image classifier",
        data_schema={"data_type": "image", "sample_count": 10_000},
    )
    assert rec.gpu_type == "A10G"

    # 小 → T4
    rec = recommend_gpu(
        intent="image classifier",
        data_schema={"data_type": "image", "sample_count": 500},
    )
    assert rec.gpu_type == "T4"


def test_heuristic_text_size_buckets():
    rec_large = recommend_gpu(
        intent="train BERT classifier",
        data_schema={"data_type": "text", "sample_count": 200_000},
    )
    assert rec_large.gpu_type == "A100"

    rec_med = recommend_gpu(
        intent="train BERT classifier",
        data_schema={"data_type": "text", "sample_count": 20_000},
    )
    assert rec_med.gpu_type == "A10G"

    rec_small = recommend_gpu(
        intent="train BERT classifier",
        data_schema={"data_type": "text", "sample_count": 200},
    )
    assert rec_small.gpu_type == "T4"


def test_heuristic_audio_default_a10g():
    rec = recommend_gpu(
        intent="speech recognition",
        data_schema={"data_type": "audio"},
    )
    assert rec.gpu_type == "A10G"


def test_heuristic_timeseries_picks_t4():
    rec = recommend_gpu(
        intent="forecast hourly traffic",
        data_schema={"data_type": "time_series"},
    )
    assert rec.gpu_type == "T4"


def test_heuristic_tabular_picks_t4():
    rec = recommend_gpu(
        intent="predict customer churn from CSV",
        data_schema={"data_type": "tabular", "rows": 100_000},
    )
    assert rec.gpu_type == "T4"


def test_heuristic_chinese_keywords_infer_modality():
    rec_img = recommend_gpu(intent="对图像分类，识别花卉")
    assert rec_img.gpu_type in {"T4", "A10G", "A100"}
    rec_txt = recommend_gpu(intent="文本情感分析", data_schema={"sample_count": 200})
    assert rec_txt.gpu_type == "T4"


def test_heuristic_unknown_falls_back_default():
    rec = recommend_gpu(intent="do something unique")
    assert rec.gpu_type == DEFAULT_GPU
    assert rec.source == "heuristic"


# ---------- constraints ----------

def test_max_cost_picks_largest_within_budget():
    rec = recommend_gpu(
        intent="train llama-7b",  # 即使 LLM 提示也会被覆盖
        constraints={"max_cost_per_hour": 0.55},
    )
    # T4 (0.35) and A10G (0.50) fit; A100 (0.60) doesn't. 取 A10G（VRAM 更大）
    assert rec.gpu_type == "A10G"


def test_max_cost_no_fit_falls_back_to_cheapest():
    rec = recommend_gpu(
        intent="x",
        constraints={"max_cost_per_hour": 0.10},
    )
    assert rec.gpu_type == "T4"  # 唯一 0.35


def test_min_vram_constraint():
    rec = recommend_gpu(
        intent="image",
        constraints={"min_vram_gb": 30},
        data_schema={"data_type": "image", "sample_count": 500},  # heuristic 否则会推 T4
    )
    # Eligible: A100 (40GB), H100 (80GB). 取最便宜 → A100
    assert rec.gpu_type == "A100"


# ---------- file_scan / size_mb 解析 ----------

def test_file_scan_total_bytes_used_for_size_estimate():
    rec = recommend_gpu(
        intent="image classifier",
        data_schema={
            "data_type": "image",
            "file_scan": {"total_bytes": 6 * 1024 * 1024 * 1024},  # 6 GB
        },
    )
    assert rec.gpu_type == "A10G"


def test_file_scan_file_count_used_for_sample_estimate():
    rec = recommend_gpu(
        intent="image classifier",
        data_schema={
            "data_type": "image",
            "file_scan": {"file_count": 60_000},
        },
    )
    assert rec.gpu_type == "A100"


# ---------- helpers ----------

def test_env_default_gpu_normalizes(monkeypatch):
    monkeypatch.setenv("MODAL_GPU", "a100-40gb")
    assert env_default_gpu() == "A100"

    monkeypatch.setenv("MODAL_GPU", "")
    assert env_default_gpu() == DEFAULT_GPU


def test_recommendation_to_dict_round_trips():
    llm = MagicMock()
    llm.chat_completion = MagicMock(return_value=json.dumps({"gpu": "T4", "reason": "x"}))
    rec = recommend_gpu(intent="x", llm_client=llm)
    d = rec.to_dict()
    assert set(d.keys()) == {"gpu_type", "reason", "source", "estimated_cost_per_hour"}
    assert d["gpu_type"] == "T4"
    assert d["estimated_cost_per_hour"] == GPU_CATALOG["T4"]["price_usd_per_hour"]
