"""
Modal Sandbox Executor - Client side

参考 Synapse-agent backend/app/gpu/modal_executor.py 的做法：
- 本类**不**拥有任何 Modal 训练逻辑（训练逻辑在 modal_functions.py）
- 只负责：把 GeneratedProgram 多文件 + 配置 + 数据集打包 → 远程调用对应 GPU 函数

对外接口与 codegen.sandbox.SandboxExecutor.execute_training 完全等价，
optimizer.HybridOptimizer 不需要任何改动即可切换后端。

切换：
    sandbox = ModalSandboxExecutor(gpu_type="A100", dataset_id="...", dataset_root="...")
    optimizer = HybridOptimizer(qa_pipeline=..., sandbox=sandbox, ...)

环境变量：
    USE_MODAL=1                       开启 Modal 后端
    MODAL_GPU=A100|T4|A10G|H100       默认 A100
    MODAL_INLINE_PACK_LIMIT_MB=200    超过此体积自动走 Volume；为 0 则禁用 inline
    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET   认证（Zeabur 容器只能用这种）
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# 默认 Modal app 名称（必须与 modal_functions.py 中的 APP_NAME 一致）
DEFAULT_APP_NAME = "vibeml-training"
QA_FUNCTION_NAME = "qa_run_python"

GPU_FUNCTION_MAP = {
    "T4": "execute_training_t4",
    "A10G": "execute_training_a10g",
    "A100": "execute_training_a100",
    "A100-40GB": "execute_training_a100",
    "H100": "execute_training_h100",
}

GPU_PRICE_USD_PER_HOUR = {
    "T4": 0.35,
    "A10G": 0.50,
    "A100": 0.60,
    "A100-40GB": 0.60,
    "H100": 1.20,
}


@dataclass
class _DatasetCache:
    """避免同一 dataset 在 BO 60 个 trial 里 zip 60 次。"""
    dataset_id: str
    mode: str  # "inline" | "volume" | "none"
    payload: dict[str, Any] = field(default_factory=dict)
    cached_at: float = 0.0


class ModalSandboxExecutor:
    """Modal-backed sandbox executor.

    与本地 ``SandboxExecutor`` 接口完全一致（同样有 ``smoke_test`` /
    ``run_python`` / ``execute_training``），只是把执行后端换成 Modal 容器：

    - ``run_python`` / ``smoke_test``：调 ``qa_run_python`` CPU 函数（~$0.0001/s）
      —— Zeabur 后端没装 torch / cv2，QA 必须在 Modal 镜像里跑。
    - ``execute_training``：调对应 GPU 函数（T4/A10G/A100/H100）跑真训练。
    """

    def __init__(
        self,
        gpu_type: str = "A100",
        dataset_id: str | None = None,
        dataset_root: str | Path | None = None,
        app_name: str = DEFAULT_APP_NAME,
        inline_pack_limit_mb: int | None = None,
        target_metric: str = "val_loss",
    ):
        self.app_name = app_name
        self.gpu_type = self._normalize_gpu_type(gpu_type)
        self.dataset_id = dataset_id
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.target_metric = target_metric

        if inline_pack_limit_mb is None:
            inline_pack_limit_mb = int(os.environ.get("MODAL_INLINE_PACK_LIMIT_MB", "200"))
        self.inline_pack_limit_mb = inline_pack_limit_mb

        self._dataset_cache: _DatasetCache | None = None
        self._function_cache: dict[str, Any] = {}

    # ---------- 公共 API ----------

    def execute_training(
        self,
        program: "GeneratedProgram",
        config: dict,
        timeout: int = 3600,
    ) -> dict:
        """与 SandboxExecutor.execute_training 等价。"""
        try:
            modal_fn = self._get_modal_function()
        except Exception as e:
            return self._error_dict(
                f"无法连接到 Modal 函数 {self.app_name}.{self._function_name()}: {e}\n"
                f"请确认：\n"
                f"  1) 已运行 `python -m modal deploy backend/codegen/deploy_modal.py` 部署函数；\n"
                f"  2) 已设置 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET 环境变量（Zeabur 容器）。"
            )

        try:
            program_files = self._extract_program_files(program)
        except Exception as e:
            return self._error_dict(f"打包训练程序失败: {e}")

        try:
            dataset_payload = self._build_dataset_payload()
        except Exception as e:
            return self._error_dict(f"打包数据集失败: {e}")

        start = time.time()

        try:
            # 同步路径：Modal 的 .spawn() + .get() 是阻塞的，
            # optimizer 在线程池里跑，我们直接同步调用即可。
            call = modal_fn.spawn(
                program_files,
                config,
                dataset_payload,
                self.target_metric,
                timeout,
            )
            # call.get(timeout=...) 会等结果或抛 modal.exception.FunctionTimeoutError
            result = call.get(timeout=timeout + 120)  # 多给 2 分钟容错
        except Exception as e:
            return self._error_dict(
                f"Modal 训练失败: {type(e).__name__}: {e}",
                duration_sec=time.time() - start,
            )

        if not isinstance(result, dict):
            return self._error_dict(
                f"Modal 函数返回了非 dict 结果: {type(result).__name__}",
                duration_sec=time.time() - start,
            )

        # 统一字段：保持和本地 SandboxExecutor 一致
        elapsed = result.get("duration_sec", time.time() - start)
        out: dict[str, Any] = {
            "status": result.get("status", "failed"),
            "metrics": result.get("metrics") or {},
            "duration_ms": elapsed * 1000,
            "duration_sec": elapsed,
            "logs": result.get("logs", ""),
        }
        if result.get("error"):
            out["error"] = result["error"]
        if result.get("traceback"):
            out["traceback"] = result["traceback"]
        return out

    def estimate_cost_usd(self, duration_sec: float) -> float:
        """根据 GPU 类型估算成本。"""
        price = GPU_PRICE_USD_PER_HOUR.get(self.gpu_type, 0.60)
        return price * duration_sec / 3600

    # ---------- QA 用：把脚本扔进 Modal CPU 容器执行 ----------

    def run_python(
        self,
        files: dict[str, str],
        entry: str = "run_tests.py",
        timeout_sec: int = 180,
    ) -> dict[str, Any]:
        """通用入口：在 Modal CPU 镜像里跑 entry 脚本。

        与 ``SandboxExecutor.run_python`` 接口一致。返回值同样是
        ``{returncode, stdout, stderr, duration_sec, timed_out}``，
        QA pipeline / UnitTestGenerator 不用关心后端是本地还是 Modal。
        """
        if entry not in files:
            raise ValueError(f"entry script {entry!r} 不在 files 中")

        try:
            qa_fn = self._get_qa_function()
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": (
                    f"无法连接到 Modal QA 函数 {self.app_name}.{QA_FUNCTION_NAME}: {e}\n"
                    f"请确认已运行 `python -m modal deploy backend/codegen/deploy_modal.py` 部署最新版函数。"
                ),
                "duration_sec": 0.0,
                "timed_out": False,
            }

        start = time.time()
        try:
            call = qa_fn.spawn(files, entry, timeout_sec)
            result = call.get(timeout=timeout_sec + 90)
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Modal qa_run_python 调用失败: {type(e).__name__}: {e}",
                "duration_sec": time.time() - start,
                "timed_out": False,
            }

        if not isinstance(result, dict):
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Modal qa_run_python 返回了非 dict: {type(result).__name__}",
                "duration_sec": time.time() - start,
                "timed_out": False,
            }

        # 透传字段，缺失就补默认
        return {
            "returncode": int(result.get("returncode", -1)),
            "stdout": result.get("stdout", "") or "",
            "stderr": result.get("stderr", "") or "",
            "duration_sec": float(result.get("duration_sec", time.time() - start)),
            "timed_out": bool(result.get("timed_out", False)),
        }

    def smoke_test(self, program: "GeneratedProgram") -> list:
        """与 SandboxExecutor.smoke_test 接口一致；脚本在 Modal 容器里跑。

        每个 stage 一次远程调用；Modal 函数容器会复用（warm 状态下 <1s/次）。
        """
        from .sandbox import (  # 避免循环 import
            SMOKE_SCRIPTS,
            SmokeTestResult,
            build_program_files,
            parse_smoke_stdout,
        )

        program_files = build_program_files(program)
        results: list = []

        for stage, script in SMOKE_SCRIPTS.items():
            files = dict(program_files)
            entry = f"test_{stage}.py"
            files[entry] = script
            try:
                run_result = self.run_python(files, entry=entry, timeout_sec=180)
            except Exception as e:
                results.append(SmokeTestResult(
                    passed=False,
                    stage=stage,
                    duration_ms=0.0,
                    error_message=f"Modal 烟测调用异常: {e}",
                ))
                continue

            if run_result.get("timed_out"):
                results.append(SmokeTestResult(
                    passed=False,
                    stage=stage,
                    duration_ms=180_000.0,
                    error_message="Modal 烟测超时（>180s）",
                ))
                continue

            results.append(parse_smoke_stdout(
                stage=stage,
                returncode=int(run_result.get("returncode", -1)),
                stdout=run_result.get("stdout", ""),
                stderr=run_result.get("stderr", ""),
                duration_ms=float(run_result.get("duration_sec", 0.0)) * 1000,
            ))

        return results

    # ---------- Modal SDK 交互 ----------

    @staticmethod
    def _normalize_gpu_type(gpu_type: str) -> str:
        gpu_type = (gpu_type or "A100").upper()
        if gpu_type not in GPU_FUNCTION_MAP:
            logger.warning("Unknown GPU type %s, falling back to A100", gpu_type)
            return "A100"
        return gpu_type

    def _function_name(self) -> str:
        return GPU_FUNCTION_MAP[self.gpu_type]

    def _get_modal_function(self):
        fn_name = self._function_name()
        cached = self._function_cache.get(fn_name)
        if cached is not None:
            return cached

        import modal  # 延迟导入，避免本地未装 modal 也能 import 本模块
        # Modal SDK ≥0.64
        fn = modal.Function.from_name(self.app_name, fn_name)
        self._function_cache[fn_name] = fn
        logger.info("Resolved Modal function %s.%s", self.app_name, fn_name)
        return fn

    def _get_qa_function(self):
        """解析 CPU QA 函数（所有 GPU 类型共享同一个）。"""
        cached = self._function_cache.get(QA_FUNCTION_NAME)
        if cached is not None:
            return cached

        import modal  # 延迟导入
        fn = modal.Function.from_name(self.app_name, QA_FUNCTION_NAME)
        self._function_cache[QA_FUNCTION_NAME] = fn
        logger.info("Resolved Modal QA function %s.%s", self.app_name, QA_FUNCTION_NAME)
        return fn

    # ---------- Program 打包 ----------

    @staticmethod
    def _extract_program_files(program: "GeneratedProgram") -> dict[str, str]:
        """把 GeneratedProgram 的 5 个代码文件抽成 {filename: source}。"""
        import yaml

        files = {
            "model.py": getattr(program, "model_code", "") or "",
            "loss.py": getattr(program, "loss_code", "") or "",
            "data_pipeline.py": getattr(program, "data_pipeline_code", "") or "",
            "train_loop.py": getattr(program, "train_loop_code", "") or "",
        }
        ss = getattr(program, "search_space", None) or {}
        try:
            files["search_space.yaml"] = yaml.dump(ss, allow_unicode=True)
        except Exception:
            files["search_space.yaml"] = ""
        return files

    # ---------- Dataset 打包 ----------

    def _build_dataset_payload(self) -> dict | None:
        """构造给 Modal 函数的 dataset 参数。

        策略：
        - 没有 dataset_id 或 dataset_root: 返回 None（用户的训练代码自给自足）
        - dataset 总大小 ≤ inline_pack_limit_mb: zip + base64 inline
        - 否则: 检查 Volume 是否已 push，否则提示用户先 push
        """
        if not self.dataset_id or not self.dataset_root:
            return None

        if self._dataset_cache and self._dataset_cache.dataset_id == self.dataset_id:
            return self._dataset_cache.payload

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root 不存在: {self.dataset_root}")

        size_bytes = self._dir_size_bytes(self.dataset_root)
        size_mb = size_bytes / 1024 / 1024
        logger.info("Dataset %s size: %.1f MB", self.dataset_id, size_mb)

        if self.inline_pack_limit_mb > 0 and size_mb <= self.inline_pack_limit_mb:
            payload = self._pack_inline()
            self._dataset_cache = _DatasetCache(
                dataset_id=self.dataset_id,
                mode="inline",
                payload=payload,
                cached_at=time.time(),
            )
            return payload

        # 走 Volume：要求客户端先调用 push_dataset_to_volume()
        payload = {"mode": "volume", "dataset_id": self.dataset_id}
        self._dataset_cache = _DatasetCache(
            dataset_id=self.dataset_id,
            mode="volume",
            payload=payload,
            cached_at=time.time(),
        )
        logger.info(
            "Dataset %s (%.1f MB) 超过 inline 阈值 %d MB，将走 Volume 模式。"
            " 请确保已调用 push_dataset_to_volume(...)",
            self.dataset_id, size_mb, self.inline_pack_limit_mb,
        )
        return payload

    def _pack_inline(self) -> dict:
        """zip 整个 dataset_root 并 base64 编码。"""
        assert self.dataset_root is not None
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in self.dataset_root.rglob("*"):
                if fp.is_file():
                    arcname = fp.relative_to(self.dataset_root).as_posix()
                    zf.write(fp, arcname)
        zip_bytes = buf.getvalue()
        return {
            "mode": "inline",
            "dataset_id": self.dataset_id,
            "zip_b64": base64.b64encode(zip_bytes).decode("ascii"),
            "uncompressed_size": self._dir_size_bytes(self.dataset_root),
            "compressed_size": len(zip_bytes),
        }

    @staticmethod
    def _dir_size_bytes(root: Path) -> int:
        total = 0
        for fp in root.rglob("*"):
            if fp.is_file():
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
        return total

    def push_dataset_to_volume(self, batch_size_mb: int = 64) -> dict:
        """把 self.dataset_root 同步到 Modal Volume，返回写入统计。

        必须在切换到 Volume 模式前先调用一次（同 dataset 只需 push 一次）。
        """
        if not self.dataset_id or not self.dataset_root:
            raise ValueError("push_dataset_to_volume 需要 dataset_id 和 dataset_root")
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root 不存在: {self.dataset_root}")

        import modal  # 延迟导入
        push_fn = modal.Function.from_name(self.app_name, "push_dataset_files")

        batch: list[dict] = []
        batch_bytes = 0
        total_bytes = 0
        total_files = 0
        results: list[dict] = []

        def flush():
            nonlocal batch, batch_bytes
            if not batch:
                return
            res = push_fn.remote(self.dataset_id, batch)
            results.append(res)
            batch = []
            batch_bytes = 0

        for fp in self.dataset_root.rglob("*"):
            if not fp.is_file():
                continue
            rel = fp.relative_to(self.dataset_root).as_posix()
            try:
                content = fp.read_bytes()
            except OSError as e:
                logger.warning("跳过无法读取的文件 %s: %s", fp, e)
                continue
            batch.append({
                "path": rel,
                "content_b64": base64.b64encode(content).decode("ascii"),
            })
            batch_bytes += len(content)
            total_bytes += len(content)
            total_files += 1
            if batch_bytes >= batch_size_mb * 1024 * 1024:
                flush()

        flush()

        return {
            "dataset_id": self.dataset_id,
            "files_pushed": total_files,
            "bytes_pushed": total_bytes,
            "batches": len(results),
            "remote_results": results,
        }

    # ---------- 工具 ----------

    @staticmethod
    def _error_dict(msg: str, duration_sec: float = 0.0) -> dict:
        return {
            "status": "failed",
            "metrics": {},
            "duration_ms": duration_sec * 1000,
            "duration_sec": duration_sec,
            "error": msg,
            "logs": msg,
        }


# ---------- 工厂函数 ----------

def make_executor_from_env(
    dataset_id: str | None = None,
    dataset_root: str | Path | None = None,
    target_metric: str = "val_loss",
):
    """根据环境变量决定返回本地 SandboxExecutor 还是 ModalSandboxExecutor。

    USE_MODAL=1 (or "true"/"yes") → Modal
    其他 → 本地 SandboxExecutor
    """
    use_modal = os.environ.get("USE_MODAL", "").strip().lower() in {"1", "true", "yes", "on"}

    if use_modal:
        gpu_type = os.environ.get("MODAL_GPU", "A100")
        return ModalSandboxExecutor(
            gpu_type=gpu_type,
            dataset_id=dataset_id,
            dataset_root=dataset_root,
            target_metric=target_metric,
        )

    from .sandbox import SandboxExecutor
    return SandboxExecutor()
