from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


GPU_FUNCTIONS = {
    "T4": "execute_training_t4",
    "A10G": "execute_training_a10g",
    "A100": "execute_training_a100",
    "H100": "execute_training_h100",
}


@dataclass
class ModalLabConfig:
    app_name: str = field(default_factory=lambda: os.getenv("MODAL_APP_NAME", "vibeml-training"))
    default_gpu: str = field(default_factory=lambda: os.getenv("MODAL_GPU", "A10G"))
    enabled: bool = field(default_factory=lambda: os.getenv("MODAL_ENABLED", "").lower() in {"1", "true", "yes"})
    allow_local_fallback: bool = field(default_factory=lambda: os.getenv("MODAL_ALLOW_LOCAL_FALLBACK", "1").lower() not in {"0", "false", "no"})


@dataclass
class ExecutionResult:
    status: str
    logs: str
    metrics: dict[str, float]
    artifacts: dict[str, bytes]
    error_message: str | None = None
    elapsed_seconds: float = 0.0
    backend: str = "local"


class ModalLabExecutor:
    """当前项目自己的 Modal Lab 客户端封装。"""

    def __init__(self, config: ModalLabConfig | None = None):
        self.config = config or ModalLabConfig()

    def run_bundle(
        self,
        *,
        bundle_files: dict[str, str],
        binary_files: dict[str, bytes] | None,
        entrypoint: str,
        job_id: str,
        gpu_type: str | None = None,
        timeout_seconds: int = 3600,
    ) -> ExecutionResult:
        gpu = (gpu_type or self.config.default_gpu or "A10G").upper()
        if self.config.enabled:
            try:
                return self._run_modal(
                    bundle_files=bundle_files,
                    binary_files=binary_files or {},
                    entrypoint=entrypoint,
                    job_id=job_id,
                    gpu_type=gpu,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                if not self.config.allow_local_fallback:
                    raise
                return self._run_local(
                    bundle_files=bundle_files,
                    binary_files=binary_files or {},
                    entrypoint=entrypoint,
                    timeout_seconds=timeout_seconds,
                    error_prefix=f"Modal execution failed, fallback to local: {exc}",
                )

        return self._run_local(
            bundle_files=bundle_files,
            binary_files=binary_files or {},
            entrypoint=entrypoint,
            timeout_seconds=timeout_seconds,
        )

    def _run_modal(
        self,
        *,
        bundle_files: dict[str, str],
        binary_files: dict[str, bytes],
        entrypoint: str,
        job_id: str,
        gpu_type: str,
        timeout_seconds: int,
    ) -> ExecutionResult:
        try:
            import modal  # type: ignore
        except Exception as exc:
            raise RuntimeError("modal package is not installed") from exc

        function_name = GPU_FUNCTIONS.get(gpu_type)
        if not function_name:
            raise ValueError(f"unsupported gpu type: {gpu_type}")

        payload = {
            "job_id": job_id,
            "files": bundle_files,
            "binary_files": {
                path: base64.b64encode(content).decode("ascii")
                for path, content in binary_files.items()
            },
            "entrypoint": entrypoint,
            "timeout_seconds": timeout_seconds,
        }

        started_at = time.time()
        fn = modal.Function.from_name(self.config.app_name, function_name)
        result = fn.remote(payload)
        return ExecutionResult(
            status=result.get("status", "failed"),
            logs=result.get("logs", ""),
            metrics=result.get("metrics", {}) or {},
            artifacts={
                name: base64.b64decode(blob["content_b64"])
                for name, blob in (result.get("artifacts") or {}).items()
                if blob.get("content_b64")
            },
            error_message=result.get("error"),
            elapsed_seconds=float(result.get("elapsed_seconds", time.time() - started_at)),
            backend="modal",
        )

    def _run_local(
        self,
        *,
        bundle_files: dict[str, str],
        binary_files: dict[str, bytes],
        entrypoint: str,
        timeout_seconds: int,
        error_prefix: str | None = None,
    ) -> ExecutionResult:
        import shutil

        started_at = time.time()
        workdir = Path(tempfile.mkdtemp(prefix="vibeml-local-"))
        artifact_dir = workdir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            for rel_path, content in bundle_files.items():
                file_path = workdir / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
            for rel_path, blob in binary_files.items():
                file_path = workdir / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(blob)

            env = os.environ.copy()
            env["VIBEML_ARTIFACT_DIR"] = str(artifact_dir)
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.run(
                [sys.executable, entrypoint],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            logs = (process.stdout or "") + ("\n[stderr]\n" + process.stderr if process.stderr else "")
            if error_prefix:
                logs = f"{error_prefix}\n\n{logs}"

            artifacts: dict[str, bytes] = {}
            for file_path in artifact_dir.rglob("*"):
                if file_path.is_file():
                    artifacts[str(file_path.relative_to(artifact_dir))] = file_path.read_bytes()

            status = "completed" if process.returncode == 0 else "failed"
            return ExecutionResult(
                status=status,
                logs=logs,
                metrics=self._extract_metrics(logs),
                artifacts=artifacts,
                error_message=None if status == "completed" else f"local runner exited with code {process.returncode}",
                elapsed_seconds=time.time() - started_at,
                backend="local",
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _extract_metrics(logs: str) -> dict[str, float]:
        import re

        metrics: dict[str, float] = {}
        patterns = {
            "accuracy": r"(?:test_acc|test accuracy|accuracy)\s*[:=]\s*([0-9.]+)",
            "loss": r"(?:test_loss|test loss|loss)\s*[:=]\s*([0-9.]+)",
            "f1": r"(?:f1|f1_score)\s*[:=]\s*([0-9.]+)",
        }
        for name, pattern in patterns.items():
            matches = re.findall(pattern, logs, re.IGNORECASE)
            if matches:
                try:
                    metrics[name] = float(matches[-1])
                except ValueError:
                    continue
        return metrics
