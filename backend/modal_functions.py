#!/usr/bin/env python3
"""
独立的 Modal 训练函数定义。

这个文件不依赖当前项目的本地模块，便于 `modal deploy` 直接部署。
"""

from __future__ import annotations

import base64
import json
from typing import Any

import modal


APP_NAME = "vibeml-training"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.5.1",
        "pillow==10.4.0",
        "pyarrow==17.0.0",
    )
)


def _extract_metrics(logs: str) -> dict[str, float]:
    import re

    metrics: dict[str, float] = {}
    patterns = {
        "accuracy": r"(?:test_acc|test accuracy|accuracy)\s*[:=]\s*([0-9.]+)",
        "loss": r"(?:test_loss|test loss|loss)\s*[:=]\s*([0-9.]+)",
        "f1": r"(?:f1|f1_score)\s*[:=]\s*([0-9.]+)",
        "precision": r"precision\s*[:=]\s*([0-9.]+)",
        "recall": r"recall\s*[:=]\s*([0-9.]+)",
    }
    for name, pattern in patterns.items():
        matches = re.findall(pattern, logs, re.IGNORECASE)
        if matches:
            try:
                metrics[name] = float(matches[-1])
            except ValueError:
                continue
    return metrics


def _execute_training(payload: dict[str, Any]) -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    import traceback
    from pathlib import Path

    started_at = time.time()
    workdir = Path(tempfile.mkdtemp(prefix="vibeml-modal-"))
    artifact_dir = workdir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    files = payload.get("files", {})
    binary_files = payload.get("binary_files", {})
    entrypoint = payload.get("entrypoint", "run_training.py")
    timeout_seconds = int(payload.get("timeout_seconds", 3600))
    env = os.environ.copy()
    env.update(payload.get("env", {}))
    env["VIBEML_ARTIFACT_DIR"] = str(artifact_dir)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        for rel_path, content in files.items():
            file_path = workdir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
        for rel_path, blob in binary_files.items():
            file_path = workdir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(base64.b64decode(blob))

        process = subprocess.run(
            [sys.executable, entrypoint],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        logs = (process.stdout or "") + ("\n[stderr]\n" + process.stderr if process.stderr else "")

        artifacts: dict[str, dict[str, str]] = {}
        for file_path in artifact_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = str(file_path.relative_to(artifact_dir))
            artifacts[rel_path] = {
                "content_b64": base64.b64encode(file_path.read_bytes()).decode("ascii"),
            }

        status = "completed" if process.returncode == 0 else "failed"
        return {
            "status": status,
            "logs": logs,
            "metrics": _extract_metrics(logs),
            "artifacts": artifacts,
            "elapsed_seconds": time.time() - started_at,
            "error": None if status == "completed" else f"training process exited with code {process.returncode}",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "logs": traceback.format_exc(),
            "metrics": {},
            "artifacts": {},
            "elapsed_seconds": time.time() - started_at,
            "error": str(exc),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.function(image=image, gpu="T4", timeout=4 * 3600)
def execute_training_t4(payload: dict[str, Any]) -> dict[str, Any]:
    return _execute_training(payload)


@app.function(image=image, gpu="A10G", timeout=4 * 3600)
def execute_training_a10g(payload: dict[str, Any]) -> dict[str, Any]:
    return _execute_training(payload)


@app.function(image=image, gpu="A100-40GB", timeout=4 * 3600)
def execute_training_a100(payload: dict[str, Any]) -> dict[str, Any]:
    return _execute_training(payload)


@app.function(image=image, gpu="H100", timeout=4 * 3600)
def execute_training_h100(payload: dict[str, Any]) -> dict[str, Any]:
    return _execute_training(payload)


@app.local_entrypoint()
def main() -> None:
    print(json.dumps({"app_name": APP_NAME, "functions": [
        "execute_training_t4",
        "execute_training_a10g",
        "execute_training_a100",
        "execute_training_h100",
    ]}, ensure_ascii=False, indent=2))
