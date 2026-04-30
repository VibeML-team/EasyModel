#!/usr/bin/env python3
"""
VibeML Modal GPU Training Functions

参考 Synapse-agent 后端 modal_functions.py 的做法：
- 这个文件**完全独立**，不 import 项目里的任何模块
- 只通过 `modal deploy backend/codegen/deploy_modal.py` 部署到 Modal 云端
- 客户端通过 `modal.Function.from_name("vibeml-training", "execute_training_a100")` 远程调用

部署后的 4 个 GPU 函数共享同一份训练 driver，区别只在装饰器的 gpu= 参数。

接收参数：
    program_files: dict[str, str]    -- {"model.py": "...", "loss.py": "...", ...}
    config:        dict[str, Any]    -- 当前 BO trial 的超参组合
    dataset:       dict | None       -- {"mode": "inline", "dataset_id": "...", "zip_b64": "..."}
                                         或 {"mode": "volume", "dataset_id": "..."} （挂载已 push 的 Volume）
                                         或 None（用例不需要外部数据）
    timeout_sec:   int               -- 训练 driver 的 wall-clock 限制
    target_metric: str               -- 用于 FINAL_RESULT 解析提示

返回：
    {
        "status": "success" | "failed",
        "metrics": {...},
        "logs": "...",
        "duration_sec": float,
        "error": str | None,
    }
"""

from __future__ import annotations

import modal


APP_NAME = "vibeml-training"

app = modal.App(APP_NAME)


# ========== 镜像 ==========
# 装机一次，所有 GPU 函数共享。
# 注意：每次镜像内容变化都会触发重建（3-10 分钟），
# 所以这里把 "QA pipeline 允许的 ML 库全集" 一次性放进去，
# 后续生成的训练代码就不会因为缺包而 ModuleNotFoundError。
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "wget",
        "ffmpeg",
        "libsndfile1",
        "libgl1",
    )
    .pip_install(
        # 核心 ML
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
        # numpy 系
        "numpy>=1.24,<2",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "pyyaml>=6.0",
        # CV
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "albumentations>=1.3",
        "timm>=0.9",
        # NLP / Transformer
        "transformers>=4.40",
        "tokenizers>=0.15",
        "sentencepiece>=0.1.99",
        "einops>=0.7",
        # 音频
        "librosa>=0.10",
        "soundfile>=0.12",
        # 训练辅助
        "tqdm>=4.66",
        "matplotlib>=3.7",
        "tensorboard>=2.14",
        # 表格 / boost
        "xgboost>=2.0",
        "lightgbm>=4.1",
    )
)


# ========== 数据 Volume（共享）==========
# 客户端可以预先把大数据集 push 到这个 Volume，训练函数挂载读取。
# 小数据走 inline base64 路径，不动 Volume。
DATA_VOLUME_NAME = "vibeml-datasets"
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
DATA_VOLUME_MOUNT = "/datasets"


# ========== 训练 driver（在 Modal 容器内跑）==========

_DRIVER_TEMPLATE = r'''
"""Auto-generated training driver. Runs inside Modal GPU container."""
import json
import os
import sys
import time
import traceback

# 1. 让 Python 能 import 训练程序的 5 个文件
WORKDIR = {workdir!r}
DATA_DIR = {data_dir!r}
CONFIG = {config!r}
TARGET_METRIC = {target_metric!r}

sys.path.insert(0, WORKDIR)
os.environ["VIBEML_WORKDIR"] = WORKDIR
os.environ["VIBEML_DATA_DIR"] = DATA_DIR or ""
os.chdir(DATA_DIR or WORKDIR)

print(f"=== Driver start, workdir={{WORKDIR}}, data_dir={{DATA_DIR}} ===", flush=True)
print(f"=== Config: {{json.dumps(CONFIG, ensure_ascii=False)}} ===", flush=True)

start = time.time()
final_payload = {{}}

try:
    # 2. import 用户的代码
    import torch  # noqa: F401  必须先 import torch 让 cuda 初始化
    from train_loop import Trainer  # type: ignore

    # 3. 实例化 Trainer 并跑训练
    #    Trainer 的 __init__ 签名由 LLM 决定，我们只能 best-effort：
    #    优先尝试 Trainer(**config)，失败再尝试 Trainer()，再 .fit(**config)
    trainer = None
    init_err = None
    try:
        trainer = Trainer(**CONFIG)
    except TypeError as e:
        init_err = e
        try:
            trainer = Trainer()
        except Exception as e2:
            raise RuntimeError(
                f"Trainer.__init__ 失败: 带参 -> {{init_err}}; 无参 -> {{e2}}"
            ) from e2
    except Exception:
        raise

    # 4. fit
    fit_kwargs = {{}}
    if init_err is not None:
        # 走的是 Trainer() 无参路径，把 config 透传给 fit
        fit_kwargs = CONFIG

    result = trainer.fit(**fit_kwargs)

    # 5. 规范化结果
    if isinstance(result, dict):
        metrics = {{k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))}}
    elif isinstance(result, (int, float)):
        metrics = {{TARGET_METRIC: float(result)}}
    elif result is None:
        # 训练函数可能不返回，从 trainer 上找
        metrics = {{}}
        for attr in ("final_metrics", "metrics", "best_metrics"):
            v = getattr(trainer, attr, None)
            if isinstance(v, dict):
                metrics.update({{k: x for k, x in v.items() if isinstance(x, (int, float, str, bool))}})
                break
    else:
        metrics = {{"raw_result": str(result)}}

    final_payload = {{
        "status": "success",
        "metrics": metrics,
        "duration_sec": time.time() - start,
        "error": None,
    }}

except Exception as e:
    final_payload = {{
        "status": "failed",
        "metrics": {{}},
        "duration_sec": time.time() - start,
        "error": f"{{type(e).__name__}}: {{e}}",
        "traceback": traceback.format_exc(),
    }}

print("FINAL_RESULT:" + json.dumps(final_payload, ensure_ascii=False), flush=True)
'''


def _materialize_program(program_files: dict, dest_dir: str) -> None:
    """把 LLM 生成的多文件代码包写到 dest_dir。"""
    import os
    os.makedirs(dest_dir, exist_ok=True)
    for name, content in program_files.items():
        if not isinstance(name, str) or not isinstance(content, str):
            continue
        # 防御：不允许写绝对路径或 ../ 跳出
        safe_name = name.lstrip("/").replace("..", "_")
        path = os.path.join(dest_dir, safe_name)
        os.makedirs(os.path.dirname(path) or dest_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def _materialize_dataset(dataset: dict | None) -> str | None:
    """根据 dataset payload 准备数据目录，返回 data_dir 路径。

    支持两种模式：
    - inline: zip_b64 + dataset_id，解压到 /tmp/dataset/<id>/
    - volume: dataset_id（必须由客户端预先 push 到 Modal Volume）

    返回 None 表示训练不需要外部数据集。
    """
    import base64
    import io
    import os
    import zipfile

    if not dataset:
        return None

    mode = dataset.get("mode")
    dataset_id = dataset.get("dataset_id") or "default"

    if mode == "inline":
        zip_b64 = dataset.get("zip_b64") or ""
        if not zip_b64:
            return None
        target = f"/tmp/dataset/{dataset_id}"
        os.makedirs(target, exist_ok=True)
        raw = base64.b64decode(zip_b64.encode("ascii"))
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            zf.extractall(target)
        print(f"=== Dataset (inline) materialized at {target} "
              f"({len(raw)/1024/1024:.1f} MB) ===", flush=True)
        return target

    if mode == "volume":
        # 已经通过 modal.Volume 挂载到 DATA_VOLUME_MOUNT，
        # 客户端 push 时把数据放到 <mount>/<dataset_id>/ 下
        target = os.path.join(DATA_VOLUME_MOUNT, dataset_id)
        if not os.path.isdir(target):
            print(f"=== WARNING: volume dataset dir {target} not found ===", flush=True)
        else:
            print(f"=== Dataset (volume) mounted at {target} ===", flush=True)
        return target

    return None


def _execute_training(
    program_files: dict,
    config: dict | None = None,
    dataset: dict | None = None,
    target_metric: str = "val_loss",
    timeout_sec: int = 3600,
) -> dict:
    """通用训练执行逻辑（被所有 GPU 函数复用）。

    在 Modal GPU 容器内执行：
    1. 把生成的 5 个代码文件写到 /workspace/program/
    2. 解压/挂载数据集
    3. 生成 driver 脚本，subprocess 跑训练
    4. 解析 FINAL_RESULT 行，返回结构化结果
    """
    import os
    import subprocess
    import tempfile
    import time as time_module

    config = config or {}
    start = time_module.time()

    # 1. 准备 workdir
    workdir = tempfile.mkdtemp(prefix="vibeml_program_")
    _materialize_program(program_files, workdir)

    # 2. 准备 dataset
    data_dir = _materialize_dataset(dataset)

    # 3. 写 driver
    driver_src = _DRIVER_TEMPLATE.format(
        workdir=workdir,
        data_dir=data_dir or "",
        config=config,
        target_metric=target_metric,
    )
    driver_path = os.path.join(workdir, "_driver.py")
    with open(driver_path, "w", encoding="utf-8") as f:
        f.write(driver_src)

    # 4. 跑 driver
    import sys as _sys
    proc = subprocess.Popen(
        [_sys.executable, driver_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=workdir,
        text=True,
        bufsize=1,
    )

    log_chunks: list[str] = []
    final_payload: dict | None = None
    deadline = start + max(timeout_sec, 60)

    try:
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            log_chunks.append(line)
            # 实时也打到 Modal 控制台，便于在 modal logs 里看
            print(line, end="", flush=True)
            if line.startswith("FINAL_RESULT:"):
                try:
                    import json as _json
                    final_payload = _json.loads(line[len("FINAL_RESULT:"):].strip())
                except Exception:
                    final_payload = None
            if time_module.time() > deadline:
                proc.kill()
                log_chunks.append(f"\n=== Killed by VibeML driver timeout ({timeout_sec}s) ===\n")
                break
        proc.wait(timeout=10)
    except Exception as e:
        log_chunks.append(f"\n=== Driver supervisor crashed: {e} ===\n")
        proc.kill()

    elapsed = time_module.time() - start
    full_logs = "".join(log_chunks)

    if final_payload is None:
        return {
            "status": "failed",
            "metrics": {},
            "duration_sec": elapsed,
            "error": "Driver did not emit FINAL_RESULT (possibly crashed or timed out)",
            "logs": full_logs[-50_000:],  # 保留最后 50KB
        }

    final_payload.setdefault("logs", full_logs[-50_000:])
    final_payload.setdefault("duration_sec", elapsed)
    return final_payload


# ========== 4 个 GPU 函数（每种 GPU 一个全局函数）==========

_COMMON_FN_KW = dict(
    image=image,
    timeout=7200,
    volumes={DATA_VOLUME_MOUNT: data_volume},
)


@app.function(gpu="T4", **_COMMON_FN_KW)
def execute_training_t4(
    program_files: dict,
    config: dict | None = None,
    dataset: dict | None = None,
    target_metric: str = "val_loss",
    timeout_sec: int = 3600,
) -> dict:
    """T4 GPU 上跑生成的训练程序。"""
    return _execute_training(program_files, config, dataset, target_metric, timeout_sec)


@app.function(gpu="A10G", **_COMMON_FN_KW)
def execute_training_a10g(
    program_files: dict,
    config: dict | None = None,
    dataset: dict | None = None,
    target_metric: str = "val_loss",
    timeout_sec: int = 3600,
) -> dict:
    """A10G GPU 上跑生成的训练程序。"""
    return _execute_training(program_files, config, dataset, target_metric, timeout_sec)


@app.function(gpu="A100-40GB", **_COMMON_FN_KW)
def execute_training_a100(
    program_files: dict,
    config: dict | None = None,
    dataset: dict | None = None,
    target_metric: str = "val_loss",
    timeout_sec: int = 3600,
) -> dict:
    """A100-40GB GPU 上跑生成的训练程序。"""
    return _execute_training(program_files, config, dataset, target_metric, timeout_sec)


@app.function(gpu="H100", **_COMMON_FN_KW)
def execute_training_h100(
    program_files: dict,
    config: dict | None = None,
    dataset: dict | None = None,
    target_metric: str = "val_loss",
    timeout_sec: int = 3600,
) -> dict:
    """H100 GPU 上跑生成的训练程序。"""
    return _execute_training(program_files, config, dataset, target_metric, timeout_sec)


# ========== QA 用的 CPU 函数 ==========
# QA pipeline 的「单元测试」和「沙箱冒烟测试」需要 import torch / cv2 / transformers
# 之类的 ML 依赖才能跑起来。Zeabur 后端容器为了控制镜像体积没装这些，
# 因此把 QA 也丢到 Modal 容器里执行（共享同一份 image，但走 CPU 函数，~$0.0001/s）。
#
# 函数协议：
#   files: {filename -> source}        必须包含 entry，可包含 model.py / loss.py / 测试运行器等
#   entry: 入口脚本名（不允许 .. 跳出）
#   timeout_sec: 子进程级 wall-clock 上限
#
# 返回：
#   {"returncode": int, "stdout": str, "stderr": str,
#    "duration_sec": float, "timed_out": bool}


@app.function(image=image, timeout=900, cpu=2.0, memory=4096)
def qa_run_python(
    files: dict | None = None,
    entry: str = "run_tests.py",
    timeout_sec: int = 180,
) -> dict:
    """在 Modal 容器内（CPU）跑一个 Python 入口脚本。

    用于 QA 阶段的单元测试 + 烟测：本地 Zeabur 容器没装 torch，把脚本扔进
    Modal 镜像里执行，复用同一份训练镜像，免镜像复制。
    """
    import os
    import subprocess
    import sys as _sys
    import tempfile
    import time as _time
    import traceback as _tb

    files = files or {}
    if entry not in files:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"entry script {entry!r} not found in files",
            "duration_sec": 0.0,
            "timed_out": False,
        }

    workdir = tempfile.mkdtemp(prefix="vibeml_qa_")
    for name, content in files.items():
        if not isinstance(name, str) or not isinstance(content, str):
            continue
        safe = name.lstrip("/").replace("..", "_")
        path = os.path.join(workdir, safe)
        os.makedirs(os.path.dirname(path) or workdir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU 函数，禁用 GPU 探测
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)

    start = _time.time()
    try:
        proc = subprocess.run(
            [_sys.executable, entry],
            capture_output=True,
            text=True,
            cwd=workdir,
            env=env,
            timeout=max(int(timeout_sec or 180), 30),
        )
        return {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-200_000:],
            "stderr": (proc.stderr or "")[-200_000:],
            "duration_sec": _time.time() - start,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": -1,
            "stdout": ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or ""))[-200_000:],
            "stderr": ((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or ""))[-200_000:],
            "duration_sec": _time.time() - start,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}\n{_tb.format_exc()}",
            "duration_sec": _time.time() - start,
            "timed_out": False,
        }


# ========== 数据集 push 辅助函数（仅本地用，不部署 GPU 函数版）==========
# 客户端用 push_dataset_to_volume 把本地 backend/data/<id> 同步到 Volume。
# 这是一个 CPU 辅助函数，不需要 GPU。


@app.function(image=image, timeout=1800, volumes={DATA_VOLUME_MOUNT: data_volume})
def push_dataset_files(dataset_id: str, files: list[dict]) -> dict:
    """把本地数据集文件批量写入 Modal Volume。

    files: 列表，每个元素 {"path": "subpath/relative", "content_b64": "..."}
    """
    import base64
    import os

    target_root = os.path.join(DATA_VOLUME_MOUNT, dataset_id)
    os.makedirs(target_root, exist_ok=True)

    written = 0
    total_bytes = 0
    for item in files:
        rel = item.get("path", "").lstrip("/")
        if not rel or ".." in rel.split("/"):
            continue
        target = os.path.join(target_root, rel)
        os.makedirs(os.path.dirname(target) or target_root, exist_ok=True)
        raw = base64.b64decode(item.get("content_b64", "").encode("ascii"))
        with open(target, "wb") as f:
            f.write(raw)
        written += 1
        total_bytes += len(raw)

    data_volume.commit()  # 持久化

    return {"dataset_id": dataset_id, "files_written": written, "bytes": total_bytes}
