"""ModalSandboxExecutor 单元测试

不需要真的连 Modal —— 我们 mock 掉 modal SDK，验证：
1. GPU 类型 → 函数名映射
2. GeneratedProgram 多文件被正确打包成 dict
3. 数据集 inline pack：体积小走 inline；体积超阈值走 volume
4. 数据集会被缓存（同 dataset 多次调用不重复 zip）
5. spawn 失败、call.get 失败时返回 status=failed 的 dict（接口与本地 SandboxExecutor 一致）
6. make_executor_from_env：USE_MODAL 控制后端
7. 训练结果解析：metrics / duration / logs 都 surface 到上层
"""

from __future__ import annotations

import sys
import types
import zipfile
import io
import base64
from pathlib import Path
from typing import Any
from unittest import mock

import pytest


# ----------------- 在 import ModalSandboxExecutor 前先 stub modal SDK ---------

class _FakeModalCall:
    def __init__(self, result: Any = None, exc: BaseException | None = None):
        self._result = result
        self._exc = exc

    def get(self, timeout: float | None = None):  # noqa: ARG002
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeModalFunction:
    def __init__(self, name: str):
        self.name = name
        self.spawn_calls: list[tuple] = []
        self.next_result: Any = {
            "status": "success",
            "metrics": {"val_loss": 0.123},
            "duration_sec": 12.3,
            "logs": "fake logs",
        }
        self.next_exception: BaseException | None = None

    def spawn(self, *args, **kwargs):  # noqa: ARG002
        self.spawn_calls.append(args)
        return _FakeModalCall(self.next_result, self.next_exception)

    def remote(self, *args, **kwargs):  # noqa: ARG002
        return {"ok": True}


_FAKE_FUNCTIONS: dict[tuple[str, str], _FakeModalFunction] = {}


def _fake_from_name(app_name: str, fn_name: str):
    key = (app_name, fn_name)
    if key not in _FAKE_FUNCTIONS:
        _FAKE_FUNCTIONS[key] = _FakeModalFunction(fn_name)
    return _FAKE_FUNCTIONS[key]


def _install_fake_modal_module():
    """Install a minimal `modal` module in sys.modules so importing executor works."""
    if "modal" in sys.modules and getattr(sys.modules["modal"], "_vibeml_fake", False):
        return

    modal_mod = types.ModuleType("modal")
    modal_mod._vibeml_fake = True  # type: ignore[attr-defined]

    class _FunctionNS:
        from_name = staticmethod(_fake_from_name)

    modal_mod.Function = _FunctionNS  # type: ignore[attr-defined]

    # exception namespace
    exc_mod = types.ModuleType("modal.exception")

    class FunctionTimeoutError(Exception):
        pass

    class NotFoundError(Exception):
        pass

    exc_mod.FunctionTimeoutError = FunctionTimeoutError  # type: ignore[attr-defined]
    exc_mod.NotFoundError = NotFoundError  # type: ignore[attr-defined]
    modal_mod.exception = exc_mod  # type: ignore[attr-defined]
    sys.modules["modal.exception"] = exc_mod
    sys.modules["modal"] = modal_mod


_install_fake_modal_module()


from backend.codegen.modal_executor import (  # noqa: E402
    GPU_FUNCTION_MAP,
    ModalSandboxExecutor,
    make_executor_from_env,
)


# ------------- 测试用的最小 GeneratedProgram stub ----------------------------


class _StubProgram:
    def __init__(self, model="m", loss="l", data="d", train="t", search_space=None):
        self.model_code = model
        self.loss_code = loss
        self.data_pipeline_code = data
        self.train_loop_code = train
        self.search_space = search_space or {"lr": {"type": "float", "low": 1e-4, "high": 1e-2}}


@pytest.fixture(autouse=True)
def _reset_fake_functions():
    _FAKE_FUNCTIONS.clear()
    yield
    _FAKE_FUNCTIONS.clear()


# ------------- tests --------------------------------------------------------


def test_gpu_function_mapping_known_types():
    for gpu in ["T4", "A10G", "A100", "H100"]:
        ex = ModalSandboxExecutor(gpu_type=gpu)
        assert ex._function_name() == GPU_FUNCTION_MAP[gpu]


def test_gpu_function_mapping_normalizes_aliases_and_unknown():
    assert ModalSandboxExecutor(gpu_type="a100-40gb")._function_name() == "execute_training_a100"
    assert ModalSandboxExecutor(gpu_type="MI300X")._function_name() == "execute_training_a100"


def test_program_files_extracted_to_dict():
    ex = ModalSandboxExecutor(gpu_type="A100")
    files = ex._extract_program_files(_StubProgram(
        model="MODEL_SRC", loss="LOSS_SRC", data="DATA_SRC", train="TRAIN_SRC",
        search_space={"lr": {"type": "float", "low": 1e-4, "high": 1e-2}},
    ))
    assert files["model.py"] == "MODEL_SRC"
    assert files["loss.py"] == "LOSS_SRC"
    assert files["data_pipeline.py"] == "DATA_SRC"
    assert files["train_loop.py"] == "TRAIN_SRC"
    assert "lr" in files["search_space.yaml"]


def test_dataset_inline_pack_small(tmp_path: Path):
    ds_dir = tmp_path / "ds1"
    ds_dir.mkdir()
    (ds_dir / "a.csv").write_text("col\n1\n2\n")
    (ds_dir / "sub").mkdir()
    (ds_dir / "sub" / "b.txt").write_text("hello")

    ex = ModalSandboxExecutor(
        gpu_type="A100",
        dataset_id="ds1",
        dataset_root=ds_dir,
        inline_pack_limit_mb=1,  # 数据小，必走 inline
    )
    payload = ex._build_dataset_payload()
    assert payload is not None
    assert payload["mode"] == "inline"
    assert payload["dataset_id"] == "ds1"

    # zip 能解开，且包含相对路径
    raw = base64.b64decode(payload["zip_b64"].encode("ascii"))
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        names = set(zf.namelist())
    assert "a.csv" in names
    assert "sub/b.txt" in names


def test_dataset_inline_cached_across_calls(tmp_path: Path):
    ds_dir = tmp_path / "ds_cached"
    ds_dir.mkdir()
    (ds_dir / "x.txt").write_text("x" * 100)

    ex = ModalSandboxExecutor(
        gpu_type="A100",
        dataset_id="ds_cached",
        dataset_root=ds_dir,
        inline_pack_limit_mb=1,
    )
    p1 = ex._build_dataset_payload()
    p2 = ex._build_dataset_payload()
    # 同一对象被复用（cache 命中），不会重新 zip
    assert p1 is p2


def test_dataset_volume_when_too_big(tmp_path: Path):
    ds_dir = tmp_path / "ds_big"
    ds_dir.mkdir()
    # 写一个 ~512KB 的文件
    (ds_dir / "big.bin").write_bytes(b"\x00" * (512 * 1024))

    ex = ModalSandboxExecutor(
        gpu_type="A100",
        dataset_id="ds_big",
        dataset_root=ds_dir,
        inline_pack_limit_mb=0,  # 关掉 inline → 必走 volume
    )
    payload = ex._build_dataset_payload()
    assert payload == {"mode": "volume", "dataset_id": "ds_big"}


def test_dataset_payload_none_without_dataset_id():
    ex = ModalSandboxExecutor(gpu_type="A100")
    assert ex._build_dataset_payload() is None


def test_execute_training_happy_path(tmp_path: Path):
    ex = ModalSandboxExecutor(
        gpu_type="A100",
        dataset_id="ds_happy",
        dataset_root=_make_tiny_dataset(tmp_path / "ds_happy"),
        inline_pack_limit_mb=1,
        target_metric="val_acc",
    )

    fn = _fake_from_name("vibeml-training", "execute_training_a100")
    fn.next_result = {
        "status": "success",
        "metrics": {"val_acc": 0.91, "val_loss": 0.21},
        "duration_sec": 42.5,
        "logs": "trained ok",
    }

    out = ex.execute_training(_StubProgram(), {"lr": 1e-3, "batch_size": 32}, timeout=600)

    # 接口字段与本地 SandboxExecutor 一致
    assert out["status"] == "success"
    assert out["metrics"] == {"val_acc": 0.91, "val_loss": 0.21}
    assert out["duration_sec"] == 42.5
    assert out["duration_ms"] == 42500.0
    assert "logs" in out

    # spawn 收到的实参符合 modal_functions 的签名顺序
    assert len(fn.spawn_calls) == 1
    args = fn.spawn_calls[0]
    program_files, config, dataset_payload, target_metric, timeout_sec = args
    assert set(program_files.keys()) >= {"model.py", "loss.py", "data_pipeline.py", "train_loop.py", "search_space.yaml"}
    assert config == {"lr": 1e-3, "batch_size": 32}
    assert dataset_payload["mode"] == "inline"
    assert dataset_payload["dataset_id"] == "ds_happy"
    assert target_metric == "val_acc"
    assert timeout_sec == 600


def test_execute_training_returns_failed_dict_on_spawn_error():
    ex = ModalSandboxExecutor(gpu_type="A100")
    fn = _fake_from_name("vibeml-training", "execute_training_a100")
    fn.next_exception = RuntimeError("boom")

    out = ex.execute_training(_StubProgram(), {"lr": 1e-3}, timeout=10)
    assert out["status"] == "failed"
    assert "boom" in out["error"]
    assert out["metrics"] == {}


def test_execute_training_handles_non_dict_modal_result():
    ex = ModalSandboxExecutor(gpu_type="A100")
    fn = _fake_from_name("vibeml-training", "execute_training_a100")
    fn.next_result = "not a dict"

    out = ex.execute_training(_StubProgram(), {}, timeout=10)
    assert out["status"] == "failed"
    assert "non dict" in out["error"] or "non-dict" in out["error"] or "non dict" in out["error"].lower() or "str" in out["error"]


def test_function_lookup_failure_surfaces_friendly_error():
    ex = ModalSandboxExecutor(gpu_type="A100")
    with mock.patch.object(ex, "_get_modal_function", side_effect=RuntimeError("not deployed")):
        out = ex.execute_training(_StubProgram(), {}, timeout=10)
    assert out["status"] == "failed"
    assert "modal deploy" in out["error"]
    assert "MODAL_TOKEN" in out["error"]


def test_estimate_cost_usd_known_gpus():
    ex = ModalSandboxExecutor(gpu_type="A100")
    # 1 小时 = 3600 sec → $0.60
    assert abs(ex.estimate_cost_usd(3600) - 0.60) < 1e-9
    ex2 = ModalSandboxExecutor(gpu_type="H100")
    assert abs(ex2.estimate_cost_usd(3600) - 1.20) < 1e-9


def test_make_executor_from_env_local(monkeypatch):
    monkeypatch.delenv("USE_MODAL", raising=False)
    ex = make_executor_from_env(dataset_id="x", dataset_root="/tmp")
    assert ex.__class__.__name__ == "SandboxExecutor"


def test_make_executor_from_env_modal(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("USE_MODAL", "1")
    monkeypatch.setenv("MODAL_GPU", "T4")
    ds_dir = _make_tiny_dataset(tmp_path / "envds")
    ex = make_executor_from_env(dataset_id="envds", dataset_root=ds_dir)
    assert isinstance(ex, ModalSandboxExecutor)
    assert ex.gpu_type == "T4"
    assert ex.dataset_id == "envds"


# ----------------- QA / run_python tests ------------------------------------


def test_run_python_routes_to_qa_function():
    """run_python 应该解析 qa_run_python 而不是 GPU 函数。"""
    ex = ModalSandboxExecutor(gpu_type="A100")
    qa_fn = _fake_from_name("vibeml-training", "qa_run_python")
    qa_fn.next_result = {
        "returncode": 0,
        "stdout": "TEST_OK",
        "stderr": "",
        "duration_sec": 1.5,
        "timed_out": False,
    }

    files = {
        "run_tests.py": "print('hi')\n",
        "model.py": "x = 1\n",
    }
    out = ex.run_python(files, entry="run_tests.py", timeout_sec=60)

    # qa_run_python 收到了正确入参
    assert len(qa_fn.spawn_calls) == 1
    sent_files, sent_entry, sent_timeout = qa_fn.spawn_calls[0]
    assert sent_entry == "run_tests.py"
    assert sent_timeout == 60
    assert sent_files["run_tests.py"] == "print('hi')\n"
    assert sent_files["model.py"] == "x = 1\n"

    # 透传字段都在
    assert out["returncode"] == 0
    assert out["stdout"] == "TEST_OK"
    assert out["timed_out"] is False
    assert out["duration_sec"] == 1.5


def test_run_python_handles_modal_failure():
    ex = ModalSandboxExecutor(gpu_type="T4")
    qa_fn = _fake_from_name("vibeml-training", "qa_run_python")
    qa_fn.next_exception = RuntimeError("modal down")

    out = ex.run_python({"run_tests.py": "pass\n"}, entry="run_tests.py", timeout_sec=10)
    assert out["returncode"] == -1
    assert "modal down" in out["stderr"]
    assert out["timed_out"] is False


def test_run_python_rejects_missing_entry():
    ex = ModalSandboxExecutor(gpu_type="A100")
    with pytest.raises(ValueError, match="不在 files 中"):
        ex.run_python({"model.py": "..."}, entry="run_tests.py")


def test_smoke_test_runs_each_stage_via_qa_function():
    """smoke_test 应该把 5 个 stage 都通过 qa_run_python 跑一遍。"""
    from backend.codegen.sandbox import SMOKE_SCRIPTS  # noqa: PLC0415

    ex = ModalSandboxExecutor(gpu_type="A10G")
    qa_fn = _fake_from_name("vibeml-training", "qa_run_python")
    # 让每次调用都返回成功（要带 _SUCCESS 关键字）
    qa_fn.next_result = {
        "returncode": 0,
        "stdout": "IMPORT_SUCCESS: 10ms",
        "stderr": "",
        "duration_sec": 0.5,
        "timed_out": False,
    }

    program = _StubProgram(
        model="class M:\n    pass\n",
        loss="class MyLoss:\n    pass\n",
        data="class DM:\n    pass\n",
        train="x = 1\n",
    )
    results = ex.smoke_test(program)

    assert len(results) == len(SMOKE_SCRIPTS)
    # 每个调用都用了不同的 entry：test_<stage>.py
    entries = [call[1] for call in qa_fn.spawn_calls]
    expected = {f"test_{stage}.py" for stage in SMOKE_SCRIPTS}
    assert set(entries) == expected

    # 每次发送的 files 至少包含 4 个程序文件 + 1 个测试入口
    for files, entry, _to in qa_fn.spawn_calls:
        assert entry in files
        for required in ("model.py", "loss.py", "data_pipeline.py", "train_loop.py"):
            assert required in files

    # 全部 passed=True
    assert all(r.passed for r in results)


def test_smoke_test_marks_failed_stage_when_modal_returns_error():
    ex = ModalSandboxExecutor(gpu_type="T4")
    qa_fn = _fake_from_name("vibeml-training", "qa_run_python")
    qa_fn.next_result = {
        "returncode": 1,
        "stdout": "IMPORT_FAILED",
        "stderr": "ModuleNotFoundError: No module named 'torch'",
        "duration_sec": 0.2,
        "timed_out": False,
    }

    program = _StubProgram()
    results = ex.smoke_test(program)
    assert all(not r.passed for r in results)
    assert any("torch" in (r.error_message or "") for r in results)


def test_run_python_via_unit_test_generator():
    """UnitTestGenerator.run_tests(sandbox=Modal) 应该把测试运行器扔进 Modal。"""
    from backend.codegen.test_generator import (  # noqa: PLC0415
        TestSuite,
        UnitTestGenerator,
    )

    gen = UnitTestGenerator()
    suite = TestSuite(
        test_cases=[],
        test_runner_code="import sys\nprint('ok')\nsys.exit(0)\n",
        coverage_targets={},
    )

    ex = ModalSandboxExecutor(gpu_type="A100")
    qa_fn = _fake_from_name("vibeml-training", "qa_run_python")
    qa_fn.next_result = {
        "returncode": 0,
        "stdout": "ok\n",
        "stderr": "",
        "duration_sec": 2.0,
        "timed_out": False,
    }

    program = _StubProgram(model="m", loss="l", data="d", train="t")
    results = gen.run_tests(suite, program, timeout=30, sandbox=ex)
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].duration_ms == 2000.0

    # 单元测试 runner 被发到 qa_run_python
    assert len(qa_fn.spawn_calls) == 1
    files, entry, _to = qa_fn.spawn_calls[0]
    assert entry == "run_tests.py"
    assert "import sys" in files["run_tests.py"]


def test_unit_test_generator_propagates_modal_failure_message():
    from backend.codegen.test_generator import (  # noqa: PLC0415
        TestSuite,
        UnitTestGenerator,
    )

    gen = UnitTestGenerator()
    suite = TestSuite(
        test_cases=[],
        test_runner_code="raise SystemExit(1)\n",
        coverage_targets={},
    )

    ex = ModalSandboxExecutor(gpu_type="A100")
    qa_fn = _fake_from_name("vibeml-training", "qa_run_python")
    qa_fn.next_result = {
        "returncode": 1,
        "stdout": "",
        "stderr": "AssertionError: shapes mismatch",
        "duration_sec": 1.1,
        "timed_out": False,
    }

    program = _StubProgram()
    results = gen.run_tests(suite, program, timeout=30, sandbox=ex)
    assert len(results) == 1
    assert results[0].passed is False
    assert "shapes mismatch" in results[0].error_message


# ----------------- helpers --------------------------------------------------


def _make_tiny_dataset(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "a.txt").write_text("hello")
    return path
