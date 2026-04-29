"""Regression tests for the auto-generated test runner.

历史问题：
- 旧 runner 把每个 test case 的源码 ``exec('''…''')`` 在模块顶层串接执行，
  导致命名空间互相污染（典型症状："name 'sys' is not defined"）。
- 一旦运行环境没装 pytest，``import pytest`` 就在 setup 阶段直接抛
  ``ModuleNotFoundError``，把 5 个生成测试一并搞挂（这就是 Zeabur 上看到的
  "QA failed after 3 attempts: setup failed: No module named 'pytest'"）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from backend.codegen.test_generator import TestCase, UnitTestGenerator


def _write_dummy_program(td: Path) -> None:
    (td / "model.py").write_text("class Dummy:\n    pass\n")
    (td / "loss.py").write_text("class Dummy:\n    pass\n")
    (td / "data_pipeline.py").write_text("class Dummy:\n    pass\n")
    (td / "train_loop.py").write_text("class Dummy:\n    pass\n")


def test_test_runner_uses_isolated_namespace_per_case():
    """每个 case 必须跑独立 namespace，sys/torch 不能被上一轮污染。"""
    gen = UnitTestGenerator()
    cases = [
        TestCase(
            name="t1",
            code="def test_a():\n    assert sys.version_info.major >= 3\n",
            target_file="model.py",
            description="references sys without explicitly importing",
        ),
        TestCase(
            name="t2",
            code="def test_b():\n    import torch as _t\n    assert torch is _t\n",
            target_file="model.py",
            description="reuses pre-injected torch",
        ),
    ]
    runner = gen._generate_test_runner(cases)

    # 关键证据：runner 必须用 _run_one(name, src) 而不是 module-level exec
    assert "_run_one(" in runner
    assert "exec('''" not in runner
    # namespace 必须预注入 sys / torch / pytest
    assert "_BASE_TEST_GLOBALS" in runner
    assert '"sys": sys' in runner
    assert '"torch": torch' in runner
    assert '"pytest": pytest' in runner

    # 跑一遍 runner 真实验证：sys 不会"未定义"，两个 case 都 PASS
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _write_dummy_program(td_path)
        runner_path = td_path / "run_tests.py"
        runner_path.write_text(runner)
        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True, text=True, timeout=60, cwd=str(td_path),
        )
        assert proc.returncode == 0, (
            f"runner 不应失败；stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        assert "[FAIL]" not in proc.stdout
        assert "[PASS] t1" in proc.stdout
        assert "[PASS] t2" in proc.stdout


def test_test_runner_falls_back_to_stub_pytest_when_missing():
    """容器里没装 pytest 时，runner 也得能跑：自动塞一份 no-op stub 进 sys.modules。

    这是为了兜住 Zeabur 镜像没装 pytest 的过渡期 —— 历史问题是
    ``import pytest`` 直接抛 ModuleNotFoundError，把全部 5 个生成测试搞挂。
    """
    gen = UnitTestGenerator()
    cases = [
        TestCase(
            name="needs_pytest",
            code=(
                "import pytest\n"
                "import torch\n"
                "\n"
                "@pytest.mark.skipif(False, reason='nope')\n"
                "def test_decorator_works():\n"
                "    assert torch.tensor(1).item() == 1\n"
                "\n"
                "def test_raises_works():\n"
                "    with pytest.raises(ValueError):\n"
                "        raise ValueError('expected')\n"
            ),
            target_file="model.py",
            description="needs pytest decorators / context managers",
        ),
    ]
    runner = gen._generate_test_runner(cases)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _write_dummy_program(td_path)
        runner_path = td_path / "run_tests.py"
        runner_path.write_text(runner)

        # 关键：构造一个 sitecustomize.py 完全屏蔽 pytest，
        # 模拟 Zeabur / Modal 容器没装 pytest 的环境。
        (td_path / "sitecustomize.py").write_text(
            "import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'pytest':\n"
            "            raise ImportError('blocked for test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
        )

        env = {**os.environ, "PYTHONPATH": str(td_path)}
        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True, text=True, timeout=60,
            cwd=str(td_path), env=env,
        )

        assert proc.returncode == 0, (
            f"缺 pytest 时 runner 不应失败；"
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        assert "[FAIL]" not in proc.stdout, proc.stdout
        assert "No module named 'pytest'" not in proc.stdout
        assert "[PASS] needs_pytest" in proc.stdout


def test_test_runner_emits_setup_failed_prefix_for_import_errors():
    """生成测试源码 import 不存在的模块时，runner 应当报告 setup failed，
    而不是把整批 case 一起拖挂。"""
    gen = UnitTestGenerator()
    cases = [
        TestCase(
            name="bad_import",
            code="import this_module_does_not_exist\n\ndef test_x():\n    assert True\n",
            target_file="model.py",
            description="import error in setup",
        ),
        TestCase(
            name="good",
            code="def test_y():\n    assert 1 + 1 == 2\n",
            target_file="model.py",
            description="a sane sibling test",
        ),
    ]
    runner = gen._generate_test_runner(cases)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _write_dummy_program(td_path)
        runner_path = td_path / "run_tests.py"
        runner_path.write_text(runner)
        proc = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True, text=True, timeout=60, cwd=str(td_path),
        )
        # bad_import 失败但 good 应当独立跑过
        assert "[FAIL] bad_import" in proc.stdout
        assert "setup failed:" in proc.stdout
        assert "[PASS] good" in proc.stdout
        # 整体 returncode 非 0（有失败）
        assert proc.returncode != 0
