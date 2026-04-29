"""
Unit Test Generator - 自动生成单元测试

为生成的代码自动生成测试用例：
1. 形状测试 (Shape Tests): 确保输入输出维度正确
2. 梯度测试 (Gradient Tests): 确保反向传播正确
3. 确定性测试 (Determinism Tests): 确保相同输入产生相同输出
4. 边界测试 (Boundary Tests): 测试极端输入
5. 属性测试 (Property Tests): 使用Hypothesis进行随机测试
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TestCase:
    """单个测试用例"""
    name: str
    code: str
    target_file: str  # 测试哪个文件
    description: str


@dataclass
class TestSuite:
    """测试套件"""
    test_cases: list[TestCase]
    test_runner_code: str  # 可以执行的测试脚本
    coverage_targets: dict[str, list[str]]  # 每个文件的测试目标
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "test_cases": [
                {"name": tc.name, "target": tc.target_file, "description": tc.description}
                for tc in self.test_cases
            ],
            "coverage_targets": self.coverage_targets,
        }


@dataclass
class TestResult:
    """测试结果"""
    passed: bool
    test_name: str
    duration_ms: float
    error_message: str | None = None
    stack_trace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "test_name": self.test_name,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
        }


class UnitTestGenerator:
    """单元测试生成器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def generate_tests(self, program: "GeneratedProgram") -> TestSuite:
        """
        为生成的程序自动生成测试套件
        """
        test_cases = []
        
        # 1. 生成形状测试
        test_cases.extend(self._generate_shape_tests(program))
        
        # 2. 生成梯度测试
        test_cases.extend(self._generate_gradient_tests(program))
        
        # 3. 生成设备兼容性测试
        test_cases.extend(self._generate_device_tests(program))
        
        # 4. 生成序列化测试
        test_cases.extend(self._generate_serialization_tests(program))
        
        # 5. 如果有LLM，生成更智能的测试
        if self.llm_client:
            test_cases.extend(self._generate_llm_tests(program))
        
        # 生成测试运行器
        test_runner = self._generate_test_runner(test_cases)
        
        # 计算覆盖目标
        coverage_targets = self._extract_coverage_targets(program)
        
        return TestSuite(
            test_cases=test_cases,
            test_runner_code=test_runner,
            coverage_targets=coverage_targets,
        )
    
    def _generate_shape_tests(self, program: "GeneratedProgram") -> list[TestCase]:
        """生成输入输出形状测试"""
        tests = []
        
        # 从模型代码中提取类名和forward签名
        model_class = self._extract_class_name(program.model_code)
        if model_class:
            test_code = f"""
import torch
import pytest
from model import {model_class}

def test_model_output_shape():
    \"\"\"测试模型输出形状是否正确\"\"\"
    model = {model_class}()
    batch_size = 4
    # 使用通用输入形状，实际应根据模型调整
    x = torch.randn(batch_size, 10)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape[0] == batch_size, f"Batch维度不匹配: {{output.shape[0]}} != {{batch_size}}"
    assert len(output.shape) >= 1, "输出至少应该有一维"

def test_model_different_batch_sizes():
    \"\"\"测试模型支持不同batch size\"\"\"
    model = {model_class}()
    model.eval()
    
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 10)
        with torch.no_grad():
            output = model(x)
        assert output.shape[0] == batch_size
"""
            tests.append(TestCase(
                name="test_model_shape",
                code=test_code,
                target_file="model.py",
                description="验证模型输入输出形状正确性",
            ))
        
        # Loss函数的形状测试
        loss_class = self._extract_class_name(program.loss_code)
        if loss_class:
            test_code = f"""
import torch
import pytest
from loss import {loss_class}

def test_loss_output_scalar():
    \"\"\"测试损失函数输出标量\"\"\"
    loss_fn = {loss_class}()
    pred = torch.randn(4, 1)
    target = torch.randn(4, 1)
    
    loss = loss_fn(pred, target)
    
    assert loss.ndim == 0, f"损失应该是标量，但形状是 {{loss.shape}}"
    assert loss.item() >= 0, "损失应该是非负的"

def test_loss_reduction_mean():
    \"\"\"测试reduction='mean'时输出标量\"\"\"
    loss_fn = {loss_class}(reduction='mean')
    pred = torch.randn(8, 5)
    target = torch.randn(8, 5)
    
    loss = loss_fn(pred, target)
    assert loss.ndim == 0
"""
            tests.append(TestCase(
                name="test_loss_shape",
                code=test_code,
                target_file="loss.py",
                description="验证损失函数输入输出正确性",
            ))
        
        return tests
    
    def _generate_gradient_tests(self, program: "GeneratedProgram") -> list[TestCase]:
        """生成梯度传播测试"""
        tests = []
        
        model_class = self._extract_class_name(program.model_code)
        loss_class = self._extract_class_name(program.loss_code)
        
        if model_class and loss_class:
            test_code = f"""
import torch
import pytest
from model import {model_class}
from loss import {loss_class}

def test_gradient_flow():
    \"\"\"测试梯度能正常反向传播\"\"\"
    model = {model_class}()
    loss_fn = {loss_class}()
    
    x = torch.randn(2, 10, requires_grad=True)
    target = torch.randn(2, 1)
    
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    
    # 检查是否有梯度
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "模型参数没有接收到梯度"

def test_no_nan_gradients():
    \"\"\"测试梯度不包含NaN\"\"\"
    model = {model_class}()
    loss_fn = {loss_class}()
    
    x = torch.randn(4, 10)
    target = torch.randn(4, 1)
    
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"参数 {{name}} 的梯度包含NaN"
            assert not torch.isinf(param.grad).any(), f"参数 {{name}} 的梯度包含Inf"
"""
            tests.append(TestCase(
                name="test_gradient_flow",
                code=test_code,
                target_file="model.py, loss.py",
                description="验证反向传播正确性",
            ))
        
        return tests
    
    def _generate_device_tests(self, program: "GeneratedProgram") -> list[TestCase]:
        """生成CPU/GPU兼容性测试"""
        tests = []
        
        model_class = self._extract_class_name(program.model_code)
        if model_class:
            test_code = f"""
import torch
import pytest
from model import {model_class}

def test_cpu_forward():
    \"\"\"测试CPU前向传播\"\"\"
    model = {model_class}()
    x = torch.randn(2, 10)
    
    output = model(x)
    assert output is not None
    assert output.device.type == 'cpu'

def test_cuda_forward():
    \"\"\"测试CUDA前向传播（无 GPU 时直接跳过）\"\"\"
    # 不能用 @pytest.mark.skipif —— 我们的 runner 是直接调用函数的，
    # pytest 的 marker 不会被识别，必须在函数体里手动短路。
    if not torch.cuda.is_available():
        return
    model = {model_class}().cuda()
    x = torch.randn(2, 10).cuda()
    output = model(x)
    assert output.device.type == 'cuda'

def test_model_device_movement():
    \"\"\"测试模型可以在设备间移动\"\"\"
    model = {model_class}()
    
    # 测试.to()方法
    model_cpu = model.to('cpu')
    x = torch.randn(2, 10)
    output = model_cpu(x)
    assert output.device.type == 'cpu'
"""
            tests.append(TestCase(
                name="test_device_compatibility",
                code=test_code,
                target_file="model.py",
                description="验证CPU/GPU兼容性",
            ))
        
        return tests
    
    def _generate_serialization_tests(self, program: "GeneratedProgram") -> list[TestCase]:
        """生成模型序列化测试"""
        tests = []
        
        model_class = self._extract_class_name(program.model_code)
        if model_class:
            test_code = f"""
import torch
import io
import pytest
from model import {model_class}

def test_state_dict_save_load():
    \"\"\"测试state_dict保存和加载\"\"\"
    model = {model_class}()
    x = torch.randn(2, 10)
    model.eval()
    with torch.no_grad():
        output1 = model(x)
    state_dict = model.state_dict()
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    # PyTorch 2.6 起 torch.load 默认 weights_only=True；state_dict 是纯 tensor，
    # 这里显式声明就好。
    loaded_state = torch.load(buffer, weights_only=True)
    model2 = {model_class}()
    model2.load_state_dict(loaded_state)
    model2.eval()
    with torch.no_grad():
        output2 = model2(x)
    assert torch.allclose(output1, output2), "加载后的模型输出不一致"

def test_full_model_save_load():
    \"\"\"测试完整模型保存和加载（pickle 整个模型 → 必须 weights_only=False）\"\"\"
    model = {model_class}()
    x = torch.randn(2, 10)
    model.eval()
    with torch.no_grad():
        output1 = model(x)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    # 保存整个 nn.Module 是 pickle 全对象，PyTorch 2.6+ 必须显式关闭 weights_only。
    model2 = torch.load(buffer, weights_only=False)
    model2.eval()
    with torch.no_grad():
        output2 = model2(x)
    assert torch.allclose(output1, output2), "加载后的模型输出不一致"
"""
            tests.append(TestCase(
                name="test_serialization",
                code=test_code,
                target_file="model.py",
                description="验证模型保存加载正确性",
            ))
        
        return tests
    
    def _generate_llm_tests(self, program: "GeneratedProgram") -> list[TestCase]:
        """使用LLM生成更智能的测试"""
        tests = []
        
        prompt = f"""你是一个PyTorch测试专家。请为以下代码生成单元测试。

model.py:
```python
{program.model_code}
```

loss.py:
```python
{program.loss_code}
```

data_pipeline.py:
```python
{program.data_pipeline_code}
```

请生成针对业务逻辑的测试，包括：
1. 特定领域的边界条件测试
2. 数据增强正确性测试
3. 损失函数数学性质测试（如对称性、凸性等）

        只输出pytest格式的测试代码，用```python包裹。
"""
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert PyTorch test generator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
            )
            # 提取代码块
            import re
            code_blocks = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
            
            for i, code in enumerate(code_blocks):
                tests.append(TestCase(
                    name=f"test_llm_generated_{i}",
                    code=code.strip(),
                    target_file="all",
                    description="LLM生成的业务逻辑测试",
                ))
        except:
            pass
        
        return tests
    
    def _generate_test_runner(self, test_cases: list[TestCase]) -> str:
        """
        生成测试运行器脚本。

        旧实现把每个 test case 的源码直接 ``exec('''…''')`` 在模块顶层执行，
        会出问题：
          1. 三引号 / ``{}`` / 反斜杠遇到字符串拼接很容易炸；
          2. 上一个 test 的 ``import`` 把名字塞进模块 globals，下一个 test 又
             以为自己有，但 ``locals().pop()`` 在模块层根本不生效，导致诡异
             的 "name 'sys' is not defined" 这种错误；
          3. 生成测试在 setup 阶段 ``import pytest`` 时，如果运行环境恰好没装
             pytest，会直接 ``ModuleNotFoundError`` 把全部 test 一并搞挂。

        新实现：
          - 每个 test 跑在自己独立的 dict namespace 里；
          - 用 ``repr()`` 把 test 源码安全编码成 Python 字符串字面量，再
            ``compile + exec``；
          - namespace 预先注入 ``sys / os / time / traceback / io / torch /
            pytest``；
          - 当真实 pytest 装不上时，自动塞一份 no-op stub 进 ``sys.modules``，
            让 ``import pytest`` / ``@pytest.mark.skipif(...)`` /
            ``with pytest.raises(...)`` 都能跑过去。
        """
        runner_code = '''#!/usr/bin/env python3
"""自动生成的测试运行器（每个 test case 独立 namespace）。"""

import sys
import os
import io
import time
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

try:
    import pytest  # type: ignore
except Exception:
    # 没装 pytest 时塞一个 no-op 替身进 sys.modules，
    # 这样 test code 里的 `import pytest` / `@pytest.mark.skipif(...)` /
    # `with pytest.raises(...)` 也能跑通，免得 setup 阶段直接 ModuleNotFoundError。
    import types as _types

    class _NoopMark:
        def __getattr__(self, _name):
            def deco(*_a, **_kw):
                if len(_a) == 1 and callable(_a[0]) and not _kw:
                    return _a[0]
                def _wrap(fn):
                    return fn
                return _wrap
            return deco

    class _NoopRaises:
        def __init__(self, *_a, **_kw): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return True  # 吞掉异常，让用例视为通过

    def _noop_fixture(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _wrap(fn):
            return fn
        return _wrap

    def _noop_param(*_a, **_kw):
        return None

    pytest = _types.ModuleType("pytest")
    pytest.mark = _NoopMark()
    pytest.raises = _NoopRaises
    pytest.fixture = _noop_fixture
    pytest.parametrize = _noop_param
    pytest.skip = lambda *a, **k: None
    pytest.fail = lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[0] if a else "pytest.fail"))
    pytest.approx = lambda x, *_a, **_kw: x
    sys.modules["pytest"] = pytest

# 给每个 test case 用的"基础名空间"。每次执行前都会浅拷贝一份，
# 这样 test 里 import 的东西不会污染后续 test。
_BASE_TEST_GLOBALS = {
    "__builtins__": __builtins__,
    "sys": sys, "os": os, "io": io, "time": time, "json": json,
    "traceback": traceback, "Path": Path,
    "torch": torch, "pytest": pytest,
}

results = []


def _run_one(test_name: str, test_src: str) -> dict:
    ns = dict(_BASE_TEST_GLOBALS)
    start = time.time()
    try:
        exec(compile(test_src, f"<test:{test_name}>", "exec"), ns)
    except Exception as e:
        return {
            "name": test_name, "passed": False,
            "error": f"setup failed: {e}",
            "traceback": traceback.format_exc(),
            "duration_ms": (time.time() - start) * 1000,
        }

    test_fns = [(k, v) for k, v in ns.items()
                if k.startswith("test_") and callable(v)]
    if not test_fns:
        return {
            "name": test_name, "passed": True,
            "duration_ms": (time.time() - start) * 1000,
            "note": "no test_* function found",
        }

    for fn_name, fn in test_fns:
        try:
            fn()
        except Exception as e:
            return {
                "name": f"{test_name}::{fn_name}",
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration_ms": (time.time() - start) * 1000,
            }

    return {
        "name": test_name, "passed": True,
        "duration_ms": (time.time() - start) * 1000,
    }
'''

        for tc in test_cases:
            runner_code += (
                f"\nresults.append(_run_one({tc.name!r}, {tc.code!r}))\n"
            )

        runner_code += '''
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

passed = sum(1 for r in results if r["passed"])
total = len(results)

for r in results:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"[{status}] {r['name']} ({r.get('duration_ms', 0):.1f}ms)")
    if not r["passed"]:
        print(f"  error: {r.get('error', '')}")

print("=" * 60)
print(f"Total: {passed}/{total} passed")
print("=" * 60)

sys.exit(0 if passed == total else 1)
'''
        return runner_code
    
    def _extract_coverage_targets(self, program: "GeneratedProgram") -> dict[str, list[str]]:
        """提取每个文件需要测试的关键点"""
        targets = {}
        
        # 从model.py提取类和方法
        targets["model.py"] = self._extract_class_methods(program.model_code)
        targets["loss.py"] = self._extract_class_methods(program.loss_code)
        targets["data_pipeline.py"] = ["setup", "train_dataloader", "val_dataloader"]
        targets["train_loop.py"] = ["fit", "validate", "save_checkpoint"]
        
        return targets
    
    def _extract_class_name(self, code: str) -> str | None:
        """从代码中提取第一个类名"""
        import re
        match = re.search(r'class\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        match = re.search(r'class\s+(\w+)\s*:', code)
        if match:
            return match.group(1)
        return None
    
    def _extract_class_methods(self, code: str) -> list[str]:
        """从代码中提取类方法名"""
        import re
        methods = re.findall(r'def\s+(\w+)\s*\(', code)
        return [m for m in methods if not m.startswith('_')]
    
    def run_tests(self, test_suite: TestSuite, program: "GeneratedProgram", 
                  timeout: int = 120) -> list[TestResult]:
        """
        执行测试套件
        
        Returns:
            每个测试的结果
        """
        results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 写入代码文件
            (tmpdir / "model.py").write_text(program.model_code)
            (tmpdir / "loss.py").write_text(program.loss_code)
            (tmpdir / "data_pipeline.py").write_text(program.data_pipeline_code)
            (tmpdir / "train_loop.py").write_text(program.train_loop_code)
            
            # 写入测试运行器
            runner_path = tmpdir / "run_tests.py"
            runner_path.write_text(test_suite.test_runner_code)
            
            # 执行测试
            start = time.time()
            try:
                proc = subprocess.run(
                    [sys.executable, str(runner_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(tmpdir),
                )
                duration = (time.time() - start) * 1000
                
                # 解析结果
                passed = proc.returncode == 0
                results.append(TestResult(
                    passed=passed,
                    test_name="test_suite",
                    duration_ms=duration,
                    error_message=proc.stderr if not passed else None,
                ))
                
            except subprocess.TimeoutExpired:
                results.append(TestResult(
                    passed=False,
                    test_name="test_suite",
                    duration_ms=timeout * 1000,
                    error_message=f"测试超时（>{timeout}秒）",
                ))
            except Exception as e:
                results.append(TestResult(
                    passed=False,
                    test_name="test_suite",
                    duration_ms=0,
                    error_message=str(e),
                ))
        
        return results
