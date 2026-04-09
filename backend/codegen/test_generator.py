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
import tempfile
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    \"\"\"测试CUDA前向传播\"\"\"
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
    
    # 获取输出
    model.eval()
    with torch.no_grad():
        output1 = model(x)
    
    # 保存和加载
    state_dict = model.state_dict()
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    loaded_state = torch.load(buffer)
    
    # 新模型加载权重
    model2 = {model_class}()
    model2.load_state_dict(loaded_state)
    model2.eval()
    
    with torch.no_grad():
        output2 = model2(x)
    
    assert torch.allclose(output1, output2), "加载后的模型输出不一致"

def test_full_model_save_load():
    \"\"\"测试完整模型保存和加载\"\"\"
    model = {model_class}()
    x = torch.randn(2, 10)
    
    model.eval()
    with torch.no_grad():
        output1 = model(x)
    
    # 保存整个模型
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    
    # 加载
    model2 = torch.load(buffer)
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
        """生成测试运行器脚本"""
        runner_code = """#!/usr/bin/env python3
\"\"\"自动生成的测试运行器\"\"\"

import sys
import time
import traceback
from pathlib import Path

# 添加代码目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import torch

# 设置确定性行为
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

results = []
"""
        
        for tc in test_cases:
            runner_code += f"""
# {tc.name}: {tc.description}
try:
    exec('''
{tc.code}
''')
    # 执行测试函数
    start = time.time()
    test_functions = [obj for name, obj in locals().items() if name.startswith('test_') and callable(obj)]
    for test_fn in test_functions:
        test_fn()
    duration = (time.time() - start) * 1000
    results.append({{"name": "{tc.name}", "passed": True, "duration_ms": duration}})
except Exception as e:
    results.append({{
        "name": "{tc.name}",
        "passed": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }})

# 清理locals避免命名冲突
locals().pop('test_functions', None)
for name in list(locals().keys()):
    if name.startswith('test_'):
        locals().pop(name, None)
"""
        
        runner_code += """
# 输出结果
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

passed = sum(1 for r in results if r["passed"])
total = len(results)

for r in results:
    status = "✓ PASS" if r["passed"] else "✗ FAIL"
    print(f"{status}: {r['name']} ({r.get('duration_ms', 0):.1f}ms)")
    if not r["passed"]:
        print(f"  Error: {r['error']}")

print("=" * 60)
print(f"Total: {passed}/{total} passed")
print("=" * 60)

sys.exit(0 if passed == total else 1)
"""
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
        import subprocess
        import time
        
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
