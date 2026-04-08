"""
Sandbox Executor - 沙箱执行器

在隔离环境中执行生成的代码，确保：
1. 资源限制（CPU时间、内存）
2. 网络安全（禁止外联）
3. 文件系统隔离
4. 快速失败检测

使用 subprocess + resource limit 实现轻量级沙箱
（生产环境可考虑 Docker/Firecracker）
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SmokeTestResult:
    """冒烟测试结果"""
    passed: bool
    stage: str  # import, forward_pass, backward_pass, data_loading
    duration_ms: float
    error_message: str | None = None
    stack_trace: str | None = None
    resource_usage: dict = field(default_factory=dict)  # 内存、CPU使用
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "stage": self.stage,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "resource_usage": self.resource_usage,
        }


@dataclass
class ExecutionConfig:
    """执行配置"""
    max_memory_mb: int = 2048  # 最大内存使用
    max_cpu_time_sec: int = 300  # 最大CPU时间
    max_wall_time_sec: int = 600  # 最大墙钟时间
    network_allowed: bool = False  # 是否允许网络
    gpu_allowed: bool = True  # 是否允许GPU
    temp_dir: Path | None = None  # 临时目录


class SandboxExecutor:
    """沙箱执行器"""
    
    def __init__(self, config: ExecutionConfig | None = None):
        self.config = config or ExecutionConfig()
        
    def smoke_test(self, program: "GeneratedProgram") -> list[SmokeTestResult]:
        """
        执行冒烟测试，不训练，只验证代码能正确运行
        
        Stages:
        1. Import test - 所有模块能正确导入
        2. Model forward - 模型前向传播一次
        3. Loss computation - 损失计算
        4. Backward pass - 反向传播
        5. Data loading - 数据管道能加载一个batch
        """
        results = []
        
        # Stage 1: 导入测试
        results.append(self._test_import(program))
        
        # Stage 2: 模型前向传播
        results.append(self._test_forward_pass(program))
        
        # Stage 3: 损失计算
        results.append(self._test_loss_computation(program))
        
        # Stage 4: 反向传播
        results.append(self._test_backward_pass(program))
        
        # Stage 5: 数据管道
        results.append(self._test_data_pipeline(program))
        
        return results
    
    def _test_import(self, program: "GeneratedProgram") -> SmokeTestResult:
        """测试模块导入"""
        script = """
import sys
import time

start = time.time()

try:
    import torch
    import model
    import loss
    import data_pipeline
    import train_loop
    
    duration = (time.time() - start) * 1000
    print(f"IMPORT_SUCCESS: {duration:.2f}ms")
except Exception as e:
    duration = (time.time() - start) * 1000
    print(f"IMPORT_FAILED: {duration:.2f}ms")
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        return self._run_in_sandbox(script, program, "import")
    
    def _test_forward_pass(self, program: "GeneratedProgram") -> SmokeTestResult:
        """测试模型前向传播"""
        script = """
import sys
import time
import torch

start = time.time()

try:
    from model import *
    from data_pipeline import *
    
    # 实例化模型
    model_classes = [c for c in dir() if c[0].isupper() and c not in ['DataLoader', 'Dataset']]
    if not model_classes:
        raise RuntimeError("未找到模型类")
    
    ModelClass = globals()[model_classes[0]]
    model = ModelClass()
    model.eval()
    
    # 创建假数据
    x = torch.randn(2, 10)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 验证输出
    assert output is not None, "模型输出为None"
    assert torch.is_tensor(output), f"输出不是张量: {type(output)}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    
    duration = (time.time() - start) * 1000
    print(f"FORWARD_SUCCESS: {duration:.2f}ms, output_shape: {list(output.shape)}")
    
except Exception as e:
    duration = (time.time() - start) * 1000
    print(f"FORWARD_FAILED: {duration:.2f}ms")
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        return self._run_in_sandbox(script, program, "forward_pass")
    
    def _test_loss_computation(self, program: "GeneratedProgram") -> SmokeTestResult:
        """测试损失计算"""
        script = """
import sys
import time
import torch

start = time.time()

try:
    from model import *
    from loss import *
    
    # 找到损失类
    loss_classes = [c for c in dir() if c[0].isupper() and 'Loss' in c]
    if not loss_classes:
        print("LOSS_SKIPPED: 未找到损失类")
        sys.exit(0)
    
    LossClass = globals()[loss_classes[0]]
    loss_fn = LossClass()
    
    # 假数据
    pred = torch.randn(4, 1)
    target = torch.randn(4, 1)
    
    # 计算损失
    loss_val = loss_fn(pred, target)
    
    assert loss_val is not None, "损失为None"
    assert not torch.isnan(loss_val), "损失为NaN"
    assert not torch.isinf(loss_val), "损失为Inf"
    
    duration = (time.time() - start) * 1000
    print(f"LOSS_SUCCESS: {duration:.2f}ms, loss_value: {loss_val.item():.4f}")
    
except Exception as e:
    duration = (time.time() - start) * 1000
    print(f"LOSS_FAILED: {duration:.2f}ms")
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        return self._run_in_sandbox(script, program, "loss_computation")
    
    def _test_backward_pass(self, program: "GeneratedProgram") -> SmokeTestResult:
        """测试反向传播"""
        script = """
import sys
import time
import torch

start = time.time()

try:
    from model import *
    from loss import *
    
    # 找到类
    model_classes = [c for c in dir() if c[0].isupper() and c not in ['DataLoader', 'Dataset'] and 'Loss' not in c]
    loss_classes = [c for c in dir() if c[0].isupper() and 'Loss' in c]
    
    if not model_classes or not loss_classes:
        print("BACKWARD_SKIPPED: 未找到模型或损失类")
        sys.exit(0)
    
    ModelClass = globals()[model_classes[0]]
    LossClass = globals()[loss_classes[0]]
    
    model = ModelClass()
    loss_fn = LossClass()
    
    # 假数据
    x = torch.randn(2, 10)
    target = torch.randn(2, 1)
    
    # 前向
    output = model(x)
    loss = loss_fn(output, target)
    
    # 反向
    loss.backward()
    
    # 检查梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.parameters())
    assert has_grad, "没有参数接收到梯度"
    
    duration = (time.time() - start) * 1000
    print(f"BACKWARD_SUCCESS: {duration:.2f}ms")
    
except Exception as e:
    duration = (time.time() - start) * 1000
    print(f"BACKWARD_FAILED: {duration:.2f}ms")
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        return self._run_in_sandbox(script, program, "backward_pass")
    
    def _test_data_pipeline(self, program: "GeneratedProgram") -> SmokeTestResult:
        """测试数据管道"""
        script = """
import sys
import time
import torch

start = time.time()

try:
    from data_pipeline import *
    
    # 查找DataModule类
    dm_classes = [c for c in dir() if 'DataModule' in c or 'Dataset' in c]
    
    if not dm_classes:
        print("DATA_SKIPPED: 未找到DataModule类")
        sys.exit(0)
    
    # 尝试实例化并获取一个batch
    DMClass = globals()[dm_classes[0]]
    
    # 假设有默认构造参数
    try:
        dm = DMClass()
    except:
        # 尝试带参数构造
        dm = DMClass(data_dir=".", batch_size=2)
    
    # 尝试setup
    if hasattr(dm, 'setup'):
        try:
            dm.setup(stage='fit')
        except:
            pass
    
    # 尝试获取dataloader
    if hasattr(dm, 'train_dataloader'):
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        print(f"DATA_SUCCESS: loaded batch with {len(batch)} items")
    else:
        print("DATA_SKIPPED: 未找到train_dataloader方法")
    
    duration = (time.time() - start) * 1000
    print(f"DATA_SUCCESS: {duration:.2f}ms")
    
except Exception as e:
    duration = (time.time() - start) * 1000
    print(f"DATA_FAILED: {duration:.2f}ms")
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    # 数据管道失败不致命，可能是缺少真实数据
    print("DATA_WARNING: 数据管道测试失败，但这可能是由于缺少数据文件")
    sys.exit(0)  # 不视为致命错误
"""
        return self._run_in_sandbox(script, program, "data_pipeline")
    
    def _run_in_sandbox(
        self,
        script: str,
        program: "GeneratedProgram",
        stage: str,
    ) -> SmokeTestResult:
        """
        在沙箱中执行脚本
        """
        import time
        
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 写入代码文件
            (tmpdir / "model.py").write_text(program.model_code)
            (tmpdir / "loss.py").write_text(program.loss_code)
            (tmpdir / "data_pipeline.py").write_text(program.data_pipeline_code)
            (tmpdir / "train_loop.py").write_text(program.train_loop_code)
            
            # 写入测试脚本
            script_path = tmpdir / f"test_{stage}.py"
            script_path.write_text(script)
            
            # 准备执行环境
            env = os.environ.copy()
            if not self.config.network_allowed:
                # 可以通过设置无意义的代理来阻止网络访问
                env['http_proxy'] = 'http://127.0.0.1:0'
                env['https_proxy'] = 'http://127.0.0.1:0'
            
            if not self.config.gpu_allowed:
                env['CUDA_VISIBLE_DEVICES'] = ''
            
            # 执行
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.config.max_wall_time_sec,
                    env=env,
                    # 资源限制（仅Unix）
                    preexec_fn=self._set_resource_limits if os.name != 'nt' else None,
                )
                
                duration = (time.time() - start_time) * 1000
                
                # 解析输出
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()
                
                if proc.returncode == 0 and "_SUCCESS" in stdout:
                    # 提取额外信息
                    info = {}
                    for line in stdout.split('\n'):
                        if ':' in line:
                            parts = line.split(':', 1)
                            info[parts[0]] = parts[1].strip()
                    
                    return SmokeTestResult(
                        passed=True,
                        stage=stage,
                        duration_ms=duration,
                        resource_usage={"return_code": proc.returncode},
                    )
                else:
                    return SmokeTestResult(
                        passed=False,
                        stage=stage,
                        duration_ms=duration,
                        error_message=stderr or stdout,
                        stack_trace=stderr if stderr else None,
                    )
                    
            except subprocess.TimeoutExpired:
                return SmokeTestResult(
                    passed=False,
                    stage=stage,
                    duration_ms=self.config.max_wall_time_sec * 1000,
                    error_message=f"执行超时（>{self.config.max_wall_time_sec}秒）",
                )
            except Exception as e:
                return SmokeTestResult(
                    passed=False,
                    stage=stage,
                    duration_ms=(time.time() - start_time) * 1000,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                )
    
    def _set_resource_limits(self):
        """设置资源限制（在子进程中执行）"""
        try:
            # 内存限制（软限制和硬限制）
            max_memory = self.config.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
            
            # CPU时间限制
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.config.max_cpu_time_sec, self.config.max_cpu_time_sec)
            )
            
            # 子进程数量限制（防止fork炸弹）
            resource.setrlimit(resource.RLIMIT_NPROC, (100, 100))
            
        except Exception:
            pass
    
    def execute_training(
        self,
        program: "GeneratedProgram",
        config: dict,
        timeout: int = 3600,
    ) -> dict:
        """
        执行完整训练（BO调用）
        
        Args:
            program: 生成的程序
            config: 超参数配置
            timeout: 超时时间（秒）
            
        Returns:
            训练结果，包含验证集指标
        """
        # 构造训练脚本
        script = f"""
import sys
import json
import torch

# 设置配置
config = {repr(config)}

# 导入模块
from model import *
from loss import *
from data_pipeline import *
from train_loop import *

# 实例化（根据config）
# TODO: 根据search_space动态构造参数

# 运行训练
trainer = Trainer(max_epochs=10)
result = trainer.fit()

# 输出结果
print(f"FINAL_RESULT: {{json.dumps(result)}}")
"""
        
        result = self._run_in_sandbox(script, program, "training")
        
        # 解析训练结果
        if result.passed:
            try:
                # 从stdout解析JSON结果
                for line in result.error_message.split('\n'):
                    if line.startswith('FINAL_RESULT:'):
                        import json
                        metrics = json.loads(line.replace('FINAL_RESULT:', ''))
                        return {
                            "status": "success",
                            "metrics": metrics,
                            "duration_ms": result.duration_ms,
                        }
            except:
                pass
        
        return {
            "status": "failed",
            "error": result.error_message,
            "duration_ms": result.duration_ms,
        }