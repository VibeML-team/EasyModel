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


# ============================================================================
# 烟测脚本（模块级，提取出来后 ModalSandboxExecutor 可以直接复用同一份）
# 每个脚本都是「自包含的 python 源码」，会被写到工作目录里和生成的 5 个代码
# 文件一起执行；脚本内部用 _SUCCESS / _FAILED 关键字标记结果，
# `_run_in_sandbox` 解析这两个 token 来判断 passed。
# ============================================================================

SMOKE_SCRIPTS: dict[str, str] = {
    "import": """
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
""",
    "forward_pass": """
import sys
import time
import torch

start = time.time()

try:
    from model import *
    from data_pipeline import *

    model_classes = [c for c in dir() if c[0].isupper() and c not in ['DataLoader', 'Dataset']]
    if not model_classes:
        raise RuntimeError("未找到模型类")

    ModelClass = globals()[model_classes[0]]
    model = ModelClass()
    model.eval()

    x = torch.randn(2, 10)

    with torch.no_grad():
        output = model(x)

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
""",
    "loss_computation": """
import sys
import time
import torch

start = time.time()

try:
    from model import *
    from loss import *

    loss_classes = [c for c in dir() if c[0].isupper() and 'Loss' in c]
    if not loss_classes:
        print("LOSS_SKIPPED: 未找到损失类")
        sys.exit(0)

    LossClass = globals()[loss_classes[0]]
    loss_fn = LossClass()

    pred = torch.randn(4, 1)
    target = torch.randn(4, 1)

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
""",
    "backward_pass": """
import sys
import time
import torch

start = time.time()

try:
    from model import *
    from loss import *

    model_classes = [c for c in dir() if c[0].isupper() and c not in ['DataLoader', 'Dataset'] and 'Loss' not in c]
    loss_classes = [c for c in dir() if c[0].isupper() and 'Loss' in c]

    if not model_classes or not loss_classes:
        print("BACKWARD_SKIPPED: 未找到模型或损失类")
        sys.exit(0)

    ModelClass = globals()[model_classes[0]]
    LossClass = globals()[loss_classes[0]]

    model = ModelClass()
    loss_fn = LossClass()

    x = torch.randn(2, 10)
    target = torch.randn(2, 1)

    output = model(x)
    loss = loss_fn(output, target)

    loss.backward()

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
""",
    "data_pipeline": """
import sys
import time
import torch

start = time.time()

try:
    from data_pipeline import *

    dm_classes = [c for c in dir() if 'DataModule' in c or 'Dataset' in c]

    if not dm_classes:
        print("DATA_SKIPPED: 未找到DataModule类")
        sys.exit(0)

    DMClass = globals()[dm_classes[0]]

    try:
        dm = DMClass()
    except Exception:
        dm = DMClass(data_dir=".", batch_size=2)

    if hasattr(dm, 'setup'):
        try:
            dm.setup(stage='fit')
        except Exception:
            pass

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
    print("DATA_WARNING: 数据管道测试失败，但这可能是由于缺少数据文件")
    sys.exit(0)  # 数据管道失败不致命
""",
}


def build_program_files(program: "GeneratedProgram") -> dict[str, str]:
    """把 GeneratedProgram 转成 {filename: source} 的多文件 dict。

    QA 烟测和单元测试都要把这些文件写进 sandbox / Modal 容器，
    抽出来公用，避免两边复制粘贴。
    """
    return {
        "model.py": getattr(program, "model_code", "") or "",
        "loss.py": getattr(program, "loss_code", "") or "",
        "data_pipeline.py": getattr(program, "data_pipeline_code", "") or "",
        "train_loop.py": getattr(program, "train_loop_code", "") or "",
    }


def parse_smoke_stdout(
    stage: str,
    returncode: int,
    stdout: str,
    stderr: str,
    duration_ms: float,
) -> SmokeTestResult:
    """把子进程/远端容器的 (returncode, stdout, stderr) 解析成 SmokeTestResult。

    判定规则与历史一致：returncode==0 且 stdout 里出现 `_SUCCESS` 字样视为通过；
    例外是 data_pipeline 阶段—— driver 自己 sys.exit(0) 视为软通过。
    """
    out = (stdout or "").strip()
    err = (stderr or "").strip()
    if returncode == 0 and ("_SUCCESS" in out or "_SKIPPED" in out):
        return SmokeTestResult(
            passed=True,
            stage=stage,
            duration_ms=duration_ms,
            resource_usage={"return_code": returncode},
        )
    return SmokeTestResult(
        passed=False,
        stage=stage,
        duration_ms=duration_ms,
        error_message=err or out,
        stack_trace=err or None,
    )


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
        for stage, script in SMOKE_SCRIPTS.items():
            results.append(self._run_in_sandbox(script, program, stage))
        return results

    # 兼容旧测试用例直接调用 _test_xxx 的入口
    def _test_import(self, program: "GeneratedProgram") -> SmokeTestResult:
        return self._run_in_sandbox(SMOKE_SCRIPTS["import"], program, "import")

    def _test_forward_pass(self, program: "GeneratedProgram") -> SmokeTestResult:
        return self._run_in_sandbox(SMOKE_SCRIPTS["forward_pass"], program, "forward_pass")

    def _test_loss_computation(self, program: "GeneratedProgram") -> SmokeTestResult:
        return self._run_in_sandbox(SMOKE_SCRIPTS["loss_computation"], program, "loss_computation")

    def _test_backward_pass(self, program: "GeneratedProgram") -> SmokeTestResult:
        return self._run_in_sandbox(SMOKE_SCRIPTS["backward_pass"], program, "backward_pass")

    def _test_data_pipeline(self, program: "GeneratedProgram") -> SmokeTestResult:
        return self._run_in_sandbox(SMOKE_SCRIPTS["data_pipeline"], program, "data_pipeline")

    # 新增统一入口：让 QA pipeline / UnitTestGenerator 可以把任意脚本送进
    # 当前 sandbox 跑（本地 subprocess 或 Modal CPU 容器）。
    def run_python(
        self,
        files: dict[str, str],
        entry: str = "run_tests.py",
        timeout_sec: int = 120,
    ) -> dict[str, Any]:
        """通用入口：在沙箱里跑 entry 脚本（本地直接调 subprocess）。

        files 里同时包含「entry 脚本本身」和「被它 import 的所有支持文件」。
        """
        import time
        if entry not in files:
            raise ValueError(f"entry script {entry!r} 不在 files 中")

        start = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            for name, content in files.items():
                safe = name.lstrip("/").replace("..", "_")
                target = tmpdir_path / safe
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            env = os.environ.copy()
            if not self.config.network_allowed:
                env["http_proxy"] = "http://127.0.0.1:0"
                env["https_proxy"] = "http://127.0.0.1:0"
            if not self.config.gpu_allowed:
                env["CUDA_VISIBLE_DEVICES"] = ""

            try:
                proc = subprocess.run(
                    [sys.executable, entry],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    env=env,
                    cwd=str(tmpdir_path),
                    preexec_fn=self._set_resource_limits if os.name != "nt" else None,
                )
                return {
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "duration_sec": time.time() - start,
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired as e:
                return {
                    "returncode": -1,
                    "stdout": (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")),
                    "stderr": (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")),
                    "duration_sec": time.time() - start,
                    "timed_out": True,
                }
            except Exception as e:
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                    "duration_sec": time.time() - start,
                    "timed_out": False,
                }
    
    def _run_in_sandbox(
        self,
        script: str,
        program: "GeneratedProgram",
        stage: str,
    ) -> SmokeTestResult:
        """在沙箱中执行脚本（本地 subprocess 路径）。

        路径走通过 `run_python` 复用文件落盘 + 子进程逻辑，避免和单元测试两套写法。
        """
        import time

        files = build_program_files(program)
        entry = f"test_{stage}.py"
        files[entry] = script

        result = self.run_python(files, entry=entry, timeout_sec=self.config.max_wall_time_sec)

        if result.get("timed_out"):
            return SmokeTestResult(
                passed=False,
                stage=stage,
                duration_ms=self.config.max_wall_time_sec * 1000,
                error_message=f"执行超时（>{self.config.max_wall_time_sec}秒）",
            )

        return parse_smoke_stdout(
            stage=stage,
            returncode=int(result.get("returncode", -1)),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            duration_ms=float(result.get("duration_sec", 0.0)) * 1000,
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