"""
QA Pipeline - 完整质量保证流水线

整合静态分析、单元测试、沙箱执行，并提供自动修复循环。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .generator import GeneratedProgram, ProgramGenerator
from .static_analyzer import StaticAnalyzer, StaticCheckResult
from .test_generator import TestSuite, TestResult, UnitTestGenerator
from .sandbox import SandboxExecutor, SmokeTestResult


@dataclass
class QAResult:
    """QA流水线结果"""
    passed: bool
    program: GeneratedProgram | None
    attempts: int
    stage_results: list[dict]  # 每个阶段的详细结果
    total_duration_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "attempts": self.attempts,
            "total_duration_ms": self.total_duration_ms,
            "stage_results": self.stage_results,
        }


class QAPipeline:
    """
    质量保证流水线
    
    执行流程：
    1. 静态分析 (语法、导入、安全、PyTorch模式)
    2. 单元测试 (形状、梯度、设备、序列化)
    3. 沙箱冒烟测试 (导入、前向、反向、数据)
    4. (可选) 自动修复循环
    """
    
    def __init__(
        self,
        generator: ProgramGenerator | None = None,
        analyzer: StaticAnalyzer | None = None,
        test_generator: UnitTestGenerator | None = None,
        sandbox: SandboxExecutor | None = None,
        max_attempts: int = 3,
    ):
        self.generator = generator or ProgramGenerator()
        self.analyzer = analyzer or StaticAnalyzer()
        self.test_generator = test_generator or UnitTestGenerator()
        self.sandbox = sandbox or SandboxExecutor()
        self.max_attempts = max_attempts
        
    def validate(
        self,
        program: GeneratedProgram,
        auto_fix: bool = True,
    ) -> QAResult:
        """
        验证生成的程序
        
        Args:
            program: 待验证的程序
            auto_fix: 是否启用自动修复
            
        Returns:
            QAResult: 验证结果
        """
        start_time = time.time()
        attempts = 0
        stage_results = []
        
        current_program = program
        
        while attempts < self.max_attempts:
            attempts += 1
            all_passed = True
            iteration_results = {"attempt": attempts, "stages": []}
            
            # Stage 1: 静态分析
            static_results = self._run_static_analysis(current_program)
            iteration_results["stages"].append({
                "name": "static_analysis",
                "results": [r.to_dict() for r in static_results],
            })
            
            if not all(r.passed for r in static_results):
                all_passed = False
                if auto_fix and attempts < self.max_attempts:
                    errors = self._collect_errors(static_results)
                    current_program = self._fix_program(
                        current_program, errors, "static_analysis"
                    )
                    continue
            
            # Stage 2: 单元测试生成与执行
            test_result = self._run_unit_tests(current_program)
            iteration_results["stages"].append({
                "name": "unit_test",
                "result": test_result.to_dict() if hasattr(test_result, 'to_dict') else test_result,
            })
            
            if isinstance(test_result, TestResult) and not test_result.passed:
                all_passed = False
                if auto_fix and attempts < self.max_attempts:
                    errors = [{"message": test_result.error_message}]
                    current_program = self._fix_program(
                        current_program, errors, "unit_test"
                    )
                    continue
            
            # Stage 3: 沙箱冒烟测试
            smoke_results = self._run_smoke_tests(current_program)
            iteration_results["stages"].append({
                "name": "smoke_test",
                "results": [r.to_dict() for r in smoke_results],
            })
            
            if not all(r.passed for r in smoke_results):
                all_passed = False
                if auto_fix and attempts < self.max_attempts:
                    errors = self._collect_smoke_errors(smoke_results)
                    current_program = self._fix_program(
                        current_program, errors, "smoke_test"
                    )
                    continue
            
            stage_results.append(iteration_results)
            
            # 全部通过
            if all_passed:
                return QAResult(
                    passed=True,
                    program=current_program,
                    attempts=attempts,
                    stage_results=stage_results,
                    total_duration_ms=(time.time() - start_time) * 1000,
                )
        
        # 达到最大尝试次数仍未通过
        return QAResult(
            passed=False,
            program=current_program if not auto_fix else None,
            attempts=attempts,
            stage_results=stage_results,
            total_duration_ms=(time.time() - start_time) * 1000,
        )
    
    def _run_static_analysis(
        self,
        program: GeneratedProgram,
    ) -> list[StaticCheckResult]:
        """执行静态分析"""
        return self.analyzer.analyze(program)
    
    def _run_unit_tests(self, program: GeneratedProgram) -> TestResult | dict:
        """生成并执行单元测试"""
        try:
            test_suite = self.test_generator.generate_tests(program)
            results = self.test_generator.run_tests(test_suite, program, timeout=120)
            
            if results:
                return results[0]  # 返回总体结果
            
            return TestResult(
                passed=False,
                test_name="unit_test",
                duration_ms=0,
                error_message="未生成测试结果",
            )
        except Exception as e:
            return TestResult(
                passed=False,
                test_name="unit_test",
                duration_ms=0,
                error_message=f"单元测试执行异常: {str(e)}",
            )
    
    def _run_smoke_tests(
        self,
        program: GeneratedProgram,
    ) -> list[SmokeTestResult]:
        """执行沙箱冒烟测试"""
        return self.sandbox.smoke_test(program)
    
    def _collect_errors(self, results: list[StaticCheckResult]) -> list[dict]:
        """从静态分析结果收集错误"""
        errors = []
        for result in results:
            for err in result.errors:
                errors.append({
                    "stage": result.stage,
                    **err,
                })
        return errors
    
    def _collect_smoke_errors(self, results: list[SmokeTestResult]) -> list[dict]:
        """从冒烟测试结果收集错误"""
        errors = []
        for result in results:
            if not result.passed:
                errors.append({
                    "stage": result.stage,
                    "message": result.error_message,
                })
        return errors
    
    def _fix_program(
        self,
        program: GeneratedProgram,
        errors: list[dict],
        stage: str,
    ) -> GeneratedProgram:
        """调用LLM修复程序"""
        if not self.generator.llm_client:
            # 没有LLM客户端，无法自动修复
            return program
        
        return self.generator.fix_code(program, errors, stage)


class QuickValidator:
    """
    快速验证器 - 用于BO循环中的快速检查
    
    只执行最关键的验证，跳过耗时操作
    """
    
    def __init__(self):
        self.analyzer = StaticAnalyzer()
        self.sandbox = SandboxExecutor()
        
    def quick_check(self, program: GeneratedProgram) -> bool:
        """
        快速检查，只验证语法和能否导入
        
        Returns:
            是否通过快速检查
        """
        # 1. 快速语法检查
        files = {
            "model.py": program.model_code,
            "loss.py": program.loss_code,
        }
        
        for filename, code in files.items():
            if not code.strip():
                return False
            
            import ast
            try:
                ast.parse(code)
            except SyntaxError:
                return False
        
        # 2. 导入检查（只检查前几条import语句）
        for filename, code in files.items():
            lines = code.split('\n')[:20]  # 只检查前20行
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    # 简单检查是否是允许的模块
                    module = line.split()[1].split('.')[0]
                    if module not in ALLOWED_IMPORTS:
                        return False
        
        return True


# 为了快速验证，复制一份允许的导入列表
ALLOWED_IMPORTS = {
    "abc", "collections", "copy", "dataclasses", "enum", "functools", "inspect",
    "itertools", "json", "logging", "math", "numbers", "os", "pathlib", "pickle",
    "random", "re", "sys", "time", "typing", "warnings", "contextlib",
    "hashlib", "typing_extensions",
    "numpy", "np", "scipy", "sklearn", "pandas", "pd",
    "torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils.data",
    "torchvision", "torchvision.transforms", "torchvision.datasets",
    "torch_geometric", "torch_geometric.data", "torch_geometric.nn",
    "pytorch_lightning", "lightning", "lightning.pytorch",
    "omegaconf", "hydra", "wandb", "tensorboard",
    "tqdm", "matplotlib", "seaborn",
}