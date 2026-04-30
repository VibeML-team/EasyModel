"""
QA Pipeline - 完整质量保证流水线

整合静态分析、单元测试、沙箱执行，并提供自动修复循环。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .generator import GeneratedProgram, ProgramGenerator
from .static_analyzer import (
    ALLOWED_IMPORTS as _STATIC_ALLOWED_IMPORTS,
    DENY_IMPORTS as _STATIC_DENY_IMPORTS,
    StaticAnalyzer,
    StaticCheckResult,
)
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
    soft_passed: bool = False           # True = 没硬通过，但剩下的问题不致命，已放行
    soft_pass_reason: str | None = None  # 用于日志展示

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "attempts": self.attempts,
            "total_duration_ms": self.total_duration_ms,
            "stage_results": self.stage_results,
            "soft_passed": self.soft_passed,
            "soft_pass_reason": self.soft_pass_reason,
        }


# 哪些阶段失败属于"非致命，可以放行训练"。
# - import：白名单缺漏导致的 warning 已经在 static_analyzer 里降级为 warning 了，
#   这里兜底兼容历史 stage 名。
# - pytorch / security_warning：风格类提示，非阻断。
# 其它（syntax / compile / unit_test / smoke_test）仍然必须真通过。
_NON_FATAL_STAGES = {"import", "pytorch", "security_warning"}


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
        on_event=None,
    ) -> QAResult:
        """
        验证生成的程序

        Args:
            program: 待验证的程序
            auto_fix: 是否启用自动修复
            on_event: 可选 callback，把 QA 各阶段状态推给上层 UI/日志：
              {"type":"attempt_start","attempt":1,"max":N}
              {"type":"stage_start","stage":"static_analysis","attempt":1}
              {"type":"stage_complete","stage":"...","passed":bool,"duration":float,"errors":int}
              {"type":"fix_start","stage":"...","errors":[...]}
              {"type":"fix_complete","stage":"..."}
              {"type":"qa_done","passed":bool,"attempts":N,"duration":float}

        Returns:
            QAResult: 验证结果
        """
        emit = on_event if callable(on_event) else (lambda evt: None)

        start_time = time.time()
        attempts = 0
        stage_results = []

        current_program = program

        while attempts < self.max_attempts:
            attempts += 1
            all_passed = True
            iteration_results = {"attempt": attempts, "stages": []}
            emit({"type": "attempt_start", "attempt": attempts, "max": self.max_attempts})

            missing_files = self.generator.missing_required_files(current_program)
            if missing_files:
                all_passed = False
                emit({
                    "type": "stage_complete",
                    "stage": "required_files",
                    "passed": False,
                    "duration": 0.0,
                    "errors": len(missing_files),
                    "detail": f"缺失文件: {', '.join(missing_files)}",
                })
                iteration_results["stages"].append({
                    "name": "required_files",
                    "results": [
                        {
                            "passed": False,
                            "stage": "required_files",
                            "errors": [
                                {"file": filename, "message": "必需代码文件为空或缺失"}
                                for filename in missing_files
                            ],
                            "warnings": [],
                        }
                    ],
                })
                if auto_fix and attempts < self.max_attempts:
                    stage_results.append(iteration_results)
                    emit({"type": "fix_start", "stage": "required_files",
                          "errors": [f for f in missing_files]})
                    current_program = self._fix_program(
                        current_program,
                        [{"file": filename, "message": "必需代码文件为空或缺失"} for filename in missing_files],
                        "required_files",
                    )
                    emit({"type": "fix_complete", "stage": "required_files"})
                    continue

            # Stage 1: 静态分析
            emit({"type": "stage_start", "stage": "static_analysis", "attempt": attempts})
            t_stage = time.time()
            static_results = self._run_static_analysis(current_program)
            stage_passed = all(r.passed for r in static_results)
            stage_errors = sum(len(r.errors) for r in static_results)
            emit({
                "type": "stage_complete",
                "stage": "static_analysis",
                "passed": stage_passed,
                "duration": round(time.time() - t_stage, 2),
                "errors": stage_errors,
            })
            iteration_results["stages"].append({
                "name": "static_analysis",
                "results": [r.to_dict() for r in static_results],
            })

            if not stage_passed:
                all_passed = False
                if auto_fix and attempts < self.max_attempts:
                    stage_results.append(iteration_results)
                    errors = self._collect_errors(static_results)
                    emit({"type": "fix_start", "stage": "static_analysis", "errors": errors[:3]})
                    current_program = self._fix_program(
                        current_program, errors, "static_analysis"
                    )
                    emit({"type": "fix_complete", "stage": "static_analysis"})
                    continue

            # Stage 2: 单元测试生成与执行
            emit({"type": "stage_start", "stage": "unit_test", "attempt": attempts})
            t_stage = time.time()
            test_result = self._run_unit_tests(current_program)
            unit_passed = not (isinstance(test_result, TestResult) and not test_result.passed)
            unit_err_msg = (test_result.error_message
                            if isinstance(test_result, TestResult) and not test_result.passed
                            else "")
            emit({
                "type": "stage_complete",
                "stage": "unit_test",
                "passed": unit_passed,
                "duration": round(time.time() - t_stage, 2),
                "errors": 0 if unit_passed else 1,
                "detail": unit_err_msg[:120] if unit_err_msg else None,
            })
            iteration_results["stages"].append({
                "name": "unit_test",
                "result": test_result.to_dict() if hasattr(test_result, 'to_dict') else test_result,
            })

            if not unit_passed:
                all_passed = False
                if auto_fix and attempts < self.max_attempts:
                    stage_results.append(iteration_results)
                    errors = [{"message": test_result.error_message}]
                    emit({"type": "fix_start", "stage": "unit_test", "errors": errors})
                    current_program = self._fix_program(
                        current_program, errors, "unit_test"
                    )
                    emit({"type": "fix_complete", "stage": "unit_test"})
                    continue

            # Stage 3: 沙箱冒烟测试
            emit({"type": "stage_start", "stage": "smoke_test", "attempt": attempts})
            t_stage = time.time()
            smoke_results = self._run_smoke_tests(current_program)
            smoke_passed = all(r.passed for r in smoke_results)
            smoke_errors = sum(0 if r.passed else 1 for r in smoke_results)
            emit({
                "type": "stage_complete",
                "stage": "smoke_test",
                "passed": smoke_passed,
                "duration": round(time.time() - t_stage, 2),
                "errors": smoke_errors,
            })
            iteration_results["stages"].append({
                "name": "smoke_test",
                "results": [r.to_dict() for r in smoke_results],
            })

            if not smoke_passed:
                all_passed = False
                if auto_fix and attempts < self.max_attempts:
                    stage_results.append(iteration_results)
                    errors = self._collect_smoke_errors(smoke_results)
                    emit({"type": "fix_start", "stage": "smoke_test", "errors": errors[:3]})
                    current_program = self._fix_program(
                        current_program, errors, "smoke_test"
                    )
                    emit({"type": "fix_complete", "stage": "smoke_test"})
                    continue

            stage_results.append(iteration_results)

            # 全部通过
            if all_passed:
                emit({
                    "type": "qa_done",
                    "passed": True,
                    "attempts": attempts,
                    "duration": round(time.time() - start_time, 2),
                })
                return QAResult(
                    passed=True,
                    program=current_program,
                    attempts=attempts,
                    stage_results=stage_results,
                    total_duration_ms=(time.time() - start_time) * 1000,
                )

        # 达到最大尝试次数仍未通过 → 检查"剩下的问题是不是其实可以放行"
        soft_passed, soft_reason = self._is_soft_passable(stage_results)

        emit({
            "type": "qa_done",
            "passed": soft_passed,        # soft_pass 也要让上层显示成"通过"
            "attempts": attempts,
            "duration": round(time.time() - start_time, 2),
            "soft_passed": soft_passed,
            "soft_pass_reason": soft_reason,
        })
        return QAResult(
            passed=soft_passed,
            program=current_program if soft_passed or not auto_fix else None,
            attempts=attempts,
            stage_results=stage_results,
            total_duration_ms=(time.time() - start_time) * 1000,
            soft_passed=soft_passed,
            soft_pass_reason=soft_reason,
        )

    def _is_soft_passable(
        self,
        stage_results: list[dict],
    ) -> tuple[bool, str | None]:
        """
        判定"虽然没硬通过，但剩下的问题可以放行"。

        条件：
          - 必须有过至少一轮的完整 stage_results
          - 最后一轮里：必需文件 / syntax / compile / unit_test / smoke_test 全部通过
          - 仅剩 import / pytorch / security_warning 之类的"风格 / 提示"类失败
        这样就避免了 cv2 这种白名单遗漏直接把训练阻断掉。
        """
        if not stage_results:
            return False, None

        last = stage_results[-1]
        bad_stage_names: list[str] = []
        for stage in last.get("stages", []):
            stage_name = stage.get("name")
            results = stage.get("results", [])
            if isinstance(results, dict):
                results = [results]

            for r in results:
                if isinstance(r, dict) and r.get("passed") is False:
                    actual_stage = r.get("stage") or stage_name
                    if actual_stage not in _NON_FATAL_STAGES:
                        return False, None
                    bad_stage_names.append(actual_stage)

            single = stage.get("result")
            if isinstance(single, dict) and single.get("passed") is False:
                if stage_name not in _NON_FATAL_STAGES:
                    return False, None
                bad_stage_names.append(stage_name)

        if not bad_stage_names:
            return False, None

        unique_stages = sorted(set(bad_stage_names))
        return True, "仅剩非致命问题已放行：" + ", ".join(unique_stages)
    
    def _run_static_analysis(
        self,
        program: GeneratedProgram,
    ) -> list[StaticCheckResult]:
        """执行静态分析"""
        return self.analyzer.analyze(program)
    
    def _run_unit_tests(self, program: GeneratedProgram) -> TestResult | dict:
        """生成并执行单元测试。

        如果 ``self.sandbox`` 暴露了 ``run_python``（本地 SandboxExecutor 或
        ModalSandboxExecutor 都符合），就把测试运行器丢进去执行——这样在
        Zeabur 这种本机不带 torch 的环境里，QA 也能靠 Modal 容器跑通。
        """
        try:
            test_suite = self.test_generator.generate_tests(program)
            sandbox = self.sandbox if hasattr(self.sandbox, "run_python") else None
            results = self.test_generator.run_tests(
                test_suite, program, timeout=180, sandbox=sandbox,
            )
            
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
        快速检查，只验证：
          1. model.py / loss.py 不为空且能被 compile()
          2. 顶部若干行的 import 没有命中 DENY_IMPORTS
            （和 StaticAnalyzer 一致：未列入白名单的库默认放行，只拦真危险的）
        """
        files = {
            "model.py": program.model_code,
            "loss.py": program.loss_code,
        }

        for filename, code in files.items():
            if not code.strip():
                return False
            try:
                compile(code, filename, "exec")
            except Exception:
                return False

        for _filename, code in files.items():
            for line in code.split("\n")[:40]:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("import "):
                    module = stripped.split()[1].split(".")[0].rstrip(",")
                    if module in DENY_IMPORTS:
                        return False
                elif stripped.startswith("from "):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        module = parts[1].split(".")[0]
                        if module and module in DENY_IMPORTS:
                            return False

        return True


# 历史兼容：之前在这里维护过一份"允许列表"，现在直接复用 static_analyzer 的全集，
# 避免两套白名单越漂越远（典型的 cv2 没加这边没加那边）。
ALLOWED_IMPORTS = _STATIC_ALLOWED_IMPORTS
DENY_IMPORTS = _STATIC_DENY_IMPORTS
