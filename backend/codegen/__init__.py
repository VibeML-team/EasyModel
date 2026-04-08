"""
Code Generation & Quality Assurance Pipeline

从自然语言生成可执行的深度学习代码，并通过多层 QA 确保正确性。
"""

from .generator import ProgramGenerator, GeneratedProgram
from .static_analyzer import StaticAnalyzer, StaticCheckResult
from .test_generator import UnitTestGenerator, TestSuite
from .sandbox import SandboxExecutor, SmokeTestResult
from .qa_pipeline import QAPipeline, QAResult

__all__ = [
    "ProgramGenerator",
    "GeneratedProgram",
    "StaticAnalyzer",
    "StaticCheckResult",
    "UnitTestGenerator",
    "TestSuite",
    "SandboxExecutor",
    "SmokeTestResult",
    "QAPipeline",
    "QAResult",
]