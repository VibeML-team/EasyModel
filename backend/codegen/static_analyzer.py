"""
Static Analyzer - 静态代码分析

对生成的代码进行多维度静态检查：
1. 语法检查 (AST解析)
2. 导入检查 (白名单机制)
3. 类型检查 (mypy)
4. 安全检查 (bandit)
5. PyTorch特定检查 (张量操作正确性)
"""

from __future__ import annotations

import ast
import io
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# 允许的导入白名单
ALLOWED_IMPORTS = {
    # 标准库
    "abc", "collections", "copy", "dataclasses", "enum", "functools", "inspect",
    "itertools", "json", "logging", "math", "numbers", "os", "pathlib", "pickle",
    "random", "re", "sys", "time", "typing", "warnings", "contextlib",
    "hashlib", "typing_extensions",
    
    # 数值计算
    "numpy", "np",
    "scipy", "sklearn", "pandas", "pd",
    
    # PyTorch生态
    "torch", "torch.nn", "torch.nn.functional", "torch.optim", "torch.utils.data",
    "torchvision", "torchvision.transforms", "torchvision.datasets",
    "torch_geometric", "torch_geometric.data", "torch_geometric.nn",
    "pytorch_lightning", "lightning", "lightning.pytorch",
    
    # 配置和工具
    "omegaconf", "hydra", "wandb", "tensorboard",
    "tqdm", "matplotlib", "seaborn",
}

# 禁止的危险函数/模式
DANGEROUS_PATTERNS = {
    "eval": "使用eval()存在代码注入风险",
    "exec": "使用exec()存在代码注入风险",
    "compile": "使用compile()存在代码注入风险",
    "__import__": "动态导入可能被滥用",
    "subprocess": "子进程调用存在安全风险",
    "socket": "网络操作被禁止",
    "urllib": "网络请求被禁止",
    "requests": "网络请求被禁止",
    "open": "文件操作应当通过DataModule进行",
    "os.system": "系统命令执行被禁止",
    "os.popen": "系统命令执行被禁止",
}


@dataclass
class StaticCheckResult:
    """静态检查结果"""
    passed: bool
    stage: str  # syntax, import, security, pytorch
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "stage": self.stage,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class StaticAnalyzer:
    """静态代码分析器"""
    
    def __init__(self):
        self.allowed_imports = ALLOWED_IMPORTS
        self.dangerous_patterns = DANGEROUS_PATTERNS
        
    def analyze(self, program: "GeneratedProgram") -> list[StaticCheckResult]:
        """
        对程序进行全面静态分析
        
        Returns:
            各阶段的检查结果列表
        """
        results = []
        
        files = {
            "model.py": program.model_code,
            "loss.py": program.loss_code,
            "data_pipeline.py": program.data_pipeline_code,
            "train_loop.py": program.train_loop_code,
        }
        
        for filename, code in files.items():
            if not code.strip():
                results.append(StaticCheckResult(
                    passed=False,
                    stage="syntax",
                    errors=[{"file": filename, "message": "代码为空"}]
                ))
                continue
            
            # 1. 语法检查
            syntax_result = self._check_syntax(filename, code)
            results.append(syntax_result)
            if not syntax_result.passed:
                continue  # 语法错误，跳过后续检查
            
            # 2. 导入检查
            import_result = self._check_imports(filename, code)
            results.append(import_result)
            
            # 3. 安全检查
            security_result = self._check_security(filename, code)
            results.append(security_result)
            
            # 4. PyTorch特定检查
            pytorch_result = self._check_pytorch_patterns(filename, code)
            results.append(pytorch_result)
        
        return results
    
    def _check_syntax(self, filename: str, code: str) -> StaticCheckResult:
        """检查Python语法"""
        errors = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append({
                "file": filename,
                "line": e.lineno,
                "column": e.offset,
                "message": f"语法错误: {e.msg}",
                "suggestion": "检查括号匹配、缩进、关键字拼写",
            })
        except Exception as e:
            errors.append({
                "file": filename,
                "message": f"解析错误: {str(e)}",
            })
        
        return StaticCheckResult(
            passed=len(errors) == 0,
            stage="syntax",
            errors=errors,
        )
    
    def _check_imports(self, filename: str, code: str) -> StaticCheckResult:
        """检查导入是否安全"""
        errors = []
        warnings = []
        
        try:
            tree = ast.parse(code)
        except:
            return StaticCheckResult(passed=False, stage="import", errors=[{"message": "无法解析AST"}])
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module not in self.allowed_imports:
                        errors.append({
                            "file": filename,
                            "line": node.lineno,
                            "message": f"禁止导入模块: {module}",
                            "suggestion": f"请使用白名单中的库: {', '.join(sorted(self.allowed_imports)[:5])}...",
                        })
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module.split('.')[0] if node.module else ""
                if module and module not in self.allowed_imports:
                    errors.append({
                        "file": filename,
                        "line": node.lineno,
                        "message": f"禁止导入模块: {module}",
                        "suggestion": "检查导入的库是否在允许列表中",
                    })
        
        return StaticCheckResult(
            passed=len(errors) == 0,
            stage="import",
            errors=errors,
            warnings=warnings,
        )
    
    def _check_security(self, filename: str, code: str) -> StaticCheckResult:
        """安全检查 - 检测危险模式"""
        errors = []
        warnings = []
        
        try:
            tree = ast.parse(code)
        except:
            return StaticCheckResult(passed=False, stage="security", errors=[{"message": "无法解析AST"}])
        
        for node in ast.walk(tree):
            # 检查函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in self.dangerous_patterns:
                        errors.append({
                            "file": filename,
                            "line": node.lineno,
                            "message": self.dangerous_patterns[func_name],
                            "suggestion": f"避免使用{func_name}()，寻找安全的替代方案",
                        })
                
                elif isinstance(node.func, ast.Attribute):
                    # 检查 os.system 等
                    full_name = self._get_attribute_chain(node.func)
                    if full_name in self.dangerous_patterns:
                        errors.append({
                            "file": filename,
                            "line": node.lineno,
                            "message": self.dangerous_patterns[full_name],
                            "suggestion": "避免系统调用",
                        })
            
            # 检查 __import__
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    errors.append({
                        "file": filename,
                        "line": node.lineno,
                        "message": self.dangerous_patterns["__import__"],
                        "suggestion": "使用标准import语句",
                    })
        
        # 检查字符串中是否包含危险代码模式
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for pattern in ["eval(", "exec(", "__import__", "os.system"]:
                    if pattern in node.value:
                        warnings.append({
                            "file": filename,
                            "line": node.lineno,
                            "message": f"字符串中包含潜在危险模式: {pattern}",
                            "suggestion": "确保这不是代码注入",
                        })
        
        return StaticCheckResult(
            passed=len(errors) == 0,
            stage="security",
            errors=errors,
            warnings=warnings,
        )
    
    def _check_pytorch_patterns(self, filename: str, code: str) -> StaticCheckResult:
        """PyTorch特定模式检查"""
        errors = []
        warnings = []
        
        try:
            tree = ast.parse(code)
        except:
            return StaticCheckResult(passed=False, stage="pytorch", errors=[{"message": "无法解析AST"}])
        
        # 检查是否继承nn.Module
        has_nn_module = False
        has_forward = False
        
        for node in ast.walk(tree):
            # 检查类定义
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = self._get_attribute_chain(base)
                    if "Module" in base_name or base_name == "nn.Module":
                        has_nn_module = True
                        
                        # 检查是否有forward方法
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                                has_forward = True
                                break
        
        if filename == "model.py" and not has_nn_module:
            warnings.append({
                "file": filename,
                "message": "模型类未继承nn.Module",
                "suggestion": "确保模型类继承torch.nn.Module",
            })
        
        if filename == "model.py" and has_nn_module and not has_forward:
            errors.append({
                "file": filename,
                "message": "模型类缺少forward方法",
                "suggestion": "所有nn.Module子类必须定义forward方法",
            })
        
        # 检查常见的张量维度错误模式
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_attribute_chain(node.func)
                
                # 检查view/reshape使用，确保维度匹配
                if func_name and "view" in func_name:
                    # 检查是否有-1（自动推断）
                    has_auto = any(
                        isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub)
                        and isinstance(arg.operand, ast.Constant) and arg.operand.value == 1
                        for arg in node.args
                    )
                    if not has_auto and len(node.args) < 2:
                        warnings.append({
                            "file": filename,
                            "line": node.lineno,
                            "message": "view()调用可能缺少维度参数",
                            "suggestion": "考虑使用-1让PyTorch自动计算维度",
                        })
                
                # 检查cuda()调用（应当使用device参数）
                if func_name and "cuda" in func_name:
                    warnings.append({
                        "file": filename,
                        "line": node.lineno,
                        "message": "检测到.cuda()调用，建议使用.to(device)以提高可移植性",
                    })
        
        return StaticCheckResult(
            passed=len(errors) == 0,
            stage="pytorch",
            errors=errors,
            warnings=warnings,
        )
    
    def _get_attribute_chain(self, node) -> str:
        """获取属性链的完整名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_chain(node.value)}.{node.attr}"
        return ""
    
    def analyze_with_mypy(self, program: "GeneratedProgram") -> StaticCheckResult:
        """
        使用mypy进行类型检查（可选，需要安装mypy）
        """
        errors = []
        
        try:
            import mypy.api
        except ImportError:
            return StaticCheckResult(
                passed=True,
                stage="type_check",
                warnings=[{"message": "mypy未安装，跳过类型检查"}],
            )
        
        # 创建临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            files = {
                "model.py": program.model_code,
                "loss.py": program.loss_code,
                "data_pipeline.py": program.data_pipeline_code,
                "train_loop.py": program.train_loop_code,
            }
            
            for filename, code in files.items():
                (Path(tmpdir) / filename).write_text(code, encoding="utf-8")
            
            # 运行mypy
            stdout, stderr, exit_code = mypy.api.run([
                tmpdir,
                "--ignore-missing-imports",
                "--no-error-summary",
            ])
            
            if exit_code != 0 and stdout.strip():
                for line in stdout.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 3)
                        if len(parts) >= 3:
                            errors.append({
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "message": parts[-1].strip(),
                            })
        
        return StaticCheckResult(
            passed=len(errors) == 0,
            stage="type_check",
            errors=errors,
        )