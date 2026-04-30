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


# =============================================================================
# Import 策略：allow-by-default + 显式黑名单
#
# 早期我们用「白名单 + 没列入就拒」的策略。结果就是 LLM 一旦写了
# `import cv2` / `import PIL` / `import transformers` / `import einops` 这种
# 完全合理的 ML 库，就被当成"危险导入"，整个 QA pipeline 三轮修不完直接 fail。
#
# 现在改成：
#   - DENY_IMPORTS：明确拦的真危险模块（联网、系统调用、shell）
#   - ALLOWED_IMPORTS：分类列出的"明确允许"集合，用于在错误信息和 LLM 修复
#     prompt 里给出"你应该用这些库"的建议；它**不再用于硬性拦截**。
#   - 任何不在 DENY_IMPORTS 中的模块默认通过；不在 ALLOWED_IMPORTS 中的会
#     产生 warning（不致命），方便我们日后观察 LLM 在用什么"奇怪"的库。
# =============================================================================

# 真正应该被拒绝的模块（联网、远程执行、shell 等，跟训练无关）
DENY_IMPORTS = {
    # 远程/网络（训练代码不应该外联）
    "subprocess", "socket", "ftplib", "smtplib", "telnetlib", "imaplib",
    "poplib", "nntplib", "xmlrpc", "http", "urllib", "urllib2", "urllib3",
    "requests", "httpx", "aiohttp", "websockets", "websocket",
    "paramiko", "fabric", "asyncssh",
    # Shell / 容器逃逸
    "pty", "ptyprocess", "pexpect", "shlex",
    # 钩子/反射类（不是 ML 必需，且容易被滥用）
    "ctypes", "cffi",
}

# 明确允许的库（按用途分类，主要给 LLM 修复 prompt 用作 "你可以用这些" 提示）
ALLOWED_IMPORT_GROUPS: dict[str, list[str]] = {
    "stdlib": [
        "abc", "argparse", "asyncio", "base64", "bisect", "builtins", "collections",
        "concurrent", "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal",
        "difflib", "dis", "enum", "errno", "fnmatch", "fractions", "functools", "gc",
        "glob", "gzip", "hashlib", "heapq", "html", "importlib", "inspect", "io",
        "ipaddress", "itertools", "json", "logging", "math", "multiprocessing",
        "numbers", "operator", "os", "pathlib", "pickle", "queue", "random", "re",
        "secrets", "shutil", "signal", "statistics", "string", "struct", "sys",
        "tarfile", "tempfile", "textwrap", "threading", "time", "timeit", "traceback",
        "types", "typing", "typing_extensions", "unicodedata", "uuid", "warnings",
        "weakref", "xml", "zipfile", "zlib",
    ],
    "core_numerical": [
        "numpy", "np", "scipy", "pandas", "pd", "polars", "pyarrow", "h5py",
        "joblib", "safetensors", "pyyaml", "yaml", "toml", "tomli",
    ],
    "torch_ecosystem": [
        "torch", "torchvision", "torchaudio", "torchtext", "torchdata",
        "torch_geometric", "torch_scatter", "torch_sparse", "torch_cluster",
        "pytorch_lightning", "lightning", "fastai",
        "torchmetrics", "torchinfo", "torchsummary",
        "einops", "einsum", "opt_einsum",
    ],
    "image_cv": [
        "cv2", "opencv", "PIL", "Pillow", "skimage", "imageio", "albumentations",
        "kornia", "imgaug", "augmentations", "imutils", "rasterio", "tifffile",
    ],
    "model_zoo": [
        "timm", "transformers", "diffusers", "accelerate", "peft", "bitsandbytes",
        "tokenizers", "datasets", "evaluate", "sentencepiece", "tiktoken",
        "segmentation_models_pytorch", "smp", "ultralytics", "yolov5", "yolov8",
        "efficientnet_pytorch", "pretrainedmodels", "huggingface_hub",
    ],
    "nlp": [
        "nltk", "spacy", "jieba", "pkuseg", "gensim", "fasttext", "stanza",
        "sacrebleu", "rouge_score", "bert_score",
    ],
    "audio": [
        "librosa", "soundfile", "audioread", "pydub", "torchaudio",
        "espnet", "speechbrain",
    ],
    "graph": [
        "networkx", "dgl", "graph_tool", "igraph", "node2vec",
    ],
    "rl": [
        "gym", "gymnasium", "stable_baselines3", "ray", "rllib", "pettingzoo",
        "tianshou", "d4rl", "highway_env",
    ],
    "ml_classical": [
        "sklearn", "xgboost", "lightgbm", "catboost", "mlxtend",
        "imbalanced_learn", "imblearn", "umap", "hdbscan",
        "shap", "lime",
    ],
    "config_logging": [
        "omegaconf", "hydra", "wandb", "tensorboard", "tensorboardX",
        "mlflow", "neptune", "comet_ml", "clearml",
        "loguru", "rich", "click", "tap", "fire", "absl",
    ],
    "viz": [
        "matplotlib", "seaborn", "plotly", "bokeh", "altair",
    ],
    "utils": [
        "tqdm", "attrs", "pydantic", "marshmallow", "cachetools",
        "more_itertools", "tenacity", "boltons", "toolz",
    ],
    "internal": [
        # 项目内相对/相邻模块（QA 时这些必然存在）
        "model", "loss", "data_pipeline", "train_loop", "utils", "dataset",
        "models", "losses", "datasets",
    ],
}

ALLOWED_IMPORTS: set[str] = {
    name for group in ALLOWED_IMPORT_GROUPS.values() for name in group
}

# 真正应当被拦截的"危险调用模式"。`open` 不在这里 —— 训练代码读 csv/parquet/
# 图像/checkpoint 完全合法，瞎拦只会让 LLM 写出更怪的绕过方案。
DANGEROUS_PATTERNS = {
    "eval": "使用eval()存在代码注入风险",
    "exec": "使用exec()存在代码注入风险",
    "__import__": "动态导入可能被滥用",
    "os.system": "系统命令执行被禁止",
    "os.popen": "系统命令执行被禁止",
    "os.execv": "系统命令执行被禁止",
    "os.execve": "系统命令执行被禁止",
    "os.spawn": "系统命令执行被禁止",
    "subprocess.run": "子进程调用被禁止",
    "subprocess.Popen": "子进程调用被禁止",
    "subprocess.call": "子进程调用被禁止",
    "subprocess.check_output": "子进程调用被禁止",
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
        self.deny_imports = DENY_IMPORTS
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

            # 1.5 编译检查
            compile_result = self._check_compile(filename, code)
            results.append(compile_result)
            if not compile_result.passed:
                continue
            
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

    def _check_compile(self, filename: str, code: str) -> StaticCheckResult:
        """检查代码能否完整编译为 Python code object。"""
        errors = []

        try:
            compile(code, filename, "exec")
        except SyntaxError as e:
            errors.append({
                "file": filename,
                "line": e.lineno,
                "column": e.offset,
                "message": f"编译失败: {e.msg}",
                "suggestion": "修复该文件直到可以被 Python compile() 成功编译",
            })
        except Exception as e:
            errors.append({
                "file": filename,
                "message": f"编译失败: {str(e)}",
                "suggestion": "检查该文件是否包含非法 Python 结构或不完整代码块",
            })

        return StaticCheckResult(
            passed=len(errors) == 0,
            stage="compile",
            errors=errors,
        )
    
    def _check_imports(self, filename: str, code: str) -> StaticCheckResult:
        """
        检查导入。

        策略（allow-by-default）：
          - 落在 DENY_IMPORTS 里的：error（联网、系统调用、远程执行）
          - 落在 ALLOWED_IMPORTS 里的：通过
          - 其它：warning（不致命，不阻断 QA），方便我们看到 LLM 在用什么
            "我们没列过但其实合理的"库，再决定要不要加进白名单。
        """
        errors: list[dict] = []
        warnings: list[dict] = []

        try:
            tree = ast.parse(code)
        except Exception:
            return StaticCheckResult(passed=False, stage="import",
                                     errors=[{"message": "无法解析AST"}])

        def _classify(module: str, lineno: int) -> None:
            if not module:
                return
            top = module.split(".")[0]
            if top in self.deny_imports:
                errors.append({
                    "file": filename,
                    "line": lineno,
                    "message": f"禁止导入模块: {top}（联网 / 系统调用 / 远程执行类，安全策略不允许）",
                    "suggestion": (
                        "训练代码不应外联或调用系统命令；"
                        "数据下载请离线完成，IO 走 torch / numpy / pandas / PIL / cv2。"
                    ),
                })
            elif top not in self.allowed_imports:
                warnings.append({
                    "file": filename,
                    "line": lineno,
                    "message": f"未在白名单内的导入: {top}（已放行，仅作提示）",
                    "suggestion": "如确有必要可继续使用；否则建议改为常见 ML 生态库。",
                })

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _classify(alias.name, node.lineno)
            elif isinstance(node, ast.ImportFrom):
                # `from . import x` / `from .foo import bar`：相对导入，永远允许
                if node.level and not node.module:
                    continue
                if node.level and node.module:
                    # 相对导入也直接放行
                    continue
                _classify(node.module or "", node.lineno)

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
