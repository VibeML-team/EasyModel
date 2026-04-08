"""
Program Generator - LLM-based Code Synthesis

根据自然语言意图生成深度学习训练代码，包括：
- model.py: 模型架构
- loss.py: 损失函数
- data_pipeline.py: 数据加载和预处理
- train_loop.py: 训练循环
- search_space.yaml: BO搜索空间定义
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# 领域特定的 Prompt 模板
DOMAIN_TEMPLATES = {
    "ai4science": {
        "system_prompt": """你是AI for Science领域的深度学习专家。
你的任务是根据用户需求生成高质量的PyTorch代码。

代码要求：
1. 使用PyTorch 2.0+语法，支持torch.compile
2. 必须包含类型注解
3. 使用torch.nn.Module作为基类
4. 处理科学数据的特殊考虑（不确定性、物理约束、多尺度等）

输出格式：
- 每个文件用 ```python filename.py 包裹
- 最后给出 search_space.yaml 内容
- 代码必须是自包含且可运行的""",
        "examples": [
            {
                "intent": "蛋白质结构预测，低质量样本多，需要不确定性建模",
                "code_structure": {
                    "model": "EvidentialGraphTransformer",
                    "loss": "EvidentialRegressionLoss",
                    "data": "QualityAwareDataModule",
                }
            }
        ]
    },
    "gnn": {
        "system_prompt": """你是图神经网络(GNN)领域的专家。

代码要求：
1. 使用PyTorch Geometric或DGL
2. 支持动态图和时序图
3. 考虑图采样和可扩展性
4. 包含边特征和节点特征的处理

特别注意：
- 处理大规模图时使用采样策略
- 时序图需要处理时间编码""",
    },
    "timeseries": {
        "system_prompt": """你是时序预测领域的专家。

代码要求：
1. 支持多尺度时间特征（小时、天、周、季节）
2. 处理缺失值和异常值
3. 考虑长期依赖和短期模式
4. 支持概率预测和区间估计

特别注意：
- 时间泄漏检查
- 外生变量处理
- 节假日和特殊事件处理""",
    },
    "rl": {
        "system_prompt": """你是强化学习领域的专家。

代码要求：
1. 支持在线和离线RL
2. 包含安全约束处理
3. 支持多智能体场景
4. 考虑样本效率和稳定性

特别注意：
- 策略约束（不要偏离behavior policy太远）
- 值函数overestimation处理
- 安全约束的硬性保证""",
    },
}


@dataclass
class GeneratedProgram:
    """生成的程序包"""
    
    # 代码文件
    model_code: str
    loss_code: str
    data_pipeline_code: str
    train_loop_code: str
    
    # 元数据
    search_space: dict[str, Any]
    domain: str
    intent: str
    dependencies: list[str] = field(default_factory=list)
    
    # 生成的辅助文件
    test_cases: list[dict] = field(default_factory=list)
    config_yaml: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model.py": self.model_code,
            "loss.py": self.loss_code,
            "data_pipeline.py": self.data_pipeline_code,
            "train_loop.py": self.train_loop_code,
            "search_space.yaml": yaml.dump(self.search_space, allow_unicode=True),
            "domain": self.domain,
            "intent": self.intent,
            "dependencies": self.dependencies,
        }
    
    def save(self, output_dir: Path | str) -> None:
        """保存程序到目录"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {
            "model.py": self.model_code,
            "loss.py": self.loss_code,
            "data_pipeline.py": self.data_pipeline_code,
            "train_loop.py": self.train_loop_code,
            "search_space.yaml": yaml.dump(self.search_space, allow_unicode=True),
        }
        
        for filename, content in files.items():
            (output_dir / filename).write_text(content, encoding="utf-8")


class ProgramGenerator:
    """程序生成器"""
    
    def __init__(self, llm_client=None, domain: str = "general"):
        self.llm_client = llm_client
        self.domain = domain
        self.template = DOMAIN_TEMPLATES.get(domain, {})
        
    def generate(
        self,
        intent: str,
        data_schema: dict | None = None,
        constraints: dict | None = None,
        examples: list[dict] | None = None,
    ) -> GeneratedProgram:
        """
        根据自然语言意图生成完整程序
        
        Args:
            intent: 用户意图，如"蛋白质结构预测，低质量样本多"
            data_schema: 数据schema描述
            constraints: 约束条件（计算资源、延迟要求等）
            examples: Few-shot示例
            
        Returns:
            GeneratedProgram: 生成的程序包
        """
        # 构建Prompt
        prompt = self._build_prompt(intent, data_schema, constraints, examples)
        
        # 调用LLM生成代码
        response = self._call_llm(prompt)
        
        # 解析响应
        program = self._parse_response(response, intent)
        
        return program
    
    def _build_prompt(
        self,
        intent: str,
        data_schema: dict | None,
        constraints: dict | None,
        examples: list[dict] | None,
    ) -> str:
        """构建生成Prompt"""
        
        system_prompt = self.template.get("system_prompt", "")
        
        user_prompt = f"""
用户需求：{intent}

"""
        if data_schema:
            user_prompt += f"数据Schema：\n{yaml.dump(data_schema, allow_unicode=True)}\n"
        
        if constraints:
            user_prompt += f"约束条件：\n{yaml.dump(constraints, allow_unicode=True)}\n"
        
        # 添加Few-shot示例
        if examples:
            user_prompt += "\n参考示例：\n"
            for ex in examples:
                user_prompt += f"- 意图: {ex['intent']}\n  关键组件: {ex.get('components', [])}\n"
        
        user_prompt += """
请生成以下文件：

1. model.py: 模型架构定义
   - 必须继承 nn.Module
   - 包含 forward 方法
   - 添加类型注解
   - 包含 get_search_space 类方法返回可搜索参数

2. loss.py: 损失函数
   - 必须继承 nn.Module
   - forward接收(pred, target, **kwargs)
   - 支持reduction参数

3. data_pipeline.py: 数据加载
   - DataModule类，包含setup, train_dataloader, val_dataloader
   - 处理数据清洗和增强
   - 支持样本权重（如果有低质量数据）

4. train_loop.py: 训练循环
   - Trainer类，包含fit, validate, test
   - 支持混合精度训练
   - 支持梯度裁剪
   - 支持早停

5. search_space.yaml: 超参数搜索空间
   格式示例：
   model:
     num_layers:
       type: int
       low: 2
       high: 8
     hidden_dim:
       type: choice
       values: [256, 512, 1024]
   optimizer:
     lr:
       type: float
       low: 1e-4
       high: 1e-2
       log: true

请用 ```python filename.py 格式输出每个文件，最后给出yaml内容。
"""
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM生成代码"""
        if self.llm_client:
            return self.llm_client.generate(prompt)
        
        # 模拟LLM调用（实际项目中替换为真实调用）
        raise NotImplementedError("需要提供LLM客户端")
    
    def _parse_response(self, response: str, intent: str) -> GeneratedProgram:
        """解析LLM响应，提取代码文件"""
        
        # 提取Python代码块
        pattern = r'```python\s+(\w+\.py)\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        files = {}
        for filename, code in matches:
            files[filename] = code.strip()
        
        # 提取YAML
        yaml_pattern = r'```yaml\s*\n(.*?)```|search_space\.yaml:\s*\n(.*?)(?=\n\n|\Z)'
        yaml_matches = re.findall(yaml_pattern, response, re.DOTALL)
        search_space = {}
        for m in yaml_matches:
            yaml_content = m[0] or m[1]
            if yaml_content:
                try:
                    search_space = yaml.safe_load(yaml_content.strip())
                except:
                    pass
        
        # 提取依赖
        deps = self._extract_dependencies(files)
        
        return GeneratedProgram(
            model_code=files.get("model.py", ""),
            loss_code=files.get("loss.py", ""),
            data_pipeline_code=files.get("data_pipeline.py", ""),
            train_loop_code=files.get("train_loop.py", ""),
            search_space=search_space,
            domain=self.domain,
            intent=intent,
            dependencies=deps,
        )
    
    def _extract_dependencies(self, files: dict[str, str]) -> list[str]:
        """从代码中提取依赖包"""
        all_imports = []
        
        for code in files.values():
            # 匹配 import xxx 和 from xxx import
            imports = re.findall(r'^(?:import|from)\s+(\w+)', code, re.MULTILINE)
            all_imports.extend(imports)
        
        # 第三方库白名单
        third_party = {
            "torch", "torchvision", "torch_geometric", "dgl",
            "numpy", "pandas", "scipy", "sklearn",
            "lightning", "pytorch_lightning", "wandb", "tensorboard",
            "tqdm", "omegaconf", "hydra",
        }
        
        return list(set(all_imports) & third_party)
    
    def fix_code(
        self,
        program: GeneratedProgram,
        errors: list[dict],
        stage: str,
    ) -> GeneratedProgram:
        """
        根据QA错误修复代码
        
        Args:
            program: 原程序
            errors: 错误列表，每项包含file, line, message
            stage: 错误阶段 (static_analysis, unit_test, smoke_test)
            
        Returns:
            修复后的程序
        """
        prompt = f"""你之前生成的代码在{stage}阶段发现了错误，请修复。

原始需求：{program.intent}

错误列表：
"""
        for err in errors:
            prompt += f"- 文件: {err.get('file', 'unknown')}, 行: {err.get('line', 'unknown')}\n"
            prompt += f"  错误: {err.get('message', 'unknown')}\n"
            if 'suggestion' in err:
                prompt += f"  建议: {err['suggestion']}\n"
        
        prompt += "\n原始代码：\n"
        prompt += f"\n### model.py\n```python\n{program.model_code}\n```\n"
        prompt += f"\n### loss.py\n```python\n{program.loss_code}\n```\n"
        prompt += f"\n### data_pipeline.py\n```python\n{program.data_pipeline_code}\n```\n"
        prompt += f"\n### train_loop.py\n```python\n{program.train_loop_code}\n```\n"
        
        prompt += """
请输出修复后的完整代码，使用相同的格式。
"""
        
        response = self._call_llm(prompt)
        return self._parse_response(response, program.intent)