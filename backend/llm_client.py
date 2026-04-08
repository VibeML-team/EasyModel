"""
LLM 客户端 - 用于意图解析和Planning

支持 OpenAI 兼容的 API 接口
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field


class LLMConfig:
    """LLM 配置"""
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME") or "gpt-4o-mini"
        
        if not self.api_key:
            raise ValueError(
                "LLM API Key 未设置。请设置 LLM_API_KEY 环境变量或在初始化时传入。"
            )


class IntentParseResult(BaseModel):
    """意图解析结果"""
    task_family: Literal[
        "binary_classification", "multiclass_classification", "regression",
        "time_series", "ranking", "anomaly_detection", "clustering"
    ] = Field(description="任务类型")
    target_description: str = Field(description="目标变量的业务含义")
    target_column_hint: str = Field(description="建议的目标列名（英文）")
    primary_metric: Literal[
        "accuracy", "precision", "recall", "f1", "auc",
        "precision_at_k", "mae", "rmse", "mape", "r2"
    ] = Field(description="主要评估指标")
    reasoning: str = Field(description="选择上述结论的推理过程")


class ObjectiveCompileResult(BaseModel):
    """Objective 编译结果"""
    task_family: str = Field(description="任务类型")
    target_column: str | None = Field(description="目标列名")
    target_description: str = Field(description="目标变量的业务定义")
    primary_metric: str = Field(description="主要评估指标")
    surrogate_loss: str = Field(description="推荐的损失函数")
    validation_strategy: str = Field(description="验证策略")
    constraints: dict[str, Any] = Field(default_factory=dict, description="业务约束")
    recommended_models: list[str] = Field(default_factory=list, description="推荐的模型列表")
    feature_engineering_hints: list[str] = Field(default_factory=list, description="特征工程建议")
    potential_issues: list[str] = Field(default_factory=list, description="潜在问题和注意事项")


class LLMClient:
    """LLM 客户端"""
    
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.client = httpx.Client(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
    
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        调用聊天补全API
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如json_schema）
        """
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if response_format:
            payload["response_format"] = response_format
        
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def parse_intent(
        self,
        user_goal: str,
        must_keep: list[str],
        worst_errors: list[str],
        data_schema: dict | None = None,
    ) -> IntentParseResult:
        """
        解析用户意图
        
        使用LLM分析用户的自然语言需求，识别任务类型和评估指标
        """
        system_prompt = """你是一个专业的机器学习任务分析师。你的任务是将用户的业务需求解析为结构化的ML任务定义。

请分析用户的描述，识别：
1. 任务类型（分类/回归/时序等）
2. 目标变量的业务含义
3. 最适合的评估指标
4. 约束条件

请用中文给出推理过程，但字段值使用英文枚举值。"""

        user_prompt = f"""用户业务需求：{user_goal}

必须保留的要素：{', '.join(must_keep) if must_keep else '无'}

不能接受的错误类型：{', '.join(worst_errors) if worst_errors else '无'}
"""
        
        if data_schema:
            user_prompt += f"\n数据 schema：{json.dumps(data_schema, ensure_ascii=False)}\n"
        
        # 检查是否支持结构化输出
        supports_structured = self._check_structured_output_support()
        
        if supports_structured:
            # 使用结构化输出
            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "intent_parse_result",
                        "schema": IntentParseResult.model_json_schema(),
                    }
                }
            )
            return IntentParseResult.model_validate_json(response)
        else:
            # 回退到普通输出
            user_prompt += "\n请按以下JSON格式输出：\n" + IntentParseResult.model_json_schema()
            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            # 提取JSON
            try:
                json_str = self._extract_json(response)
                return IntentParseResult.model_validate_json(json_str)
            except:
                # 如果解析失败，返回默认值
                return IntentParseResult(
                    task_family="binary_classification",
                    target_description="目标变量",
                    target_column_hint="target",
                    primary_metric="f1",
                    reasoning="解析失败，使用默认配置",
                )
    
    def compile_objective(
        self,
        user_goal: str,
        must_keep: list[str],
        can_change: list[str],
        worst_errors: list[str],
        priority: str,
        data_schema: dict | None = None,
    ) -> ObjectiveCompileResult:
        """
        编译完整的 ObjectiveSpec
        
        生成详细的训练配置建议
        """
        system_prompt = """你是一个专业的 AutoML 系统设计师。请将用户的业务需求编译为详细的机器学习任务规范。

你需要提供：
1. 具体的任务类型和配置
2. 推荐的评估指标和损失函数
3. 适合的模型列表（按优先级排序）
4. 数据处理和特征工程建议
5. 潜在问题和注意事项

请确保输出是实际可执行的配置。"""

        user_prompt = f"""用户业务需求：{user_goal}

必须保留的要素：{', '.join(must_keep) if must_keep else '无'}
可以调整的要素：{', '.join(can_change) if can_change else '无'}

不能接受的错误类型：{', '.join(worst_errors) if worst_errors else '无'}

优化优先级：{priority}（quality=质量优先, latency=速度优先, cost=成本优先）
"""
        
        if data_schema:
            user_prompt += f"\n数据 schema：{json.dumps(data_schema, ensure_ascii=False, indent=2)}\n"
        
        # 构建详细的输出提示
        output_template = """
请按以下JSON格式输出你的分析结果：

{
    "task_family": "任务类型，如 binary_classification/regression/time_series",
    "target_column": "建议的目标列名（英文）",
    "target_description": "目标变量的业务定义",
    "primary_metric": "主要评估指标，如 f1/recall/precision/mae/rmse",
    "surrogate_loss": "推荐的损失函数，如 cross_entropy/focal_loss/mse",
    "validation_strategy": "验证策略：train_test_split/time_series_split/group_split",
    "constraints": {
        "key": "value"  // 提取的业务约束，如 daily_budget, latency_ms
    },
    "recommended_models": ["xgboost", "lightgbm", "random_forest"],  // 按优先级排序
    "feature_engineering_hints": ["建议的特征工程操作"],
    "potential_issues": ["潜在问题和注意事项"]
}
"""
        
        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + output_template},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        
        # 尝试解析JSON响应
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return ObjectiveCompileResult(**data)
        except Exception as e:
            # 如果解析失败，使用LLM进行第二次解析
            return self._fallback_compile(response, str(e))
    
    def clarify_ambiguity(
        self,
        user_goal: str,
        must_keep: list[str],
        worst_errors: list[str],
    ) -> tuple[bool, list[str], list[str]]:
        """
        检测需求中的模糊性并生成追问
        
        Returns:
            (是否模糊, 原因列表, 追问列表)
        """
        system_prompt = """你是一个需求分析师。请分析用户的ML任务描述是否足够清晰。

判断标准：
- 是否明确了目标变量？
- 是否明确了任务类型（分类/回归等）？
- 是否提供了约束条件？
- 是否明确了评估指标偏好？

如果需要澄清，请生成3-5个关键问题。"""

        user_prompt = f"""用户描述：{user_goal}

必须保留：{', '.join(must_keep) if must_keep else '无'}

不可接受错误：{', '.join(worst_errors) if worst_errors else '无'}

请分析以上描述是否足够清晰，并按以下JSON格式输出：

{{
    "is_ambiguous": true/false,
    "reasons": ["模糊原因1", "模糊原因2"],
    "follow_up_questions": ["追问问题1", "追问问题2"]
}}"""

        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return (
                data.get("is_ambiguous", False),
                data.get("reasons", []),
                data.get("follow_up_questions", []),
            )
        except:
            # 解析失败返回默认值
            is_ambiguous = len(user_goal) < 20 or not must_keep or not worst_errors
            reasons = []
            questions = []
            
            if len(user_goal) < 20:
                reasons.append("目标描述过短")
                questions.append("请详细描述业务场景和成功标准。")
            
            if not must_keep:
                reasons.append("缺少必须保留的要素")
                questions.append("哪些业务要素绝对不能改变？")
            
            if not worst_errors:
                reasons.append("缺少错误类型定义")
                questions.append("哪种错误对业务影响更大（误报 vs 漏报）？")
            
            return is_ambiguous, reasons, questions
    
    def _check_structured_output_support(self) -> bool:
        """检查是否支持结构化输出"""
        # OpenAI gpt-4 系列支持 structured outputs
        supported_models = [
            "gpt-4", "gpt-4o", "gpt-4o-mini",
            "gpt-4-turbo", "gpt-4-turbo-preview"
        ]
        return any(m in self.config.model_name.lower() for m in supported_models)
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON"""
        # 尝试找到JSON代码块
        import re
        
        # 查找 ```json ... ``` 格式
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # 查找单独的 {...}
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json_match.group(1)
        
        return text
    
    def _fallback_compile(self, raw_response: str, error_msg: str) -> ObjectiveCompileResult:
        """编译失败时的回退处理"""
        # 使用LLM尝试从文本中提取信息
        fix_prompt = f"""之前的JSON解析失败了（错误：{error_msg}）。

请从以下文本中提取关键信息，并输出有效的JSON：

{raw_response}

请只输出JSON，不要其他内容："""

        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.1,
            )
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return ObjectiveCompileResult(**data)
        except:
            # 最终回退：返回默认配置
            return ObjectiveCompileResult(
                task_family="binary_classification",
                target_column="target",
                target_description="目标变量",
                primary_metric="f1",
                surrogate_loss="cross_entropy",
                validation_strategy="train_test_split",
                recommended_models=["xgboost", "random_forest", "logistic_regression"],
                potential_issues=["使用默认配置，建议人工复核"],
            )


# 全局LLM客户端实例
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """获取全局LLM客户端"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def init_llm_client(api_key: str | None = None, base_url: str | None = None, model_name: str | None = None):
    """初始化LLM客户端（使用自定义配置）"""
    global _llm_client
    config = LLMConfig(api_key=api_key, base_url=base_url, model_name=model_name)
    _llm_client = LLMClient(config)
    return _llm_client
