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
        编译完整的 ObjectiveSpec（旧接口，保留兼容）
        """
        system_prompt = """你是一个专业的 AutoML 系统设计师。请将用户的业务需求编译为详细的机器学习任务规范。"""

        user_prompt = f"""用户业务需求：{user_goal}
必须保留的要素：{', '.join(must_keep) if must_keep else '无'}
可以调整的要素：{', '.join(can_change) if can_change else '无'}
不能接受的错误类型：{', '.join(worst_errors) if worst_errors else '无'}
优化优先级：{priority}"""
        
        if data_schema:
            user_prompt += f"\n数据 schema：{json.dumps(data_schema, ensure_ascii=False, indent=2)}\n"
        
        output_template = """
请按以下JSON格式输出：
{
    "task_family": "binary_classification/regression/time_series等",
    "target_column": "目标列名",
    "target_description": "目标变量的业务定义",
    "primary_metric": "f1/recall/precision/mae等",
    "surrogate_loss": "cross_entropy/focal_loss/mse等",
    "validation_strategy": "train_test_split/time_series_split等",
    "constraints": {},
    "recommended_models": [],
    "feature_engineering_hints": [],
    "potential_issues": []
}"""
        
        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + output_template},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return ObjectiveCompileResult(**data)
        except Exception as e:
            return self._fallback_compile(response, str(e))
    
    def compile_personalized_plan(
        self,
        user_goal: str,
        must_keep: list[str] | None = None,
        worst_errors: list[str] | None = None,
        priority: str = "quality",
        data_schema: dict | None = None,
    ) -> dict:
        """
        编译个性化训练方案（双层输出）
        
        核心设计：
        - 业务层：让非专业人士确信需求被理解、方案是个性化的
        - 技术层：让ML专家可以审查、反馈、指导
        
        LLM 不是在填模版，而是真正理解业务后生成定制方案。
        """
        system_prompt = """你是 VibeML 的核心 AI 引擎。你的任务是将用户模糊的自然语言业务需求，转化为**个性化的、针对具体场景定制的**机器学习训练方案。

你不是在填标准模版。你需要真正理解用户的业务场景，然后设计一套**为这个场景量身定制**的训练流程——包括但不限于：特殊的数据处理方法、定制的损失函数、甚至改造模型架构。

你需要输出两个层次的信息：

## 业务层（给非技术人员看）
用通俗易懂的语言：
1. **我的理解**：用自己的话复述用户的需求，让用户确认你真的理解了
2. **个性化方案概述**：用类比或业务语言解释你打算怎么做，为什么这个方案是为他们的场景定制的（而不是通用方案）
3. **核心承诺**：你的方案会特别保证哪几点（用业务语言，不用技术术语）
4. **与标准方案的区别**：一句话说明这个定制方案比"通用做法"好在哪里

## 技术层（给ML专家展开查看）
为每个技术决策提供：
1. **数据策略**：数据清洗、特征工程、采样策略、增强方法——要具体到为什么这么做，不只是列名称
2. **模型策略**：架构选择及定制点、为什么选这个而不是那个
3. **损失函数**：设计理念、公式描述、各项权重的业务依据
4. **评估方案**：用什么指标、为什么这个指标最能反映业务价值
5. **风险与备选**：可能遇到的问题及应对方案

关键原则：
- 每个技术决策都要有**业务依据**（不是"因为这是best practice"，而是"因为用户说了xxx，所以我们需要xxx"）
- 如果标准做法就够用，说明为什么够用；如果需要定制，具体说明定制了什么
- 诚实面对不确定性——如果某些决策需要看到数据才能确定，明确说出来"""

        user_prompt = f"""用户的业务需求：
"{user_goal}"
"""
        
        if must_keep:
            user_prompt += f"\n用户强调必须保留：{', '.join(must_keep)}"
        if worst_errors:
            user_prompt += f"\n用户最不能接受的错误：{', '.join(worst_errors)}"
        if priority and priority != "quality":
            priority_map = {"latency": "速度优先", "cost": "成本优先"}
            user_prompt += f"\n优化偏好：{priority_map.get(priority, priority)}"
        if data_schema:
            user_prompt += f"\n\n数据结构：\n{json.dumps(data_schema, ensure_ascii=False, indent=2)}"

        user_prompt += """

请严格按以下 JSON 输出（中文）：

{
    "business_layer": {
        "understanding": "用你自己的话复述用户的核心需求（1-2句）",
        "personalized_approach": "用通俗语言描述你的定制方案（2-3句，用类比让非技术人员理解）",
        "core_promises": [
            "承诺1：用业务语言描述",
            "承诺2：用业务语言描述"
        ],
        "vs_standard": "一句话：这个定制方案比通用做法好在哪"
    },
    "technical_layer": {
        "data_strategy": {
            "title": "数据处理策略的一句话标题",
            "reasoning": "为什么为这个场景这样处理数据",
            "details": "具体的数据清洗、特征工程、采样、增强方法"
        },
        "model_strategy": {
            "title": "模型策略的一句话标题",
            "reasoning": "为什么为这个场景选择/定制这个架构",
            "details": "具体的模型架构、定制点、超参范围"
        },
        "loss_function": {
            "title": "损失函数设计的一句话标题",
            "reasoning": "损失函数的业务依据",
            "details": "具体的损失函数设计、权重分配、公式"
        },
        "evaluation": {
            "title": "评估方案的一句话标题",
            "reasoning": "为什么用这个指标来衡量成功",
            "details": "具体的评估指标、验证策略、阈值"
        },
        "risks_and_alternatives": {
            "title": "风险评估",
            "items": [
                {"risk": "风险描述", "mitigation": "应对方案"}
            ]
        }
    },
    "confidence": 0.85,
    "needs_more_info": ["如果需要更多信息才能确定的点"]
}"""

        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=3000,
        )
        
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return data
        except Exception:
            # 解析失败 → 返回原始文本作为 understanding
            return {
                "business_layer": {
                    "understanding": "我正在分析你的需求，但生成结构化方案时遇到问题。",
                    "personalized_approach": response[:500] if response else "请重新描述你的需求。",
                    "core_promises": [],
                    "vs_standard": "",
                },
                "technical_layer": {},
                "confidence": 0.3,
                "needs_more_info": ["请重新描述你的需求，我会再次尝试生成方案"],
            }
    
    def clarify_ambiguity(
        self,
        user_goal: str,
        must_keep: list[str],
        worst_errors: list[str],
    ) -> tuple[bool, list[str], list[str]]:
        """
        使用 LLM 理解用户输入：
        1. 先判断是否是有意义的 ML/数据科学相关需求
        2. 如果是，再判断需求是否足够清晰
        
        Returns:
            (是否需要澄清, 原因列表, 追问列表)
        """
        system_prompt = """你是 VibeML 的需求分析师。你的职责是判断用户输入是否是一个**可执行的机器学习/数据科学需求**。

**核心原则：用户的自然语言描述本身就是需求的完整来源。** 不要因为某些表单字段为空就判定为"缺少"——用户把约束、偏好、目标都写在了描述里。你应该从描述中理解和提取这些信息。

### 第一步：这是 ML/AI 相关的需求吗？
- 寒暄（"hi"、"你好"）→ 不是
- 无关问题 → 不是
- 任何涉及数据分析、预测、分类、检测、模型训练、图像识别等的描述 → 是

### 第二步：如果是 ML 需求，描述是否足够清晰到可以开始设计方案？
判断标准（从描述文本中提取，不依赖额外表单字段）：
- 能否推断出任务类型？（如用户提到 YOLO/检测 → 目标检测）
- 是否描述了业务场景或数据类型？
- 是否提到了挑战、约束或期望？

**重要：如果用户的描述足够详细（超过50字且包含具体的任务/场景描述），即使没有额外填写约束字段，也应该判断为 is_ambiguous=false。** 用户已经在自然语言中表达了他们的需求。"""

        user_prompt = f"""用户输入：
\"\"\"{user_goal}\"\"\"
"""
        # 只在用户明确提供了额外约束时才附加
        extras = []
        if must_keep:
            extras.append(f"用户额外强调必须保留：{', '.join(must_keep)}")
        if worst_errors:
            extras.append(f"用户额外强调不可接受的错误：{', '.join(worst_errors)}")
        if extras:
            user_prompt += "\n" + "\n".join(extras) + "\n"

        user_prompt += """
请严格按以下 JSON 格式输出（不要输出其他内容）：

{
    "is_ml_request": true或false,
    "is_ambiguous": true或false,
    "reasons": ["如果模糊，说明具体缺什么信息"],
    "follow_up_questions": ["如果模糊，生成针对性追问"],
    "friendly_message": "一句话总结你对用户需求的理解"
}"""

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
            
            is_ml_request = data.get("is_ml_request", False)
            
            if not is_ml_request:
                # 不是 ML 需求 → 直接标记为需澄清，附上引导语
                friendly = data.get("friendly_message", "")
                reasons = [friendly] if friendly else ["输入不包含可识别的机器学习/数据科学需求"]
                questions = data.get("follow_up_questions", [
                    "请描述你的业务场景，例如：'我有一份电商数据，想预测哪些客户可能流失'",
                    "你想解决什么问题？（预测、分类、异常检测等）",
                    "你有什么数据？",
                ])
                return True, reasons, questions
            
            # 是 ML 需求
            is_ambiguous = data.get("is_ambiguous", True)
            
            # 关键保护：如果 LLM 认为这是 ML 需求，且用户描述足够详细（>50字），
            # 即使 LLM 标记为"模糊"，也强制认为足够清晰。
            # 原因：LLM 倾向于保守地要求更多信息，但 VibeML 的设计理念是
            # "从自然语言中提取需求"，而不是要求用户填写更多表单字段。
            if is_ambiguous and len(user_goal.strip()) >= 50:
                print(f"[INFO] LLM 判断为模糊但描述已足够详细({len(user_goal.strip())}字)，"
                      f"覆盖为清晰。LLM reasons: {data.get('reasons', [])}")
                is_ambiguous = False
            
            return (
                is_ambiguous,
                data.get("reasons", []),
                data.get("follow_up_questions", []),
            )
        except Exception:
            # JSON 解析失败 → 安全侧，标记为模糊
            return True, ["无法解析需求，请重新描述"], [
                "请用一两句话描述：你有什么数据、想解决什么业务问题？",
            ]
    
    def explore_data(
        self,
        columns_info: list[dict],
        sample_rows: list[dict],
        n_rows: int,
        n_cols: int,
        filename: str,
        user_goal: str | None = None,
    ) -> dict:
        """
        LLM 驱动的数据探索 Agent
        
        让 LLM 像一个数据科学家一样"打开数据看一看"：
        - 理解每列的业务含义（不是靠列名关键词匹配）
        - 识别哪些是 feature，哪些是 target
        - 发现数据质量问题
        - 给出数据预处理建议
        """
        system_prompt = """你是一位资深数据科学家。你正在查看一份新的数据集，需要快速理解它并给出分析。

你的任务：
1. **理解每一列的业务含义** — 不要只看列名，还要看数据分布、取值范围、样本值来推断
2. **识别 target（目标变量）** — 这是监督学习要预测的列。判断依据：
   - 列名暗示（如 target, label, churn, fraud, price 等）
   - 数据特征（二分类目标通常只有 0/1 或 yes/no）
   - 业务语境（如果用户说了目标，优先用用户说的）
3. **识别 features（特征）** — 哪些列可以用于预测，哪些应该排除（如 ID、时间戳索引）
4. **发现潜在问题** — 缺失值、异常值、类别不平衡、可能的数据泄漏
5. **给出建议** — 特征工程方向、需要注意的事项

请用非专业人士能理解的语言描述你的发现，但同时提供技术细节给专家参考。"""

        # 构建数据摘要供 LLM 查看
        col_descriptions = []
        for col in columns_info:
            desc = f"- **{col['name']}** (类型: {col['dtype']}, 缺失: {col['missing_count']}/{n_rows}, 唯一值: {col['unique_count']})"
            if col.get('sample_values'):
                desc += f"\n  样本值: {col['sample_values'][:5]}"
            if col.get('statistics'):
                stats = col['statistics']
                if 'mean' in stats:
                    desc += f"\n  统计: 均值={stats['mean']:.2f}, 标准差={stats.get('std', 0):.2f}, 范围=[{stats.get('min', '?')}, {stats.get('max', '?')}]"
                elif 'top_categories' in stats:
                    top = list(stats['top_categories'].items())[:3]
                    desc += f"\n  分布: {', '.join(f'{k}({v})' for k, v in top)}"
            col_descriptions.append(desc)

        user_prompt = f"""数据集: {filename}
规模: {n_rows} 行 × {n_cols} 列

{'用户目标: ' + user_goal if user_goal else '用户尚未描述目标'}

=== 列信息 ===
{chr(10).join(col_descriptions)}

=== 前几行数据样本 ===
{json.dumps(sample_rows[:5], ensure_ascii=False, indent=2)}

请分析这份数据，严格按以下 JSON 格式输出：

{{
    "data_understanding": {{
        "summary": "一句话总结这是什么数据",
        "domain": "数据所属领域（如电商、金融、医疗等）",
        "columns_analysis": [
            {{
                "name": "列名",
                "business_meaning": "这列在业务上代表什么",
                "role": "target / feature / id / datetime_index / exclude",
                "role_reason": "为什么判断为这个角色",
                "data_quality": "数据质量评价",
                "engineering_hint": "特征工程建议（如果是feature的话）"
            }}
        ]
    }},
    "target_recommendation": {{
        "column": "推荐的目标列名",
        "confidence": 0.9,
        "reasoning": "为什么选这列作为目标",
        "task_type": "binary_classification / multiclass / regression / 等",
        "alternatives": ["备选目标列（如果有的话）"]
    }},
    "data_quality_report": {{
        "overall_score": 0.8,
        "issues": [
            {{
                "severity": "high / medium / low",
                "description": "问题描述",
                "affected_columns": ["列名"],
                "suggestion": "建议的处理方式"
            }}
        ]
    }},
    "business_summary": "用通俗语言告诉用户：这份数据是关于什么的、哪些信息可以用来预测、系统打算预测什么",
    "suggested_next_steps": ["建议的下一步操作"]
}}"""

        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except Exception:
            return {
                "data_understanding": {
                    "summary": "数据分析完成，但结构化输出失败。",
                    "raw_analysis": response[:2000],
                },
                "business_summary": response[:500] if response else "数据分析失败，请重试。",
                "suggested_next_steps": ["请重试数据探索"],
            }
    
    def suggest_data_sources(
        self,
        user_goal: str,
        task_type: str | None = None,
    ) -> dict:
        """
        根据用户目标，推荐公开数据集 + 合成数据方案
        
        LLM 负责：
        1. 理解用户场景，生成搜索关键词
        2. 推荐 HuggingFace / GitHub / ModelScope 上可能匹配的数据集
        3. 描述一个可以 AI 生成的合成数据方案
        """
        system_prompt = """你是一个数据工程师。用户描述了一个 ML 任务但没有提供数据。
你需要帮他们找到合适的公开数据集，或设计一个合成数据方案。

你对公开数据集生态非常熟悉：
- Hugging Face Datasets（huggingface.co/datasets）
- GitHub 上的开源数据集
- ModelScope（modelscope.cn）
- Kaggle 经典数据集
- UCI ML Repository

请根据用户的业务场景推荐最匹配的数据集，并给出具体的数据集名称和链接。
同时设计一个合成数据的 schema，以防公开数据不可用。"""

        user_prompt = f"""用户的业务目标：{user_goal}
{f'任务类型：{task_type}' if task_type else ''}

请严格按以下 JSON 输出：

{{
    "search_keywords": {{
        "en": ["English search keywords for datasets"],
        "zh": ["中文搜索关键词"]
    }},
    "recommended_datasets": [
        {{
            "name": "数据集名称",
            "source": "huggingface / github / modelscope / kaggle / uci",
            "url": "完整链接",
            "description": "数据集描述及与用户场景的匹配度",
            "relevance_score": 0.9,
            "columns_preview": "关键列说明",
            "size_hint": "大约多少行/多大",
            "license": "开源协议"
        }}
    ],
    "synthetic_data_plan": {{
        "description": "用通俗语言描述合成数据方案",
        "schema": [
            {{
                "column_name": "列名",
                "dtype": "int / float / str / bool / datetime",
                "description": "这列代表什么",
                "generation_strategy": "如何生成（分布、范围、规则等）"
            }}
        ],
        "target_column": "目标列名",
        "suggested_rows": 1000,
        "caveats": "合成数据的局限性说明"
    }},
    "recommendation": "综合建议：优先用哪个公开数据集，还是合成数据更合适"
}}"""

        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=2500,
        )

        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except Exception:
            return {
                "search_keywords": {"en": [user_goal], "zh": [user_goal]},
                "recommended_datasets": [],
                "synthetic_data_plan": None,
                "recommendation": response[:500] if response else "无法生成推荐",
            }

    def generate_synthetic_data(
        self,
        user_goal: str,
        schema: list[dict],
        target_column: str,
        n_rows: int = 500,
    ) -> str:
        """
        用 LLM 生成符合业务场景的合成训练数据（CSV 格式）
        """
        columns_desc = "\n".join(
            f"- {col['column_name']} ({col['dtype']}): {col['description']} — 生成策略: {col.get('generation_strategy', '自动')}"
            for col in schema
        )

        system_prompt = """你是一个数据生成专家。请根据用户的业务场景和 schema，生成真实感强的合成训练数据。

要求：
1. 数据要符合业务逻辑（不是随机噪音）
2. 类别分布要合理（如二分类目标约 20-30% 正样本）
3. 特征之间要有合理的相关性
4. 输出纯 CSV 格式（含表头），不要 markdown 代码块
5. 每行一条数据，逗号分隔"""

        user_prompt = f"""业务场景：{user_goal}

数据 Schema：
{columns_desc}

目标列：{target_column}
生成行数：{n_rows}

请直接输出 CSV 内容（含表头行），不要任何其他文字："""

        response = self.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=4000,
        )

        # 清理可能的 markdown 包裹
        csv_text = response.strip()
        if csv_text.startswith("```"):
            lines = csv_text.split("\n")
            # 去掉首尾 ``` 行
            lines = [l for l in lines if not l.strip().startswith("```")]
            csv_text = "\n".join(lines)

        return csv_text

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
