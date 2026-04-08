"""
Intent Compiler V2 - 意图编译器 (用户可介入层)

核心设计原则:
- 用户可介入的是"意图"，不是"机制"
- 系统理解必须显式展示，用户可修正
- 技术细节(optimizer/lr等)默认隐藏
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualitySpeedCost(Enum):
    """质量/速度/成本权衡 - 用户可理解的选择"""
    QUALITY_FIRST = "quality"      # 质量优先
    BALANCED = "balanced"          # 平衡
    SPEED_FIRST = "speed"          # 速度/成本优先
    EDGE_DEPLOY = "edge"           # 边缘部署


class AugmentationStyle(Enum):
    """数据增强风格 - 语义化表达"""
    CONSERVATIVE = "conservative"  # 保守：只允许微小变化
    STANDARD = "standard"          # 标准：适度变化
    AGGRESSIVE = "aggressive"      # 激进：允许大幅变化


class ErrorPreference(Enum):
    """错误偏好 - 业务语义"""
    PREFER_FALSE_POSITIVE = "prefer_fp"   # 宁可误报
    PREFER_FALSE_NEGATIVE = "prefer_fn"   # 宁可漏报
    BALANCED = "balanced"                  # 平衡


@dataclass
class ClarificationQuestion:
    """澄清问题"""
    question_id: str = ""                # 问题ID
    question_text: str = ""              # 问题内容
    question_type: str = "open"          # open/multiple_choice/confirm
    options: list[str] = field(default_factory=list)  # 选项（如果是选择题）
    context: str = ""                    # 为什么问这个问题
    is_answered: bool = False            # 是否已回答
    answer: str = ""                     # 用户回答
    priority: int = 0                    # 优先级（0最高）


@dataclass
class AmbiguityDetection:
    """模糊性检测结果"""
    is_ambiguous: bool = False           # 是否模糊
    ambiguity_score: float = 0.0         # 模糊度分数 (0-1)
    missing_elements: list[str] = field(default_factory=list)  # 缺失的关键要素
    unclear_aspects: list[str] = field(default_factory=list)   # 不清晰的方面
    suggested_questions: list[ClarificationQuestion] = field(default_factory=list)  # 建议追问
    can_proceed_with_caution: bool = False  # 是否可以谨慎继续


@dataclass
class IntentCard:
    """
    需求编译卡 - 系统理解的用户意图，可观测可修改
    
    这是用户和系统之间的"合同"，必须在训练前显式确认
    """
    # === 基础意图 ===
    task_description: str = ""           # 用户原始描述
    task_type: str = ""                 # 系统识别的任务类型
    
    # === 约束与偏好 ===
    must_keep: list[str] = field(default_factory=list)      # 必须保留的元素
    can_change: list[str] = field(default_factory=list)     # 允许变化的元素
    quality_speed_cost: QualitySpeedCost = QualitySpeedCost.BALANCED
    error_preference: ErrorPreference = ErrorPreference.BALANCED
    
    # === 数据理解 ===
    data_summary: dict = field(default_factory=dict)        # 数据概览
    sample_importance: dict = field(default_factory=dict)   # 样本重要性标注
    augmentation_style: AugmentationStyle = AugmentationStyle.STANDARD
    
    # === 验收标准 ===
    success_criteria: list[str] = field(default_factory=list)   # 怎样算成功
    unacceptable_errors: list[str] = field(default_factory=list)  # 哪些错误不能接受
    
    # === 系统推理过程 ===
    system_reasoning: str = ""           # 系统为什么这样理解
    confidence_score: float = 0.0        # 系统置信度
    ambiguity_flags: list[str] = field(default_factory=list)  # 不确定的地方（简化列表）
    
    # === 模糊性检测与澄清 ===
    ambiguity_detection: AmbiguityDetection = field(default_factory=AmbiguityDetection)
    clarification_history: list[ClarificationQuestion] = field(default_factory=list)  # 澄清历史
    
    def to_user_friendly_dict(self) -> dict[str, Any]:
        """转换为前端展示的友好格式"""
        result = {
            "task": {
                "description": self.task_description,
                "type": self.task_type,
                "confidence": self.confidence_score,
            },
            "constraints": {
                "must_keep": self.must_keep,
                "can_change": self.can_change,
                "priority": self.quality_speed_cost.value,
                "error_preference": self.error_preference.value,
            },
            "data": {
                "summary": self.data_summary,
                "augmentation_style": self.augmentation_style.value,
            },
            "acceptance": {
                "success_criteria": self.success_criteria,
                "unacceptable_errors": self.unacceptable_errors,
            },
            "system_notes": {
                "reasoning": self.system_reasoning,
                "uncertainties": self.ambiguity_flags,
            },
            "clarification": {
                "is_ambiguous": self.ambiguity_detection.is_ambiguous,
                "ambiguity_score": self.ambiguity_detection.ambiguity_score,
                "missing_elements": self.ambiguity_detection.missing_elements,
                "unclear_aspects": self.ambiguity_detection.unclear_aspects,
                "suggested_questions": [
                    {
                        "id": q.question_id,
                        "text": q.question_text,
                        "type": q.question_type,
                        "options": q.options,
                        "context": q.context,
                        "priority": q.priority,
                    }
                    for q in self.ambiguity_detection.suggested_questions
                ],
                "can_proceed": self.ambiguity_detection.can_proceed_with_caution,
            }
        }
        return result


@dataclass
class DataConstructionCard:
    """
    数据与样本构造卡 - 用户可筛选、配对、标记重要性
    
    这是最能体现业务价值的介入点
    """
    # === 数据概览 ===
    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    data_quality_issues: list[dict] = field(default_factory=list)
    
    # === 样本层面介入 ===
    # 用户可以标记哪些样本重要/不重要
    high_priority_samples: list[str] = field(default_factory=list)   # 样本ID列表
    excluded_samples: list[str] = field(default_factory=list)        # 被用户排除的样本
    
    # === 配对关系 (针对生成/编辑任务) ===
    pairings: list[dict] = field(default_factory=list)               # 输入-输出配对
    pairing_validation_status: str = "auto"                          # auto / user_verified
    
    # === 数据处理语义化表达 ===
    # 用户不需要知道random_crop，只需要知道"主体位置能否变化"
    allowed_variations: list[str] = field(default_factory=list)      # 允许的变化类型
    forbidden_variations: list[str] = field(default_factory=list)    # 禁止的变化类型
    augmentation_style: AugmentationStyle = AugmentationStyle.STANDARD  # 增强风格
    
    # === 预览 ===
    sample_previews: list[dict] = field(default_factory=list)        # 样本预览
    
    def to_user_friendly_dict(self) -> dict[str, Any]:
        """转换为前端展示"""
        return {
            "overview": {
                "total": self.total_samples,
                "train": self.train_samples,
                "val": self.val_samples,
                "issues": self.data_quality_issues,
            },
            "sample_control": {
                "high_priority_count": len(self.high_priority_samples),
                "excluded_count": len(self.excluded_samples),
                "can_review": True,  # 用户可以查看和修改
            },
            "variations": {
                "allowed": self.allowed_variations,
                "forbidden": self.forbidden_variations,
                "style": self.augmentation_style.value,  # conservative/standard/aggressive
            },
            "pairings": {
                "count": len(self.pairings),
                "status": self.pairing_validation_status,
            },
            "previews": self.sample_previews[:10],  # 只展示前10个
        }


@dataclass
class TrainingPlanCard:
    """
    训练方案卡 - 系统选择的路线，可观测有限介入
    
    用户看到的是"策略选择"，不是技术参数
    """
    # === 路线选择 ===
    approach: str = ""                   # e.g., "fine-tuning", "distillation", "paired-edit"
    approach_reasoning: str = ""         # 为什么选择这条路线
    
    # === 目标偏好 ===
    primary_goal: str = ""               # 主要目标
    secondary_goals: list[str] = field(default_factory=list)
    
    # === 方案级选择 (用户可改) ===
    quality_speed_cost: QualitySpeedCost = QualitySpeedCost.BALANCED
    deployment_target: str = "cloud"     # cloud / edge / mobile
    
    # === 隐含的机制层配置 (默认不展示) ===
    _mechanism_config: dict = field(default_factory=dict, repr=False)
    
    def to_user_friendly_dict(self, expose_mechanism: bool = False) -> dict[str, Any]:
        """转换为前端展示"""
        result = {
            "approach": {
                "name": self.approach,
                "reasoning": self.approach_reasoning,
            },
            "goals": {
                "primary": self.primary_goal,
                "secondary": self.secondary_goals,
            },
            "preferences": {
                "quality_speed_cost": self.quality_speed_cost.value,
                "deployment": self.deployment_target,
            },
        }
        
        # 高级模式才暴露机制层
        if expose_mechanism:
            result["_mechanism"] = self._mechanism_config
        
        return result


@dataclass
class TrainingObservation:
    """
    训练中的观测 - 进度、曲线、风险提示
    
    用户在这里做"偏好校正"和"checkpoint选择"
    """
    # === 进度 ===
    stage: str = ""                      # preparing / training / validating / completed
    current_step: int = 0
    total_steps: int = 0
    eta_seconds: int = 0
    
    # === 核心曲线 ===
    metrics_history: dict = field(default_factory=dict)   # train/val曲线
    best_checkpoint_step: int = 0
    best_checkpoint_metric: float = 0.0
    
    # === 中间预览 (针对生成任务) ===
    intermediate_outputs: list[dict] = field(default_factory=list)
    
    # === 风险提示 ===
    risk_alerts: list[dict] = field(default_factory=list)  # 过拟合、模式塌缩等
    
    # === 用户介入点 ===
    available_actions: list[str] = field(default_factory=list)  # pause/resume/stop/select_checkpoint
    
    def to_user_friendly_dict(self) -> dict[str, Any]:
        """转换为前端展示"""
        return {
            "progress": {
                "stage": self.stage,
                "current": self.current_step,
                "total": self.total_steps,
                "eta": self.eta_seconds,
                "percent": int(100 * self.current_step / max(1, self.total_steps)),
            },
            "metrics": self.metrics_history,
            "best_checkpoint": {
                "step": self.best_checkpoint_step,
                "metric": self.best_checkpoint_metric,
            },
            "previews": self.intermediate_outputs[-5:],  # 最近5个
            "alerts": self.risk_alerts,
            "actions": self.available_actions,
        }


@dataclass
class AcceptanceCard:
    """
    验收与权重定版 - 最终对比和选择
    """
    # === 候选版本 ===
    candidate_checkpoints: list[dict] = field(default_factory=list)
    
    # === 对比维度 ===
    comparison_dimensions: list[str] = field(default_factory=list)
    
    # === 用户决策 ===
    selected_checkpoint: str = ""        # 用户选择的版本
    user_feedback: list[dict] = field(default_factory=list)  # 用户对各版本的反馈
    
    # === 定版信息 ===
    final_model_path: str = ""
    deployment_recommendations: dict = field(default_factory=dict)
    
    def to_user_friendly_dict(self) -> dict[str, Any]:
        """转换为前端展示"""
        return {
            "candidates": self.candidate_checkpoints,
            "comparison": self.comparison_dimensions,
            "selected": self.selected_checkpoint,
            "feedback_summary": {
                "total_feedback": len(self.user_feedback),
                "positive": sum(1 for f in self.user_feedback if f.get("liked")),
            },
            "final": {
                "path": self.final_model_path,
                "deployment": self.deployment_recommendations,
            } if self.final_model_path else None,
        }


class IntentCompilerV2:
    """
    意图编译器 V2
    
    将自然语言编译为上述"卡片"，支持用户介入修正
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._init_default_client()
    
    def _init_default_client(self):
        """初始化默认LLM客户端（如果未提供）"""
        if self.llm_client is None:
            try:
                from backend.llm_client import get_llm_client
                self.llm_client = get_llm_client()
            except:
                self.llm_client = None
        
    def compile(
        self,
        intent: str,
        data_preview: dict | None = None,
        domain: str = "general",
    ) -> tuple[IntentCard, DataConstructionCard, TrainingPlanCard]:
        """
        编译意图为三张核心卡片
        
        Returns:
            (intent_card, data_card, plan_card)
        """
        # 尝试使用LLM生成卡片
        if self.llm_client:
            try:
                return self._compile_with_llm(intent, data_preview, domain)
            except Exception as e:
                print(f"LLM编译失败，使用规则回退: {e}")
        
        # 规则回退
        return self._compile_with_rules(intent, data_preview, domain)
    
    def _compile_with_llm(
        self,
        intent: str,
        data_preview: dict | None,
        domain: str,
    ) -> tuple[IntentCard, DataConstructionCard, TrainingPlanCard]:
        """使用LLM生成卡片"""
        
        system_prompt = """你是一个专业的 ML 意图编译器。请将用户的自然语言需求编译为三张"卡片"，用于指导AutoML训练。

你需要输出三张卡的内容：

1. **IntentCard（需求编译卡）**：系统理解的用户意图
   - task_type: 任务类型（二分类/多分类/回归/时序等）
   - must_keep: 必须保留的元素列表（如"高价值客户优先"）
   - can_change: 允许变化的元素列表
   - quality_speed_cost: 质量/速度/成本优先级（quality/speed/balanced/edge）
   - error_preference: 错误偏好（prefer_fp/prefer_fn/balanced）
   - success_criteria: 验收标准列表
   - unacceptable_errors: 不可接受的错误列表
   - system_reasoning: 系统为什么这样理解的解释
   - confidence_score: 置信度（0-1）
   - ambiguity_flags: 不确定的地方列表

2. **DataConstructionCard（数据构造卡）**：数据层面的用户介入点
   - allowed_variations: 允许的数据变化（如"允许光照变化"）
   - forbidden_variations: 禁止的数据变化（如"不允许主体位置漂移"）
   - augmentation_style: 增强风格（conservative/standard/aggressive）

3. **TrainingPlanCard（训练方案卡）**：系统选择的训练路线
   - approach: 训练方法（fine-tuning/distillation/paired-edit等）
   - approach_reasoning: 为什么选择这个方法
   - primary_goal: 主要优化目标
   - secondary_goals: 次要目标列表

请用中文给出reasoning，让用户理解系统的思考过程。"""

        user_prompt = f"""用户业务需求：{intent}

数据预览：{data_preview if data_preview else '未提供'}

领域：{domain}

请按以下JSON格式输出三张卡的内容：

{{
    "intent_card": {{
        "task_type": "任务类型",
        "must_keep": ["必须保留1", "必须保留2"],
        "can_change": ["可以变化1"],
        "quality_speed_cost": "balanced",
        "error_preference": "balanced",
        "success_criteria": ["标准1", "标准2"],
        "unacceptable_errors": ["错误1"],
        "system_reasoning": "系统理解过程...",
        "confidence_score": 0.85,
        "ambiguity_flags": ["不确定的地方"]
    }},
    "data_card": {{
        "allowed_variations": ["允许的变化"],
        "forbidden_variations": ["禁止的变化"],
        "augmentation_style": "standard"
    }},
    "plan_card": {{
        "approach": "训练方法",
        "approach_reasoning": "选择理由...",
        "primary_goal": "主要目标",
        "secondary_goals": ["次要目标1"]
    }}
}}"""

        response = self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        
        # 解析JSON响应
        import json
        json_str = self.llm_client._extract_json(response)
        data = json.loads(json_str)
        
        # 构建卡片
        intent_data = data.get("intent_card", {})
        data_data = data.get("data_card", {})
        plan_data = data.get("plan_card", {})
        
        intent_card = IntentCard(
            task_description=intent,
            task_type=intent_data.get("task_type", "unknown"),
            must_keep=intent_data.get("must_keep", []),
            can_change=intent_data.get("can_change", []),
            quality_speed_cost=QualitySpeedCost(intent_data.get("quality_speed_cost", "balanced")),
            error_preference=ErrorPreference(intent_data.get("error_preference", "balanced")),
            success_criteria=intent_data.get("success_criteria", []),
            unacceptable_errors=intent_data.get("unacceptable_errors", []),
            system_reasoning=intent_data.get("system_reasoning", ""),
            confidence_score=intent_data.get("confidence_score", 0.5),
            ambiguity_flags=intent_data.get("ambiguity_flags", []),
        )
        
        data_card = DataConstructionCard(
            total_samples=data_preview.get("total", 0) if data_preview else 0,
            train_samples=data_preview.get("train", 0) if data_preview else 0,
            val_samples=data_preview.get("val", 0) if data_preview else 0,
            allowed_variations=data_data.get("allowed_variations", []),
            forbidden_variations=data_data.get("forbidden_variations", []),
            augmentation_style=AugmentationStyle(data_data.get("augmentation_style", "standard")),
        )
        
        plan_card = TrainingPlanCard(
            approach=plan_data.get("approach", "standard"),
            approach_reasoning=plan_data.get("approach_reasoning", ""),
            primary_goal=plan_data.get("primary_goal", ""),
            secondary_goals=plan_data.get("secondary_goals", []),
            quality_speed_cost=intent_card.quality_speed_cost,
        )
        
        return intent_card, data_card, plan_card
    
    def _compile_with_rules(
        self,
        intent: str,
        data_preview: dict | None,
        domain: str,
    ) -> tuple[IntentCard, DataConstructionCard, TrainingPlanCard]:
        """使用规则回退生成卡片"""
        
        intent_lower = intent.lower()
        
        # 识别任务类型
        if any(kw in intent_lower for kw in ["时序", "时间", "未来", "明天", "趋势"]):
            task_type = "time_series"
        elif any(kw in intent_lower for kw in ["是否", "会不会", "分类", "判断", "识别", "流失", "欺诈"]):
            task_type = "binary_classification"
        elif any(kw in intent_lower for kw in ["数值", "预测", "价格", "销量", "多少"]):
            task_type = "regression"
        else:
            task_type = "binary_classification"
        
        # 识别错误偏好
        if any(kw in intent_lower for kw in ["漏报", "漏掉", "不能错过", "召回"]):
            error_pref = ErrorPreference.PREFER_FALSE_NEGATIVE
        elif any(kw in intent_lower for kw in ["误报", "错杀", "精确", "精准"]):
            error_pref = ErrorPreference.PREFER_FALSE_POSITIVE
        else:
            error_pref = ErrorPreference.BALANCED
        
        # 识别质量/速度/成本
        if any(kw in intent_lower for kw in ["快速", "实时", "速度", "低延迟"]):
            qsc = QualitySpeedCost.SPEED_FIRST
        elif any(kw in intent_lower for kw in ["边缘", "端侧", "mobile", "嵌入式"]):
            qsc = QualitySpeedCost.EDGE_DEPLOY
        elif any(kw in intent_lower for kw in ["质量", "精度", "准确", "最好"]):
            qsc = QualitySpeedCost.QUALITY_FIRST
        else:
            qsc = QualitySpeedCost.BALANCED
        
        # 执行模糊性检测
        ambiguity = self._detect_ambiguity_rules(intent, data_preview)
        
        intent_card = IntentCard(
            task_description=intent,
            task_type=task_type,
            must_keep=["业务逻辑正确性"],
            can_change=["模型类型", "训练时间"],
            quality_speed_cost=qsc,
            error_preference=error_pref,
            success_criteria=["验证集指标达到预期"],
            unacceptable_errors=["系统性偏差"],
            system_reasoning=f"基于规则分析：识别为{task_type}任务，采用默认配置。建议提供更多信息以获得更精准的编译结果。",
            confidence_score=0.5 if ambiguity.is_ambiguous else 0.7,
            ambiguity_flags=ambiguity.unclear_aspects if ambiguity.is_ambiguous else [],
            ambiguity_detection=ambiguity,
        )
        
        data_card = DataConstructionCard(
            total_samples=data_preview.get("total", 0) if data_preview else 0,
            augmentation_style=AugmentationStyle.STANDARD,
            allowed_variations=["数值范围变化"],
            forbidden_variations=["标签错误"],
        )
        
        plan_card = TrainingPlanCard(
            approach="standard_automl",
            approach_reasoning="使用标准AutoML流程进行超参数搜索和模型选择",
            primary_goal="优化主要评估指标",
            quality_speed_cost=qsc,
        )
        
        return intent_card, data_card, plan_card
    
    def refine_intent(
        self,
        current_card: IntentCard,
        user_feedback: dict,
    ) -> IntentCard:
        """
        根据用户反馈精化意图卡
        
        user_feedback example:
        {
            "corrected_task_description": "更精确的描述",
            "added_must_keep": ["必须保留的元素"],
            "changed_priority": "quality",
        }
        """
        # 应用用户修改
        if "corrected_task_description" in user_feedback:
            current_card.task_description = user_feedback["corrected_task_description"]
        
        if "added_must_keep" in user_feedback:
            current_card.must_keep.extend(user_feedback["added_must_keep"])
        
        if "changed_priority" in user_feedback:
            current_card.quality_speed_cost = QualitySpeedCost(
                user_feedback["changed_priority"]
            )
        
        # 重新编译（可选，如果需要LLM重新推理）
        # ...
        
        return current_card
    
    def refine_data_construction(
        self,
        current_card: DataConstructionCard,
        user_feedback: dict,
    ) -> DataConstructionCard:
        """
        根据用户反馈精化数据构造卡
        
        user_feedback example:
        {
            "excluded_sample_ids": ["id1", "id2"],
            "high_priority_sample_ids": ["id3"],
            "forbidden_variations": ["主体位置变化"],
        }
        """
        if "excluded_sample_ids" in user_feedback:
            current_card.excluded_samples.extend(user_feedback["excluded_sample_ids"])
        
        if "high_priority_sample_ids" in user_feedback:
            current_card.high_priority_samples.extend(user_feedback["high_priority_sample_ids"])
        
        if "forbidden_variations" in user_feedback:
            current_card.forbidden_variations = user_feedback["forbidden_variations"]
        
        return current_card
    
    def generate_summary(self, cards: tuple) -> str:
        """生成用户友好的总结文本"""
        intent_card, data_card, plan_card = cards
        
        summary = f"""
## 训练方案总结

### 任务理解
{intent_card.task_description}

系统将采用 **{plan_card.approach}** 路线，{plan_card.approach_reasoning}

### 关键约束
- 必须保留: {', '.join(intent_card.must_keep) if intent_card.must_keep else '无'}
- 优先目标: {intent_card.quality_speed_cost.value}
- 错误偏好: {intent_card.error_preference.value}

### 数据处理
- 总样本: {data_card.total_samples}
- 增强风格: {data_card.augmentation_style.value}
- 排除样本: {len(data_card.excluded_samples)}个

### 验收标准
{chr(10).join('- ' + c for c in intent_card.success_criteria) if intent_card.success_criteria else '使用默认标准'}

请确认以上理解是否正确，或进行修改。
"""
        return summary
    
    def _detect_ambiguity_rules(
        self,
        intent: str,
        data_preview: dict | None,
    ) -> AmbiguityDetection:
        """基于规则的模糊性检测"""
        
        ambiguity = AmbiguityDetection()
        missing = []
        unclear = []
        questions = []
        
        # 1. 检查描述长度
        if len(intent) < 20:
            missing.append("详细的任务描述")
            unclear.append("描述过短，难以准确理解任务")
            questions.append(ClarificationQuestion(
                question_id="q_desc_detail",
                question_text="请详细描述您的业务场景：这个模型要解决什么具体问题？",
                question_type="open",
                context="当前描述过于简短，系统难以准确理解任务类型和目标",
                priority=0,
            ))
        
        # 2. 检查目标变量是否明确
        target_indicators = ["预测", "判断", "识别", "分类", "检测", "估计", "计算"]
        has_target = any(ind in intent for ind in target_indicators)
        if not has_target:
            missing.append("明确的目标变量")
            unclear.append("未明确要预测或判断的目标")
            questions.append(ClarificationQuestion(
                question_id="q_target",
                question_text="您希望模型预测或判断什么？（例如：用户是否会购买、图片中的物体类别）",
                question_type="open",
                context="需要明确模型输出的目标变量",
                priority=0,
            ))
        
        # 3. 检查错误偏好
        error_indicators = ["误报", "漏报", "精确", "召回", "准确", "覆盖"]
        has_error_pref = any(ind in intent for ind in error_indicators)
        if not has_error_pref:
            missing.append("错误类型偏好")
            unclear.append("未说明哪种错误更不能接受")
            questions.append(ClarificationQuestion(
                question_id="q_error_pref",
                question_text="如果模型可能犯错，您更不能接受哪种情况？",
                question_type="multiple_choice",
                options=[
                    "宁可误报（把正常的判断为异常）",
                    "宁可漏报（把异常的判断为正常）",
                    "两者都要避免（平衡）"
                ],
                context="不同的错误偏好会影响模型优化目标和阈值选择",
                priority=1,
            ))
        
        # 4. 检查质量/速度偏好
        priority_indicators = ["质量", "精度", "速度", "实时", "快速", "准确", "边缘"]
        has_priority = any(ind in intent for ind in priority_indicators)
        if not has_priority:
            missing.append("质量/速度/成本优先级")
            unclear.append("未明确优化优先级")
            questions.append(ClarificationQuestion(
                question_id="q_priority",
                question_text="您更看重以下哪个方面？",
                question_type="multiple_choice",
                options=[
                    "预测精度越高越好（质量优先）",
                    "响应速度越快越好（速度优先）",
                    "希望在精度和速度间平衡"
                ],
                context="影响模型选择和训练策略",
                priority=2,
            ))
        
        # 5. 检查数据信息
        if not data_preview or data_preview.get("total", 0) == 0:
            missing.append("训练数据信息")
            unclear.append("未提供数据集")
            questions.append(ClarificationQuestion(
                question_id="q_data",
                question_text="您是否有准备好的训练数据？数据包含哪些字段？",
                question_type="open",
                context="需要了解数据情况来设计特征工程和数据处理策略",
                priority=1,
            ))
        
        # 计算模糊度分数
        ambiguity_score = min(1.0, (len(missing) * 0.2 + len(unclear) * 0.1))
        
        # 按优先级排序问题
        questions.sort(key=lambda q: q.priority)
        
        ambiguity.is_ambiguous = len(missing) > 0 or ambiguity_score > 0.3
        ambiguity.ambiguity_score = ambiguity_score
        ambiguity.missing_elements = missing
        ambiguity.unclear_aspects = unclear
        ambiguity.suggested_questions = questions
        ambiguity.can_proceed_with_caution = ambiguity_score < 0.6 and len(missing) <= 2
        
        return ambiguity
    
    def clarify_with_answers(
        self,
        current_card: IntentCard,
        answers: list[dict],
    ) -> IntentCard:
        """
        根据用户回答更新意图卡
        
        Args:
            answers: [{"question_id": "...", "answer": "..."}, ...]
        """
        # 将回答记录到澄清历史
        for ans in answers:
            q_id = ans.get("question_id")
            answer = ans.get("answer", "")
            
            # 更新对应的问题状态
            for q in current_card.ambiguity_detection.suggested_questions:
                if q.question_id == q_id:
                    q.is_answered = True
                    q.answer = answer
                    current_card.clarification_history.append(q)
                    break
            
            # 根据问题类型应用不同的更新逻辑
            if q_id == "q_target" and answer:
                # 用户补充了目标描述，更新任务描述
                current_card.task_description += f"\n补充：目标变量 - {answer}"
                
            elif q_id == "q_error_pref":
                # 更新错误偏好
                if "误报" in answer or "fp" in answer.lower():
                    current_card.error_preference = ErrorPreference.PREFER_FALSE_POSITIVE
                elif "漏报" in answer or "fn" in answer.lower():
                    current_card.error_preference = ErrorPreference.PREFER_FALSE_NEGATIVE
                else:
                    current_card.error_preference = ErrorPreference.BALANCED
                    
            elif q_id == "q_priority":
                # 更新优先级
                if "质量" in answer or "精度" in answer:
                    current_card.quality_speed_cost = QualitySpeedCost.QUALITY_FIRST
                elif "速度" in answer or "实时" in answer:
                    current_card.quality_speed_cost = QualitySpeedCost.SPEED_FIRST
                else:
                    current_card.quality_speed_cost = QualitySpeedCost.BALANCED
        
        # 重新评估模糊性
        # 如果回答了核心问题，降低模糊度
        answered_core = sum(1 for q in current_card.clarification_history 
                           if q.is_answered and q.priority <= 1)
        if answered_core >= 2:
            current_card.ambiguity_detection.is_ambiguous = False
            current_card.ambiguity_detection.ambiguity_score *= 0.5
            current_card.confidence_score = min(0.8, current_card.confidence_score + 0.2)
        
        return current_card