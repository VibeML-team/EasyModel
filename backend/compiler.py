"""
Objective Compiler - 将自然语言需求编译为机器可执行的优化目标

实现 chat2objective 层：
- 使用LLM进行意图解析和任务识别
- 编译为结构化 ObjectiveSpec
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class TaskFamily(str, Enum):
    """任务类型族"""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    RANKING = "ranking"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"


class ObjectiveMetric(str, Enum):
    """评估指标"""
    # 分类
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    PRECISION_AT_K = "precision_at_k"
    # 回归
    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    R2 = "r2"
    # 业务指标
    EXPECTED_PROFIT = "expected_profit"
    CUSTOM_UTILITY = "custom_utility"


class ConstraintType(str, Enum):
    """约束类型"""
    LATENCY_MS = "latency_ms"
    MEMORY_MB = "memory_mb"
    DAILY_BUDGET = "daily_budget"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    MIN_PRECISION = "min_precision"
    MIN_RECALL = "min_recall"
    ABSTAIN_ALLOWED = "abstain_allowed"


@dataclass
class LabelDefinition:
    """标签定义"""
    target_column: str | None = None
    task_family: TaskFamily = TaskFamily.BINARY_CLASSIFICATION
    horizon_days: int | None = None  # 时序预测用
    positive_class: str | None = None  # 二分类正类
    event_definition: str | None = None  # 事件定义（如"流失"）


@dataclass
class ObjectiveSpec:
    """
    优化目标规范 - 系统的核心中间表示
    
    对应评估文档中的 "ObjectiveSpec schema"
    """
    # 原始需求
    raw_intent: str = ""
    
    # 任务定义
    task_family: TaskFamily = TaskFamily.BINARY_CLASSIFICATION
    label: LabelDefinition = field(default_factory=LabelDefinition)
    
    # 优化目标
    primary_metric: ObjectiveMetric = ObjectiveMetric.F1
    surrogate_loss: str = "cross_entropy"
    
    # 约束条件
    constraints: dict[str, Any] = field(default_factory=dict)
    
    # 评估协议
    validation_strategy: Literal["train_test_split", "time_series_split", "group_split"] = "train_test_split"
    test_size: float = 0.2
    random_state: int = 42
    
    # 训练配置
    max_training_time: int = 300  # 秒
    max_trials: int = 50  # 超参搜索次数
    
    # 交付配置
    delivery_mode: Literal["batch", "online", "edge"] = "batch"
    output_format: Literal["score", "class", "top_k", "full_pipeline"] = "full_pipeline"
    
    # 推荐模型族
    recommended_models: list[str] = field(default_factory=list)
    
    # LLM 分析结果
    interpretation: str = ""  # 推理过程说明
    feature_engineering_hints: list[str] = field(default_factory=list)
    potential_issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_intent": self.raw_intent,
            "task_family": self.task_family.value,
            "label": {
                "target_column": self.label.target_column,
                "task_family": self.label.task_family.value,
                "horizon_days": self.label.horizon_days,
                "positive_class": self.label.positive_class,
                "event_definition": self.label.event_definition,
            },
            "primary_metric": self.primary_metric.value,
            "surrogate_loss": self.surrogate_loss,
            "constraints": self.constraints,
            "validation_strategy": self.validation_strategy,
            "test_size": self.test_size,
            "delivery_mode": self.delivery_mode,
            "output_format": self.output_format,
            "recommended_models": self.recommended_models,
            "interpretation": self.interpretation,
            "feature_engineering_hints": self.feature_engineering_hints,
            "potential_issues": self.potential_issues,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveSpec:
        label_data = data.get("label", {})
        return cls(
            raw_intent=data.get("raw_intent", ""),
            task_family=TaskFamily(data.get("task_family", "binary_classification")),
            label=LabelDefinition(
                target_column=label_data.get("target_column"),
                task_family=TaskFamily(label_data.get("task_family", "binary_classification")),
                horizon_days=label_data.get("horizon_days"),
                positive_class=label_data.get("positive_class"),
                event_definition=label_data.get("event_definition"),
            ),
            primary_metric=ObjectiveMetric(data.get("primary_metric", "f1")),
            surrogate_loss=data.get("surrogate_loss", "cross_entropy"),
            constraints=data.get("constraints", {}),
            validation_strategy=data.get("validation_strategy", "train_test_split"),
            test_size=data.get("test_size", 0.2),
            delivery_mode=data.get("delivery_mode", "batch"),
            output_format=data.get("output_format", "full_pipeline"),
            recommended_models=data.get("recommended_models", []),
            interpretation=data.get("interpretation", ""),
            feature_engineering_hints=data.get("feature_engineering_hints", []),
            potential_issues=data.get("potential_issues", []),
        )


def _use_llm_compiler() -> bool:
    """检查是否使用LLM编译器"""
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    return api_key is not None and api_key.strip() != ""


def detect_ambiguity(req) -> tuple[bool, list[str], list[str]]:
    """
    检测需求模糊性
    
    如果有LLM配置，使用LLM分析；否则使用规则
    """
    if _use_llm_compiler():
        try:
            from backend.llm_client import get_llm_client
            client = get_llm_client()
            return client.clarify_ambiguity(
                user_goal=req.user_goal,
                must_keep=req.must_keep,
                worst_errors=req.worst_errors,
            )
        except Exception as e:
            # LLM 失败时回退到规则
            pass
    
    # 规则回退
    reasons = []
    follow_ups = []
    
    if not req.must_keep:
        reasons.append("缺少必须保留项")
        follow_ups.append("哪些元素绝对不能被改变？请至少给 1-3 条。")
    
    if not req.worst_errors:
        reasons.append("缺少不可接受错误定义")
        follow_ups.append("最不能接受的错误是什么（例如漏报、误报、格式错误）？")
    
    if len(req.user_goal) < 15:
        reasons.append("目标描述过短")
        follow_ups.append("请补充一句：谁在什么场景使用、怎样才算成功。")
    
    if not req.dataset_id:
        reasons.append("未上传数据集")
        follow_ups.append("请上传包含训练数据的 CSV 文件。")
    
    return len(reasons) > 0, reasons, follow_ups


def compile_objective(
    user_goal: str,
    must_keep: list[str],
    can_change: list[str],
    worst_errors: list[str],
    priority: str = "quality",
    data_schema: dict | None = None,
) -> ObjectiveSpec:
    """
    主编译函数 - 将用户自然语言需求编译为 ObjectiveSpec
    
    这是 chat2objective 的核心实现
    优先使用LLM，失败时回退到规则
    """
    
    # 尝试使用LLM编译
    if _use_llm_compiler():
        try:
            from backend.llm_client import get_llm_client
            client = get_llm_client()
            
            result = client.compile_objective(
                user_goal=user_goal,
                must_keep=must_keep,
                can_change=can_change,
                worst_errors=worst_errors,
                priority=priority,
                data_schema=data_schema,
            )
            
            # 转换为目标格式
            spec = ObjectiveSpec(
                raw_intent=user_goal,
                task_family=TaskFamily(result.task_family),
                label=LabelDefinition(
                    target_column=result.target_column,
                    task_family=TaskFamily(result.task_family),
                    event_definition=result.target_description,
                ),
                primary_metric=ObjectiveMetric(result.primary_metric),
                surrogate_loss=result.surrogate_loss,
                constraints=result.constraints,
                validation_strategy=result.validation_strategy,
                recommended_models=result.recommended_models,
                interpretation=result.reasoning if hasattr(result, 'reasoning') else "",
                feature_engineering_hints=result.feature_engineering_hints,
                potential_issues=result.potential_issues,
            )
            
            # 应用优先级调整
            if priority == "latency":
                spec.constraints[ConstraintType.LATENCY_MS.value] = 50
            elif priority == "cost":
                spec.max_training_time = 120
                spec.max_trials = 20
            
            return spec
            
        except Exception as e:
            # LLM 失败时回退到规则编译
            print(f"LLM编译失败，回退到规则: {e}")
    
    # 规则编译回退
    return _rule_based_compile(
        user_goal, must_keep, can_change, worst_errors, priority
    )


def _rule_based_compile(
    user_goal: str,
    must_keep: list[str],
    can_change: list[str],
    worst_errors: list[str],
    priority: str,
) -> ObjectiveSpec:
    """基于规则的编译（LLM失败时的回退）"""
    import re
    
    spec = ObjectiveSpec(raw_intent=user_goal)
    
    # 1. 识别任务类型
    goal_lower = user_goal.lower()
    
    if any(kw in goal_lower for kw in ["时序", "时间", "未来", "明天", "趋势", "走势"]):
        spec.task_family = TaskFamily.TIME_SERIES
    elif any(kw in goal_lower for kw in ["排序", "排名", "优先", "top", "推荐"]):
        spec.task_family = TaskFamily.RANKING
    elif any(kw in goal_lower for kw in ["是否", "会不会", "分类", "判断", "识别", "流失", "欺诈"]):
        spec.task_family = TaskFamily.BINARY_CLASSIFICATION
    elif any(kw in goal_lower for kw in ["数值", "预测", "价格", "销量", "多少", "回归"]):
        spec.task_family = TaskFamily.REGRESSION
    else:
        spec.task_family = TaskFamily.BINARY_CLASSIFICATION  # 默认
    
    spec.label.task_family = spec.task_family
    
    # 2. 推断目标列
    for keyword, col_name in [
        ("流失", "churn"),
        ("购买", "purchase"),
        ("点击", "click"),
        ("欺诈", "fraud"),
        ("违约", "default"),
    ]:
        if keyword in goal_lower:
            spec.label.target_column = col_name
            spec.label.event_definition = keyword
            break
    else:
        spec.label.target_column = "target"
    
    # 3. 选择评估指标
    if spec.task_family == TaskFamily.BINARY_CLASSIFICATION:
        if any("漏" in e or "recall" in e.lower() for e in worst_errors):
            spec.primary_metric = ObjectiveMetric.RECALL
        elif any("误" in e or "precision" in e.lower() for e in worst_errors):
            spec.primary_metric = ObjectiveMetric.PRECISION
        else:
            spec.primary_metric = ObjectiveMetric.F1
    elif spec.task_family == TaskFamily.REGRESSION:
        spec.primary_metric = ObjectiveMetric.RMSE
    elif spec.task_family == TaskFamily.TIME_SERIES:
        spec.primary_metric = ObjectiveMetric.MAPE
    elif spec.task_family == TaskFamily.RANKING:
        spec.primary_metric = ObjectiveMetric.PRECISION_AT_K
    
    # 4. 推荐模型
    if priority == "latency":
        spec.recommended_models = ["logistic_regression", "decision_tree"]
        spec.constraints["latency_ms"] = 50
    else:
        spec.recommended_models = ["xgboost", "lightgbm", "random_forest"]
    
    # 5. 验证策略
    if spec.task_family == TaskFamily.TIME_SERIES:
        spec.validation_strategy = "time_series_split"
    
    # 6. 应用优先级
    if priority == "cost":
        spec.max_training_time = 120
        spec.max_trials = 20
    
    spec.interpretation = "使用基于规则的回退编译（LLM未配置或不可用）"
    
    return spec
