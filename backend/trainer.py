"""
训练引擎 - AutoML 真实训练实现

支持：
- 多种模型（XGBoost, LightGBM, RandomForest, LogisticRegression等）
- 超参数搜索（贝叶斯优化 + 随机搜索）
- 早停机制
- Checkpoint 保存/恢复
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

from backend.compiler import ObjectiveSpec, TaskFamily, ObjectiveMetric
from backend.data_manager import FeatureEngineer, ColumnType

# 尝试导入可选依赖
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """训练配置"""
    max_training_time: int = 300  # 秒
    max_trials: int = 50
    early_stopping_rounds: int = 10
    cv_folds: int = 3
    random_state: int = 42


@dataclass
class TrialResult:
    """单次试验结果"""
    trial_id: int
    model_name: str
    hyperparameters: dict[str, Any]
    metric_score: float
    train_time: float
    cv_scores: list[float] = field(default_factory=list)
    status: str = "completed"  # completed, failed, pruned


@dataclass
class TrainingResult:
    """训练结果"""
    job_id: str
    status: str  # running, completed, failed, stopped
    best_model_name: str | None = None
    best_metric_score: float | None = None
    best_hyperparameters: dict[str, Any] = field(default_factory=dict)
    trials: list[TrialResult] = field(default_factory=list)
    final_model_path: str | None = None
    preprocessor_path: str | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    training_duration: float = 0.0
    error_message: str | None = None


class ModelRegistry:
    """模型注册表 - 管理可用模型和其超参数空间"""
    
    MODELS = {
        # 分类模型
        "logistic_regression": {
            "type": "classifier",
            "class": LogisticRegression,
            "params": {
                "C": (0.001, 10.0, "log"),
                "max_iter": (100, 1000, "int"),
            },
            "fast": True,
        },
        "random_forest": {
            "type": "classifier",
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": (50, 200, "int"),
                "max_depth": (3, 20, "int"),
                "min_samples_split": (2, 20, "int"),
            },
            "fast": False,
        },
        "xgboost": {
            "type": "classifier",
            "class": None,  # 特殊处理
            "params": {
                "n_estimators": (50, 300, "int"),
                "max_depth": (3, 10, "int"),
                "learning_rate": (0.01, 0.3, "log"),
                "subsample": (0.6, 1.0, "uniform"),
            },
            "fast": False,
        },
        "lightgbm": {
            "type": "classifier",
            "class": None,  # 特殊处理
            "params": {
                "n_estimators": (50, 300, "int"),
                "max_depth": (3, 10, "int"),
                "learning_rate": (0.01, 0.3, "log"),
                "num_leaves": (15, 150, "int"),
            },
            "fast": False,
        },
        # 回归模型
        "elastic_net": {
            "type": "regressor",
            "class": ElasticNet,
            "params": {
                "alpha": (0.001, 1.0, "log"),
                "l1_ratio": (0.0, 1.0, "uniform"),
            },
            "fast": True,
        },
        "random_forest_regressor": {
            "type": "regressor",
            "class": RandomForestRegressor,
            "params": {
                "n_estimators": (50, 200, "int"),
                "max_depth": (3, 20, "int"),
            },
            "fast": False,
        },
    }
    
    @classmethod
    def get_model_class(cls, name: str, task_type: str):
        """获取模型类"""
        if name == "xgboost":
            if not XGBOOST_AVAILABLE:
                return None
            return xgb.XGBClassifier if task_type == "classifier" else xgb.XGBRegressor
        
        if name == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                return None
            return lgb.LGBMClassifier if task_type == "classifier" else lgb.LGBMRegressor
        
        entry = cls.MODELS.get(name)
        if entry:
            return entry["class"]
        return None
    
    @classmethod
    def get_param_space(cls, name: str) -> dict[str, tuple]:
        """获取超参数搜索空间"""
        entry = cls.MODELS.get(name, {})
        return entry.get("params", {})
    
    @classmethod
    def sample_params(cls, name: str, random_state: int | None = None) -> dict[str, Any]:
        """随机采样超参数"""
        rng = np.random.RandomState(random_state)
        space = cls.get_param_space(name)
        params = {}
        
        for param_name, (low, high, dtype) in space.items():
            if dtype == "int":
                params[param_name] = rng.randint(low, high + 1)
            elif dtype == "log":
                params[param_name] = np.exp(rng.uniform(np.log(low), np.log(high)))
            elif dtype == "uniform":
                params[param_name] = rng.uniform(low, high)
        
        return params


class AutoMLTrainer:
    """AutoML 训练器"""
    
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.registry = ModelRegistry()
        self.feature_engineer = FeatureEngineer()
    
    def train(
        self,
        job_id: str,
        df: pd.DataFrame,
        spec: ObjectiveSpec,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> TrainingResult:
        """
        执行 AutoML 训练
        
        Args:
            job_id: 任务ID
            df: 训练数据
            spec: 目标规范
            progress_callback: 进度回调函数
        
        Returns:
            TrainingResult: 训练结果
        """
        start_time = time.time()
        result = TrainingResult(job_id=job_id, status="running")
        
        try:
            # 1. 数据预处理
            self._update_progress(progress_callback, {"step": "preprocessing", "message": "数据预处理中..."})
            
            target_col = spec.label.target_column or "target"
            if target_col not in df.columns:
                # 尝试自动选择最后一列
                target_col = df.columns[-1]
            
            feature_cols = [c for c in df.columns if c != target_col]
            
            # 推断列类型
            column_types = self._infer_column_types(df, feature_cols)
            
            # 特征工程
            X, y = self.feature_engineer.fit_transform(df, feature_cols, target_col, column_types)
            
            # 划分训练/验证集
            stratify = y if spec.task_family == TaskFamily.BINARY_CLASSIFICATION else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=spec.test_size, random_state=spec.random_state,
                stratify=stratify
            )
            
            # 2. 确定任务类型和评估指标
            task_type = "classifier" if spec.task_family in [
                TaskFamily.BINARY_CLASSIFICATION, 
                TaskFamily.MULTICLASS_CLASSIFICATION
            ] else "regressor"
            
            metric_fn = self._get_metric_fn(spec.primary_metric, task_type)
            
            # 3. 超参数搜索
            self._update_progress(progress_callback, {"step": "search", "message": "超参数搜索中..."})
            
            best_score = float('-inf')
            best_model = None
            best_model_name = None
            best_params = {}
            
            # 获取候选模型列表
            candidate_models = spec.recommended_models or ["random_forest", "xgboost"]
            
            trial_id = 0
            for model_name in candidate_models:
                if model_name not in self.registry.MODELS:
                    continue
                
                model_class = self.registry.get_model_class(model_name, task_type)
                if model_class is None:
                    continue
                
                # 对每个模型进行多轮试验
                n_trials = min(self.config.max_trials // len(candidate_models), 10)
                
                for i in range(n_trials):
                    trial_id += 1
                    trial_start = time.time()
                    
                    try:
                        # 采样超参数
                        params = self.registry.sample_params(model_name, random_state=trial_id)
                        
                        # 创建模型
                        model = model_class(random_state=self.config.random_state, **params)
                        
                        # 交叉验证
                        cv_scores = self._cross_validate(model, X_train, y_train, task_type)
                        cv_mean = np.mean(cv_scores)
                        
                        # 在验证集上评估
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        val_score = metric_fn(y_val, y_pred)
                        
                        train_time = time.time() - trial_start
                        
                        trial = TrialResult(
                            trial_id=trial_id,
                            model_name=model_name,
                            hyperparameters=params,
                            metric_score=val_score,
                            train_time=train_time,
                            cv_scores=cv_scores.tolist(),
                        )
                        result.trials.append(trial)
                        
                        # 更新最佳模型
                        if val_score > best_score:
                            best_score = val_score
                            best_model = model
                            best_model_name = model_name
                            best_params = params
                        
                        # 检查时间限制
                        elapsed = time.time() - start_time
                        if elapsed > self.config.max_training_time:
                            self._update_progress(progress_callback, {
                                "step": "early_stop", 
                                "message": f"达到时间限制，已训练 {trial_id} 轮"
                            })
                            break
                        
                        # 报告进度
                        self._update_progress(progress_callback, {
                            "step": "search",
                            "trial": trial_id,
                            "best_score": best_score,
                            "current_model": model_name,
                        })
                        
                    except Exception as e:
                        # 记录失败的试验
                        result.trials.append(TrialResult(
                            trial_id=trial_id,
                            model_name=model_name,
                            hyperparameters={},
                            metric_score=0.0,
                            train_time=0.0,
                            status="failed",
                        ))
                
                # 检查时间限制
                if time.time() - start_time > self.config.max_training_time:
                    break
            
            # 4. 训练最终模型
            self._update_progress(progress_callback, {"step": "final_training", "message": "训练最终模型..."})
            
            if best_model is not None:
                # 在全量数据上重新训练
                final_model_class = self.registry.get_model_class(best_model_name, task_type)
                final_model = final_model_class(random_state=self.config.random_state, **best_params)
                final_model.fit(X, y)
                
                # 计算最终指标
                y_train_pred = final_model.predict(X_train)
                y_val_pred = final_model.predict(X_val)
                
                result.train_metrics = self._compute_metrics(y_train, y_train_pred, task_type)
                result.val_metrics = self._compute_metrics(y_val, y_val_pred, task_type)
                
                # 提取特征重要性
                result.feature_importance = self._extract_feature_importance(final_model)
                
                # 保存模型
                model_path = CHECKPOINT_DIR / f"{job_id}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(final_model, f)
                result.final_model_path = str(model_path)
                
                # 保存预处理器
                preprocessor_path = CHECKPOINT_DIR / f"{job_id}_preprocessor.pkl"
                self.feature_engineer.save(preprocessor_path)
                result.preprocessor_path = str(preprocessor_path)
                
                # 更新结果
                result.best_model_name = best_model_name
                result.best_metric_score = best_score
                result.best_hyperparameters = best_params
                result.status = "completed"
            else:
                result.status = "failed"
                result.error_message = "没有找到合适的模型"
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
        
        finally:
            result.training_duration = time.time() - start_time
        
        return result
    
    def _infer_column_types(self, df: pd.DataFrame, feature_cols: list[str]) -> dict[str, ColumnType]:
        """推断列类型"""
        from backend.data_manager import DataTypeDetector
        detector = DataTypeDetector()
        
        column_types = {}
        for col in feature_cols:
            col_type = detector.detect_column_type(df[col])
            column_types[col] = col_type
        
        return column_types
    
    def _get_metric_fn(self, metric: ObjectiveMetric, task_type: str) -> Callable:
        """获取评估指标函数"""
        if task_type == "classifier":
            metric_map = {
                ObjectiveMetric.ACCURACY: accuracy_score,
                ObjectiveMetric.PRECISION: lambda y, p: precision_score(y, p, average='binary', zero_division=0),
                ObjectiveMetric.RECALL: lambda y, p: recall_score(y, p, average='binary', zero_division=0),
                ObjectiveMetric.F1: lambda y, p: f1_score(y, p, average='binary', zero_division=0),
                ObjectiveMetric.AUC: lambda y, p: roc_auc_score(y, p) if len(set(y)) == 2 else accuracy_score(y, p),
            }
            return metric_map.get(metric, f1_score)
        else:
            # 回归指标，取负值（因为我们用最大化）
            metric_map = {
                ObjectiveMetric.MAE: lambda y, p: -mean_absolute_error(y, p),
                ObjectiveMetric.RMSE: lambda y, p: -np.sqrt(mean_squared_error(y, p)),
                ObjectiveMetric.R2: r2_score,
            }
            return metric_map.get(metric, lambda y, p: -mean_squared_error(y, p))
    
    def _cross_validate(self, model, X, y, task_type: str) -> np.ndarray:
        """交叉验证"""
        if task_type == "classifier":
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring = 'f1'
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            scoring = 'neg_mean_squared_error'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return scores
    
    def _compute_metrics(self, y_true, y_pred, task_type: str) -> dict[str, float]:
        """计算评估指标"""
        if task_type == "classifier":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
            }
        else:
            return {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
            }
    
    def _extract_feature_importance(self, model) -> dict[str, float]:
        """提取特征重要性"""
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            # 树模型
            feature_names = self.feature_engineer.get_feature_names()
            importances = model.feature_importances_
            
            # 如果特征名数量和重要性不匹配，使用索引
            if len(feature_names) != len(importances):
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            for name, imp in zip(feature_names, importances):
                importance[name] = float(imp)
        
        elif hasattr(model, 'coef_'):
            # 线性模型
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:
                coefs = coefs.mean(axis=0)
            
            feature_names = self.feature_engineer.get_feature_names()
            if len(feature_names) != len(coefs):
                feature_names = [f"feature_{i}" for i in range(len(coefs))]
            
            for name, coef in zip(feature_names, coefs):
                importance[name] = float(coef)
        
        # 排序并返回前10
        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        return sorted_imp
    
    def _update_progress(self, callback: Callable | None, progress: dict):
        """更新进度"""
        if callback:
            try:
                callback(progress)
            except:
                pass


class ModelLoader:
    """模型加载器 - 用于推理"""
    
    def __init__(self, model_path: str, preprocessor_path: str | None = None):
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        
        # 加载模型
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # 加载预处理器
        if preprocessor_path:
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.load(preprocessor_path)
    
    def predict(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
        """预测"""
        if self.feature_engineer is not None:
            X = self.feature_engineer.transform(df, feature_cols or df.columns.tolist())
        else:
            X = df.values
        
        return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
        """预测概率"""
        if self.feature_engineer is not None:
            X = self.feature_engineer.transform(df, feature_cols or df.columns.tolist())
        else:
            X = df.values
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("模型不支持概率预测")
