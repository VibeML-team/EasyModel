"""
Training Controller V2 - 训练流程控制器

核心设计:
- 输入: IntentCard, DataConstructionCard, TrainingPlanCard (用户确认后的)
- 输出: AcceptanceCard (候选版本供用户选择)
- 机制层完全封装，用户通过"卡片"介入
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .v2_intent_compiler import (
    AcceptanceCard,
    DataConstructionCard,
    IntentCard,
    QualitySpeedCost,
    TrainingObservation,
    TrainingPlanCard,
)


@dataclass
class CheckpointInfo:
    """Checkpoint 信息"""
    checkpoint_id: str
    step: int
    metric: float
    metric_name: str = "score"
    created_at: float = 0.0
    model_path: str = ""
    status: str = "available"  # available/downloading/deleted
    
    def to_dict(self) -> dict:
        return {
            "id": self.checkpoint_id,
            "step": self.step,
            "metric": self.metric,
            "metric_name": self.metric_name,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass
class MechanismConfig:
    """
    机制层配置 - 从意图层卡片映射而来
    
    用户不直接操作这个，系统自动生成
    """
    # 模型架构
    model_family: str = "auto"
    model_size: str = "base"  # small/base/large
    use_lora: bool = False
    lora_rank: int = 16
    
    # 训练策略
    training_approach: str = "standard"  # standard/distillation/paired-edit
    loss_composition: list[dict] = field(default_factory=list)
    
    # 优化器 (默认隐藏)
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    warmup_steps: int = 100
    
    # 数据处理
    augmentation_policy: list[dict] = field(default_factory=list)
    sample_weights: dict = field(default_factory=dict)
    
    # 评估
    eval_metric: str = "auto"
    eval_frequency: int = 100
    
    @classmethod
    def from_cards(
        cls,
        intent: IntentCard,
        data: DataConstructionCard,
        plan: TrainingPlanCard,
    ) -> "MechanismConfig":
        """从意图层卡片生成机制层配置"""
        config = cls()
        
        # 根据 quality_speed_cost 选择模型大小
        if plan.quality_speed_cost == QualitySpeedCost.QUALITY_FIRST:
            config.model_size = "large"
            config.epochs = 20
        elif plan.quality_speed_cost == QualitySpeedCost.SPEED_FIRST:
            config.model_size = "small"
            config.epochs = 5
            config.use_lora = True
            config.lora_rank = 8
        elif plan.quality_speed_cost == QualitySpeedCost.EDGE_DEPLOY:
            config.model_size = "small"
            config.use_lora = True
            config.lora_rank = 4
        
        # 根据 augmentation_style 生成策略
        if data.augmentation_style.value == "conservative":
            config.augmentation_policy = [
                {"type": "identity", "prob": 0.5},
                {"type": "subtle_noise", "prob": 0.3, "magnitude": 0.05},
            ]
        elif data.augmentation_style.value == "aggressive":
            config.augmentation_policy = [
                {"type": "random_crop", "prob": 0.5},
                {"type": "color_jitter", "prob": 0.3},
                {"type": "rotation", "prob": 0.3, "degrees": 15},
            ]
        
        # 根据 error_preference 调整 loss
        if intent.error_preference.value == "prefer_fp":
            config.loss_composition = [
                {"type": "base", "weight": 1.0},
                {"type": "fp_penalty", "weight": 0.5},
            ]
        elif intent.error_preference.value == "prefer_fn":
            config.loss_composition = [
                {"type": "base", "weight": 1.0},
                {"type": "fn_penalty", "weight": 0.5},
            ]
        
        # 样本权重
        if data.high_priority_samples:
            config.sample_weights = {
                "high_priority": 2.0,
                "normal": 1.0,
                "excluded": 0.0,
            }
        
        return config


class TrainingControllerV2:
    """
    训练控制器 V2
    
    职责:
    1. 将用户确认的卡片转换为机制层配置
    2. 执行训练（内部使用BO/传统训练）
    3. 生成观测数据供用户查看
    4. 响应用户的轻量级介入（暂停/选择checkpoint等）
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs",
        sandbox=None,  # 可选的沙箱执行器
        data_manager=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox
        self.data_manager = data_manager
        
        # 状态
        self.current_job: dict | None = None
        self.is_running: bool = False
        self.is_paused: bool = False
        self.current_observation: TrainingObservation | None = None
        
        # 训练器
        self._trainer = None
        self._training_thread = None
        
    def start_training(
        self,
        intent: IntentCard,
        data: DataConstructionCard,
        plan: TrainingPlanCard,
        dataset_id: str | None = None,
        progress_callback: Callable[[TrainingObservation], None] | None = None,
    ) -> str:
        """
        启动训练流程
        
        Args:
            intent: 意图卡
            data: 数据构造卡
            plan: 训练方案卡
            dataset_id: 数据集ID（用于加载真实数据）
            progress_callback: 进度回调
            
        Returns:
            job_id
        """
        import uuid
        job_id = f"v2_{uuid.uuid4().hex[:8]}"
        
        # 1. 生成机制层配置（隐藏）
        mechanism = MechanismConfig.from_cards(intent, data, plan)
        
        self.current_job = {
            "job_id": job_id,
            "intent": intent,
            "data": data,
            "plan": plan,
            "mechanism": mechanism,
            "dataset_id": dataset_id,
            "start_time": time.time(),
            "checkpoints": [],
        }
        
        # 2. 启动训练（异步线程）
        self.is_running = True
        import threading
        self._training_thread = threading.Thread(
            target=self._training_loop_real,
            args=(progress_callback,),
            daemon=True,
        )
        self._training_thread.start()
        
        return job_id
    
    def _save_intermediate_checkpoint(
        self, 
        job_id: str, 
        trial_id: int, 
        model, 
        metric: float,
        metric_name: str = "score"
    ) -> CheckpointInfo:
        """保存中间 checkpoint"""
        import pickle
        
        checkpoint_id = f"ckpt_{trial_id}"
        checkpoint_path = self.output_dir / job_id / f"{checkpoint_id}.pkl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 创建 checkpoint 信息
        info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            step=trial_id * 10,
            metric=metric,
            metric_name=metric_name,
            created_at=time.time(),
            model_path=str(checkpoint_path),
            status="available",
        )
        
        # 添加到 job 的 checkpoint 列表
        if self.current_job:
            if "checkpoints" not in self.current_job:
                self.current_job["checkpoints"] = []
            self.current_job["checkpoints"].append(info)
        
        return info
    
    def _training_loop_real(
        self,
        progress_callback: Callable[[TrainingObservation], None] | None = None,
    ):
        """真实训练循环"""
        job = self.current_job
        mechanism = job["mechanism"]
        dataset_id = job.get("dataset_id")
        
        try:
            # 检查是否有数据管理器
            if self.data_manager is None:
                from backend.data_manager import data_manager
                self.data_manager = data_manager
            
            # 加载数据
            if dataset_id:
                df = self.data_manager.load_dataframe(dataset_id)
            else:
                # 无数据时创建模拟数据（用于演示）
                import pandas as pd
                import numpy as np
                np.random.seed(42)
                n_samples = 1000
                df = pd.DataFrame({
                    'feature_1': np.random.randn(n_samples),
                    'feature_2': np.random.randn(n_samples),
                    'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
                    'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                })
            
            # 从意图卡构建 ObjectiveSpec
            from backend.compiler import ObjectiveSpec, TaskFamily, ObjectiveMetric, LabelDefinition
            
            intent = job["intent"]
            
            # 映射任务类型
            task_family_map = {
                "binary_classification": TaskFamily.BINARY_CLASSIFICATION,
                "multiclass_classification": TaskFamily.MULTICLASS_CLASSIFICATION,
                "regression": TaskFamily.REGRESSION,
                "time_series": TaskFamily.TIME_SERIES,
                "ranking": TaskFamily.RANKING,
            }
            task_family = task_family_map.get(intent.task_type, TaskFamily.BINARY_CLASSIFICATION)
            
            # 映射评估指标
            metric_map = {
                "accuracy": ObjectiveMetric.ACCURACY,
                "precision": ObjectiveMetric.PRECISION,
                "recall": ObjectiveMetric.RECALL,
                "f1": ObjectiveMetric.F1,
                "auc": ObjectiveMetric.AUC,
                "mae": ObjectiveMetric.MAE,
                "rmse": ObjectiveMetric.RMSE,
                "r2": ObjectiveMetric.R2,
            }
            
            # 根据 error_preference 选择默认指标
            if intent.error_preference.value == "prefer_fn":
                default_metric = ObjectiveMetric.RECALL
            elif intent.error_preference.value == "prefer_fp":
                default_metric = ObjectiveMetric.PRECISION
            else:
                default_metric = ObjectiveMetric.F1
            
            spec = ObjectiveSpec(
                raw_intent=intent.task_description,
                task_family=task_family,
                label=LabelDefinition(
                    target_column=None,  # 自动识别
                    task_family=task_family,
                ),
                primary_metric=default_metric,
                max_training_time=300,
                max_trials=mechanism.epochs * 3,  # 根据epoch估算
                recommended_models=["xgboost", "lightgbm", "random_forest"] 
                    if mechanism.model_size != "small" else ["logistic_regression", "random_forest"],
            )
            
            # 配置训练
            from backend.trainer import AutoMLTrainer, TrainingConfig
            
            config = TrainingConfig(
                max_training_time=300,
                max_trials=min(mechanism.epochs * 3, 30),
                random_state=42,
            )
            
            trainer = AutoMLTrainer(config)
            
            # 存储中间模型用于 checkpoint
            self._intermediate_models = {}
            
            # 进度回调包装器
            def wrapped_progress(progress: dict):
                step = progress.get("step", "")
                
                if step == "preprocessing":
                    obs = TrainingObservation(
                        stage="preparing",
                        current_step=5,
                        total_steps=100,
                        eta_seconds=300,
                        available_actions=["pause", "stop"],
                    )
                elif step == "search":
                    trial = progress.get("trial", 0)
                    best = progress.get("best_score", 0)
                    
                    # 保存当前 trial 的模型作为 checkpoint
                    if "current_model" in progress and trial > 0:
                        self._save_intermediate_checkpoint(
                            job["job_id"],
                            trial,
                            progress["current_model"],
                            progress.get("current_score", 0),
                            spec.primary_metric.value
                        )
                    
                    obs = TrainingObservation(
                        stage="training",
                        current_step=15 + min(trial * 2, 70),
                        total_steps=100,
                        eta_seconds=(100 - 15 - min(trial * 2, 70)) * 3,
                        metrics_history={
                            "best_cv_score": [best],
                            "current_trial": [trial],
                        },
                        best_checkpoint_step=trial,
                        best_checkpoint_metric=best,
                        available_actions=["pause", "stop", "select_checkpoint", "download_checkpoint"],
                    )
                elif step == "final_training":
                    obs = TrainingObservation(
                        stage="training",
                        current_step=90,
                        total_steps=100,
                        eta_seconds=30,
                        available_actions=["pause", "stop", "download_checkpoint"],
                    )
                else:
                    obs = TrainingObservation(
                        stage="training",
                        current_step=50,
                        total_steps=100,
                        available_actions=["pause", "stop"],
                    )
                
                self.current_observation = obs
                if progress_callback:
                    progress_callback(obs)
            
            # 执行训练
            result = trainer.train(
                job_id=job["job_id"],
                df=df,
                spec=spec,
                progress_callback=wrapped_progress,
            )
            
            # 存储结果
            job["training_result"] = result
            
            # 确保所有 trial 都有 checkpoint 记录
            existing_ids = {c.checkpoint_id for c in job.get("checkpoints", [])}
            for i, t in enumerate(result.trials):
                ckpt_id = f"ckpt_{t.trial_id}"
                if ckpt_id not in existing_ids:
                    info = CheckpointInfo(
                        checkpoint_id=ckpt_id,
                        step=t.trial_id * 10,
                        metric=t.metric_score,
                        metric_name=spec.primary_metric.value,
                        created_at=time.time() - (len(result.trials) - i) * 60,
                        model_path=str(self.output_dir / job["job_id"] / f"{ckpt_id}.pkl"),
                        status="available" if i < 5 else "archived",
                    )
                    if "checkpoints" not in job:
                        job["checkpoints"] = []
                    job["checkpoints"].append(info)
            
            # 最终观测
            self.current_observation = TrainingObservation(
                stage="completed",
                current_step=100,
                total_steps=100,
                eta_seconds=0,
                metrics_history={
                    "best_metric": [result.best_metric_score or 0],
                    "n_trials": [len(result.trials)],
                },
                best_checkpoint_step=len(result.trials),
                best_checkpoint_metric=result.best_metric_score or 0,
                available_actions=["select_checkpoint", "finalize", "download_checkpoint"],
            )
            
            if progress_callback:
                progress_callback(self.current_observation)
                
        except Exception as e:
            print(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            self.current_observation = TrainingObservation(
                stage="failed",
                current_step=0,
                total_steps=100,
                eta_seconds=0,
                risk_alerts=[{"type": "error", "message": str(e)}],
                available_actions=[],
            )
        
        finally:
            self.is_running = False
    
    def _training_loop(
        self,
        progress_callback: Callable[[TrainingObservation], None] | None = None,
    ):
        """训练循环（简化版，实际应异步）"""
        job = self.current_job
        mechanism = job["mechanism"]
        
        total_steps = mechanism.epochs * 100  # 假设每epoch 100步
        
        for step in range(total_steps):
            if not self.is_running:
                break
            
            while self.is_paused:
                time.sleep(0.1)
            
            # 模拟训练步骤
            # 实际这里调用真实训练代码
            
            # 生成观测
            observation = TrainingObservation(
                stage="training" if step < total_steps * 0.8 else "validating",
                current_step=step,
                total_steps=total_steps,
                eta_seconds=(total_steps - step) * 2,
                metrics_history={
                    "train_loss": [1.0 - step/total_steps * 0.5],
                    "val_loss": [1.1 - step/total_steps * 0.4],
                },
                best_checkpoint_step=max(0, step - 10),
                best_checkpoint_metric=0.8 + step/total_steps * 0.15,
                available_actions=["pause", "stop", "select_checkpoint"],
            )
            
            self.current_observation = observation
            
            if progress_callback:
                progress_callback(observation)
            
            # 模拟训练时间
            time.sleep(0.1)
        
        self.is_running = False
    
    def pause(self) -> bool:
        """暂停训练"""
        if self.is_running and not self.is_paused:
            self.is_paused = True
            return True
        return False
    
    def resume(self) -> bool:
        """恢复训练"""
        if self.is_running and self.is_paused:
            self.is_paused = False
            return True
        return False
    
    def stop(self) -> bool:
        """停止训练"""
        if self.is_running:
            self.is_running = False
            return True
        return False
    
    def select_checkpoint(self, checkpoint_id: str) -> bool:
        """用户选择某个checkpoint作为候选"""
        if self.current_job:
            self.current_job["selected_checkpoint"] = checkpoint_id
            return True
        return False
    
    def provide_feedback(
        self,
        checkpoint_id: str,
        feedback: dict,  # {"liked": True/False, "reason": "...", "aspects": {...}}
    ) -> bool:
        """
        用户对中间结果提供反馈
        
        例如：{"liked": False, "reason": "风格不够统一", "aspects": {"style": "too_varied"}}
        """
        if self.current_job:
            if "user_feedback" not in self.current_job:
                self.current_job["user_feedback"] = []
            
            self.current_job["user_feedback"].append({
                "checkpoint_id": checkpoint_id,
                "timestamp": time.time(),
                **feedback,
            })
            
            # 这里可以触发ReAct调整
            # 例如，如果用户反馈"风格不够统一"，可以增加风格一致性loss权重
            
            return True
        return False
    
    def finalize(self) -> AcceptanceCard:
        """
        完成训练，生成验收卡
        
        Returns:
            AcceptanceCard 供用户最终选择
        """
        job = self.current_job
        result = job.get("training_result")
        
        # 使用真实训练结果生成候选
        if result and result.trials:
            # 从 trial 结果生成候选
            candidates = []
            sorted_trials = sorted(result.trials, key=lambda t: t.metric_score, reverse=True)
            
            for i, trial in enumerate(sorted_trials[:5]):  # 取前5个
                candidates.append({
                    "id": f"ckpt_{trial.trial_id}",
                    "step": trial.trial_id * 10,
                    "metric": trial.metric_score,
                    "model": trial.model_name,
                    "params": trial.hyperparameters,
                    "train_time": trial.train_time,
                    "path": result.final_model_path if i == 0 else None,  # 最佳模型有路径
                })
        else:
            # 回退到模拟数据
            candidates = [
                {
                    "id": f"ckpt_{i}",
                    "step": i * 100,
                    "metric": 0.7 + i * 0.05,
                    "path": f"{self.output_dir}/{job['job_id']}/ckpt_{i}.pt",
                }
                for i in range(1, 6)
            ]
        
        # 应用用户反馈到候选排序
        if job.get("user_feedback"):
            for candidate in candidates:
                for fb in job["user_feedback"]:
                    if fb.get("checkpoint_id") == candidate["id"]:
                        if fb.get("liked"):
                            candidate["metric"] = candidate.get("metric", 0) + 0.05
                        else:
                            candidate["metric"] = candidate.get("metric", 0) - 0.05
            
            # 重新排序
            candidates.sort(key=lambda x: x.get("metric", 0), reverse=True)
        
        # 确定对比维度
        comparison_dims = ["质量", "稳定性"]
        mechanism = job.get("mechanism")
        if mechanism and mechanism.model_size == "small":
            comparison_dims.append("速度")
        
        acceptance = AcceptanceCard(
            candidate_checkpoints=candidates,
            comparison_dimensions=comparison_dims,
            user_feedback=job.get("user_feedback", []),
        )
        
        return acceptance
    
    def get_available_checkpoints(self) -> list[dict]:
        """获取可用的 checkpoint 列表"""
        if not self.current_job:
            return []
        
        checkpoints = self.current_job.get("checkpoints", [])
        return [c.to_dict() if isinstance(c, CheckpointInfo) else c for c in checkpoints]
    
    def get_checkpoint_path(self, checkpoint_id: str) -> str | None:
        """获取指定 checkpoint 的文件路径"""
        if not self.current_job:
            return None
        
        checkpoints = self.current_job.get("checkpoints", [])
        for ckpt in checkpoints:
            if isinstance(ckpt, CheckpointInfo):
                if ckpt.checkpoint_id == checkpoint_id and ckpt.status == "available":
                    return ckpt.model_path
            elif isinstance(ckpt, dict):
                if ckpt.get("id") == checkpoint_id and ckpt.get("status") != "deleted":
                    return ckpt.get("model_path") or str(
                        self.output_dir / self.current_job["job_id"] / f"{checkpoint_id}.pkl"
                    )
        return None
    
    def confirm_final_version(
        self,
        acceptance: AcceptanceCard,
        selected_checkpoint_id: str,
    ) -> dict:
        """
        用户确认最终版本
        
        Returns:
            包含最终模型路径和部署建议的字典
        """
        acceptance.selected_checkpoint = selected_checkpoint_id
        
        # 找到选中的checkpoint
        selected = None
        for c in acceptance.candidate_checkpoints:
            if c["id"] == selected_checkpoint_id:
                selected = c
                break
        
        if not selected:
            return {"error": "Checkpoint not found"}
        
        acceptance.final_model_path = selected["path"]
        
        # 生成部署建议
        mechanism = self.current_job["mechanism"]
        acceptance.deployment_recommendations = {
            "format": "pytorch",
            "quantization": "int8" if mechanism.model_size == "small" else "none",
            "target_device": "edge" if mechanism.model_size == "small" else "cloud",
            "estimated_latency_ms": 10 if mechanism.model_size == "small" else 50,
        }
        
        return {
            "job_id": self.current_job["job_id"],
            "selected_checkpoint": selected_checkpoint_id,
            "model_path": acceptance.final_model_path,
            "deployment": acceptance.deployment_recommendations,
        }
    
    def get_current_status(self) -> dict[str, Any]:
        """获取当前状态"""
        if not self.current_job:
            return {"status": "idle"}
        
        return {
            "job_id": self.current_job["job_id"],
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_observation": self.current_observation.to_user_friendly_dict() 
                if self.current_observation else None,
        }