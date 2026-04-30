"""
ML Engineer Agent - 核心入口

整合整个流程：
自然语言意图 -> 代码生成 -> QA验证 -> BO/ReAct优化 -> 可部署模型

使用示例:
    agent = MLEngineerAgent(llm_client=client)
    result = agent.train(
        intent="预测蛋白质结构，低质量样本多，需要不确定性建模",
        domain="ai4science",
        budget=60,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codegen import (
    GeneratedProgram,
    GPURecommendation,
    ModalSandboxExecutor,
    ProgramGenerator,
    QAPipeline,
    SandboxExecutor,
    StaticAnalyzer,
    UnitTestGenerator,
    env_default_gpu,
    recommend_gpu,
)
from .optimizer import HybridOptimizer, OptimizationResult


logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _format_qa_failure(stage_results: list[dict[str, Any]] | None) -> str | None:
    """Extract a compact first-failure summary from QA stage results."""
    if not stage_results:
        return None

    for attempt in stage_results:
        for stage in attempt.get("stages", []):
            stage_name = stage.get("name", "unknown")

            for result in stage.get("results", []):
                if result.get("passed", True):
                    continue
                for error in result.get("errors", []):
                    message = error.get("message")
                    if message:
                        location = error.get("file") or stage_name
                        line = error.get("line")
                        if line:
                            location = f"{location}:{line}"
                        return f"{stage_name} failed at {location}: {message}"
                return f"{stage_name} failed"

            result = stage.get("result")
            if isinstance(result, dict) and not result.get("passed", True):
                message = result.get("error_message") or result.get("stack_trace") or result.get("message")
                if message:
                    return f"{stage_name} failed: {message}"
                return f"{stage_name} failed"

    return None


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    intent: str
    domain: str
    status: str = "pending"  # pending, generating, validating, optimizing, completed, failed
    
    # 中间产物
    generated_program: GeneratedProgram | None = None
    qa_result: dict | None = None
    optimization_result: OptimizationResult | None = None
    
    # 最终结果
    final_model_path: str | None = None
    final_metrics: dict = field(default_factory=dict)
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    logs: list[str] = field(default_factory=list)
    gpu_recommendation: dict | None = None


class MLEngineerAgent:
    """
    ML Engineer Agent - 全自动深度学习训练
    
    这是系统的核心入口，整合了整个流程：
    1. 理解自然语言意图
    2. 生成领域特定的训练代码
    3. 多层QA确保代码正确性
    4. BO搜索最优超参数
    5. ReAct动态调整训练策略
    6. 输出可部署模型
    """
    
    def __init__(
        self,
        llm_client=None,
        output_dir: str = "./outputs",
        max_qa_attempts: int = 3,
        max_opt_iterations: int = 3,
        bo_budget_per_iter: int = 20,
        use_modal: bool | None = None,
        gpu_type_override: str | None = None,
    ):
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Modal 远端 GPU 训练后端开关：参数 > 环境变量
        if use_modal is None:
            use_modal = _env_truthy("USE_MODAL")
        self.use_modal = use_modal

        # GPU 选型不再在 ctor 时固定。
        # `gpu_type_override` 提供了一个"实例级 override"，主要给程序化调用用；
        # train() 还允许接收一次性的 `gpu_type=`；都没传时 Agent 自己决策。
        self.gpu_type_override = (gpu_type_override or "").strip() or None
        # 兜底：如果决策完全失败（理论上 recommend_gpu 永远不会失败），
        # 取 ENV `MODAL_GPU` 或硬编码默认。
        self.fallback_gpu = env_default_gpu()

        # 初始化组件
        self.generator = ProgramGenerator(llm_client=llm_client)
        self.analyzer = StaticAnalyzer()
        self.test_generator = UnitTestGenerator(llm_client=llm_client)

        # QA pipeline 的 sandbox：
        # - use_modal=False（开发机）→ 本地 SandboxExecutor，subprocess 跑
        # - use_modal=True（Zeabur）  → ModalSandboxExecutor 的 CPU QA 函数
        #   原因：Zeabur 后端容器不装 torch / cv2 / transformers，本地 QA 必失败。
        #   Modal qa_run_python 共用同一份训练镜像，CPU 运行只算 ~$0.0001/s，
        #   可以接受。
        if self.use_modal:
            # GPU 类型对 QA 函数无意义（QA 永远走 CPU 函数 qa_run_python），
            # 给个稳定占位即可。
            self.qa_sandbox = ModalSandboxExecutor(
                gpu_type=self.fallback_gpu,
                dataset_id=None,
                dataset_root=None,
            )
            logger.info("QA sandbox = ModalSandboxExecutor (CPU qa_run_python)")
        else:
            self.qa_sandbox = SandboxExecutor()

        # 训练 sandbox 默认指向 QA sandbox；调用 train() 时若 use_modal=True
        # 会被替换成带具体 dataset 的 ModalSandboxExecutor。
        self.train_sandbox = self.qa_sandbox

        self.qa_pipeline = QAPipeline(
            generator=self.generator,
            analyzer=self.analyzer,
            test_generator=self.test_generator,
            sandbox=self.qa_sandbox,
            max_attempts=max_qa_attempts,
        )

        self.optimizer = HybridOptimizer(
            qa_pipeline=self.qa_pipeline,
            sandbox=self.train_sandbox,
            llm_client=llm_client,
            max_iterations=max_opt_iterations,
            bo_budget_per_iter=bo_budget_per_iter,
        )

        # 任务记录
        self.jobs: dict[str, TrainingJob] = {}

    # 兼容旧字段名：optimizer/qa_pipeline 之外的代码可能还引用 self.sandbox
    @property
    def sandbox(self) -> SandboxExecutor:
        return self.train_sandbox

    def _build_train_sandbox(
        self,
        dataset_id: str | None,
        dataset_root: str | Path | None,
        target_metric: str,
        gpu_type: str | None = None,
    ) -> SandboxExecutor:
        """根据配置返回这次训练用的 sandbox。"""
        if not self.use_modal:
            return self.qa_sandbox

        chosen_gpu = (gpu_type or self.fallback_gpu).upper()
        sandbox = ModalSandboxExecutor(
            gpu_type=chosen_gpu,
            dataset_id=dataset_id,
            dataset_root=dataset_root,
            target_metric=target_metric,
        )
        logger.info(
            "Using ModalSandboxExecutor (gpu=%s, dataset_id=%s, dataset_root=%s)",
            chosen_gpu, dataset_id, dataset_root,
        )
        return sandbox
    
    def train(
        self,
        intent: str,
        domain: str = "general",
        data_schema: dict | None = None,
        constraints: dict | None = None,
        budget: int = 60,
        time_budget_sec: int = 3600,
        target_metric: str = "val_f1",
        target_threshold: float | None = None,
        progress_callback=None,
        dataset_id: str | None = None,
        dataset_root: str | Path | None = None,
        gpu_type: str | None = None,
    ) -> dict[str, Any]:
        """
        执行完整训练流程
        
        Args:
            intent: 自然语言意图，如"预测用户流失，宁可误报不要漏报"
            domain: 领域 (ai4science, gnn, timeseries, rl, general)
            data_schema: 数据schema描述
            constraints: 约束条件 (max_memory_gb, max_latency_ms, etc.)
            budget: BO trial预算
            time_budget_sec: 时间预算
            target_metric: 目标优化指标
            target_threshold: 目标阈值（达到即停止）
            
        Returns:
            包含训练结果、代码、指标的字典
        """
        job_id = f"job_{int(time.time())}_{hash(intent) % 10000}"
        job = TrainingJob(
            job_id=job_id,
            intent=intent,
            domain=domain,
        )
        self.jobs[job_id] = job

        # 自动从 data_schema 里推断 dataset_root（兼容 training_router 现有调用）
        if dataset_id is None and isinstance(data_schema, dict):
            dataset_id = data_schema.get("dataset_id")
        if dataset_root is None and isinstance(data_schema, dict):
            dataset_root = data_schema.get("storage_path")
        if dataset_root is None and dataset_id:
            # backend/data/<id> 兜底
            try:
                from backend.data_manager import DATA_DIR  # noqa: PLC0415
                candidate = Path(DATA_DIR) / dataset_id
                if candidate.exists():
                    dataset_root = candidate
            except Exception:
                pass

        # ─────────── Agent 自主决定 GPU 类型 ───────────
        # 优先级：本次调用 gpu_type > 实例 override > LLM 推荐 > 启发式 > ENV 兜底
        gpu_recommendation: GPURecommendation | None = None
        chosen_gpu: str | None = None
        if self.use_modal:
            override = gpu_type or self.gpu_type_override
            gpu_recommendation = recommend_gpu(
                intent=intent,
                data_schema=data_schema,
                constraints=constraints,
                llm_client=self.llm_client,
                user_override=override,
            )
            chosen_gpu = gpu_recommendation.gpu_type
            self._log(
                job,
                f"🤖 Agent 选定 GPU: {chosen_gpu} "
                f"(来源={gpu_recommendation.source}, "
                f"~${gpu_recommendation.estimated_cost_per_hour:.2f}/hr); "
                f"理由: {gpu_recommendation.reason}",
                progress_callback,
                "planning",
            )
            job.logs.append(
                f"GPU_DECISION: {gpu_recommendation.to_dict()}"
            )

        # 选择训练 sandbox（本地 vs Modal 远端 GPU）并注入 optimizer
        self.train_sandbox = self._build_train_sandbox(
            dataset_id=dataset_id,
            dataset_root=dataset_root,
            target_metric=target_metric,
            gpu_type=chosen_gpu,
        )
        self.optimizer.sandbox = self.train_sandbox

        # 把决策结果挂到 job 上，便于结果中导出
        job.gpu_recommendation = (
            gpu_recommendation.to_dict() if gpu_recommendation else None
        )

        try:
            # Stage 1: 代码生成（流式 + 5 文件并行）
            self._log(job, "Stage 1: 准备生成 5 个训练代码文件（并行流式）...", progress_callback, "generating")
            job.status = "generating"

            # 把 generator 的细粒度事件转换成 progress_callback 能消费的 agentic_log
            file_started_at: dict[str, float] = {}
            file_seen_chars: dict[str, int] = {}

            def _on_codegen_event(evt):
                etype = evt.get("type")
                if etype == "plan":
                    files = evt.get("files", [])
                    self._log(
                        job,
                        f"📦 计划生成 {len(files)} 个文件: {', '.join(files)}",
                        progress_callback, "generating",
                    )
                elif etype == "file_start":
                    name = evt["file"]
                    file_started_at[name] = time.time()
                    file_seen_chars[name] = 0
                    self._log(job, f"⚡ 开始生成 {name} ...", progress_callback, "generating")
                elif etype == "token":
                    # token 太碎，不用每个都打日志；改成每写满 ~1KB 报告一次
                    name = evt["file"]
                    file_seen_chars[name] = file_seen_chars.get(name, 0) + len(evt.get("text", ""))
                    if file_seen_chars[name] % 1024 < len(evt.get("text", "")):
                        self._log(
                            job,
                            f"… {name} 已生成 {file_seen_chars[name] / 1024:.1f} KB",
                            progress_callback, "generating",
                        )
                elif etype == "file_complete":
                    name = evt["file"]
                    self._log(
                        job,
                        f"✅ {name} 完成 ({evt.get('size', 0)} chars, {evt.get('duration', '?')}s)",
                        progress_callback, "generating",
                    )
                elif etype == "file_failed":
                    self._log(
                        job,
                        f"❌ {evt['file']} 第 {evt.get('attempt', 1)} 次尝试失败: {evt.get('error', '')[:120]}",
                        progress_callback, "generating",
                    )
                elif etype == "warning":
                    self._log(job, f"⚠️ {evt.get('message', '')}", progress_callback, "generating")
                elif etype == "done":
                    self._log(
                        job,
                        f"🎯 全部文件生成完成（总耗时 {evt.get('duration', '?')}s）",
                        progress_callback, "generating",
                    )

            program = self.generator.generate(
                intent=intent,
                data_schema=data_schema,
                constraints=constraints,
                on_event=_on_codegen_event,
            )
            job.generated_program = program
            
            # 保存生成的代码
            job_dir = self.output_dir / job_id
            program.save(job_dir / "generated")
            self._log(job, f"Code generated and saved to {job_dir / 'generated'}", progress_callback, "generating")
            
            # Stage 2: QA验证
            self._log(job, "Stage 2: 启动 QA pipeline（静态分析 → 单元测试 → 冒烟测试）...", progress_callback, "validating")
            job.status = "validating"

            stage_label = {
                "static_analysis": "静态分析",
                "unit_test": "单元测试",
                "smoke_test": "沙箱冒烟测试",
                "required_files": "必需文件检查",
            }

            def _on_qa_event(evt):
                etype = evt.get("type")
                if etype == "attempt_start":
                    self._log(
                        job,
                        f"🔁 QA 第 {evt['attempt']}/{evt.get('max', '?')} 轮开始",
                        progress_callback, "validating",
                    )
                elif etype == "stage_start":
                    name = stage_label.get(evt["stage"], evt["stage"])
                    self._log(job, f"🔍 {name} 检查中...", progress_callback, "validating")
                elif etype == "stage_complete":
                    name = stage_label.get(evt["stage"], evt["stage"])
                    n_errors = evt.get("errors", 0)
                    if evt.get("passed"):
                        # 真通过 → 绿勾
                        self._log(
                            job,
                            f"✅ {name} 通过 ({evt.get('duration', '?')}s)",
                            progress_callback, "validating",
                        )
                    else:
                        # 没通过：QA 设计就是「检查 → 自动修复 → 再检查」的循环，
                        # 这里不是"失败"，是"发现待修复问题"。改成中性措辞。
                        detail = evt.get("detail")
                        suffix = f" — {detail}" if detail else ""
                        self._log(
                            job,
                            f"🔍 {name} 发现 {n_errors} 处需调整 ({evt.get('duration', '?')}s){suffix}",
                            progress_callback, "validating",
                        )
                elif etype == "fix_start":
                    name = stage_label.get(evt["stage"], evt["stage"])
                    self._log(job, f"🔧 调用 LLM 修复 {name} 的问题中...", progress_callback, "validating")
                elif etype == "fix_complete":
                    name = stage_label.get(evt["stage"], evt["stage"])
                    self._log(job, f"🔧 {name} 修复完成，进入下一轮 QA", progress_callback, "validating")
                elif etype == "qa_done":
                    if evt.get("passed"):
                        if evt.get("soft_passed"):
                            reason = evt.get("soft_pass_reason") or "存在非致命提示"
                            self._log(
                                job,
                                f"⚠️ QA pipeline 软通过（共 {evt.get('attempts', '?')} 轮，"
                                f"{evt.get('duration', '?')}s）；{reason}，已放行训练",
                                progress_callback, "validating",
                            )
                        else:
                            self._log(
                                job,
                                f"✅ QA pipeline 全部通过（共 {evt.get('attempts', '?')} 轮，"
                                f"{evt.get('duration', '?')}s）",
                                progress_callback, "validating",
                            )
                    else:
                        self._log(
                            job,
                            f"❌ QA pipeline 在 {evt.get('attempts', '?')} 轮内未能修复全部问题，请检查日志",
                            progress_callback, "validating",
                        )

            qa_result = self.qa_pipeline.validate(program, auto_fix=True, on_event=_on_qa_event)
            job.qa_result = qa_result.to_dict()

            if not qa_result.passed:
                summary = _format_qa_failure(qa_result.stage_results)
                if summary:
                    self._log(job, f"QA failed after {qa_result.attempts} attempts: {summary}", progress_callback, "validating")
                else:
                    self._log(job, f"QA failed after {qa_result.attempts} attempts", progress_callback, "validating")
                job.status = "failed"
                return self._build_result(job)

            validated_program = qa_result.program
            if qa_result.soft_passed:
                # 软通过：剩下的问题不致命，已放行训练；上面 _on_qa_event 已经
                # 打过一条警告，这里再补一条带 attempt 数的 summary 方便 UI 检索。
                self._log(
                    job,
                    f"QA soft-passed in {qa_result.attempts} attempts: "
                    f"{qa_result.soft_pass_reason or '存在非致命提示'}",
                    progress_callback, "validating",
                )
            else:
                self._log(job, f"QA passed in {qa_result.attempts} attempts",
                          progress_callback, "validating")
            
            # 保存验证后的代码
            validated_program.save(job_dir / "validated")
            
            # Stage 3: 优化训练
            if isinstance(self.train_sandbox, ModalSandboxExecutor):
                backend_label = f"Modal 远端 GPU ({self.train_sandbox.gpu_type})"
            else:
                backend_label = "本地沙箱"
            self._log(
                job,
                f"Stage 3: Starting BO + ReAct optimization on {backend_label}...",
                progress_callback, "optimizing",
            )
            job.status = "optimizing"

            # 更新优化器目标
            self.optimizer.target_metric = target_metric
            self.optimizer.target_threshold = target_threshold
            
            opt_result = self.optimizer.optimize(
                initial_program=validated_program,
                total_budget=budget,
                time_budget_sec=time_budget_sec,
            )
            job.optimization_result = opt_result
            
            # Stage 4: 保存最终结果
            job.status = "completed"
            job.completed_at = time.time()
            
            if opt_result.best_program:
                opt_result.best_program.save(job_dir / "final")
                job.final_model_path = str(job_dir / "final")
                job.final_metrics = {
                    "best_score": opt_result.best_score,
                    "target_metric": target_metric,
                }
                self._log(job, f"Training completed. Best score: {opt_result.best_score:.4f}", progress_callback, "completed")
            
            return self._build_result(job)
            
        except Exception as e:
            job.status = "failed"
            self._log(job, f"Error: {str(e)}", progress_callback, "failed")
            import traceback
            self._log(job, traceback.format_exc(), progress_callback, "failed")
            return self._build_result(job)
    
    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """获取任务状态"""
        job = self.jobs.get(job_id)
        if not job:
            return {"error": "Job not found"}
        
        return {
            "job_id": job_id,
            "status": job.status,
            "intent": job.intent,
            "created_at": job.created_at,
            "elapsed_sec": time.time() - job.created_at,
            "logs": job.logs[-20:] if job.logs else [],  # 最后20条日志
        }
    
    def _log(self, job: TrainingJob, message: str, progress_callback=None, stage: str | None = None):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        job.logs.append(log_entry)
        print(log_entry)
        if progress_callback:
            progress_callback({
                "step": "agentic_log",
                "agent_stage": stage or job.status,
                "message": message,
                "log_entry": log_entry,
            })
    
    def _build_result(self, job: TrainingJob) -> dict[str, Any]:
        """构建返回结果"""
        result = {
            "job_id": job.job_id,
            "status": job.status,
            "intent": job.intent,
            "domain": job.domain,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
        
        if job.qa_result:
            result["qa"] = {
                "passed": job.qa_result.get("passed"),
                "attempts": job.qa_result.get("attempts"),
                "duration_ms": job.qa_result.get("total_duration_ms"),
                "stage_results": job.qa_result.get("stage_results", []),
                "failure_summary": _format_qa_failure(job.qa_result.get("stage_results")),
            }
        
        if job.optimization_result:
            opt = job.optimization_result
            result["optimization"] = {
                "status": opt.status,
                "best_score": opt.best_score,
                "best_config": opt.best_config,
                "total_trials": len(opt.all_trials),
                "iterations": opt.iterations,
                "duration_sec": opt.total_duration_sec,
            }

        if job.gpu_recommendation:
            result["gpu"] = job.gpu_recommendation
        
        if job.final_model_path:
            result["output"] = {
                "model_path": job.final_model_path,
                "metrics": job.final_metrics,
            }
        
        if job.logs:
            result["logs"] = job.logs
        
        return result


# 便捷函数
def quick_train(
    intent: str,
    domain: str = "general",
    llm_client=None,
    progress_callback=None,
    **kwargs
) -> dict[str, Any]:
    """
    快速训练入口

    支持的额外 kwargs（透传到 MLEngineerAgent.train）：
        - data_schema, constraints, budget, time_budget_sec
        - target_metric, target_threshold
        - dataset_id, dataset_root
        - gpu_type  ← Agent 也会自己决定 GPU；除非这里显式指定，否则不会强制覆盖

    Example:
        result = quick_train(
            intent="预测用户流失，宁可误报不要漏报",
            domain="general",
            budget=30,
        )
    """
    agent = MLEngineerAgent(llm_client=llm_client)
    return agent.train(
        intent=intent,
        domain=domain,
        progress_callback=progress_callback,
        **kwargs
    )
