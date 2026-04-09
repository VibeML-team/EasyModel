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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codegen import (
    GeneratedProgram,
    ProgramGenerator,
    QAPipeline,
    SandboxExecutor,
    StaticAnalyzer,
    UnitTestGenerator,
)
from .optimizer import HybridOptimizer, OptimizationResult


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
    ):
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.generator = ProgramGenerator(llm_client=llm_client)
        self.analyzer = StaticAnalyzer()
        self.test_generator = UnitTestGenerator(llm_client=llm_client)
        self.sandbox = SandboxExecutor()
        
        self.qa_pipeline = QAPipeline(
            generator=self.generator,
            analyzer=self.analyzer,
            test_generator=self.test_generator,
            sandbox=self.sandbox,
            max_attempts=max_qa_attempts,
        )
        
        self.optimizer = HybridOptimizer(
            qa_pipeline=self.qa_pipeline,
            sandbox=self.sandbox,
            llm_client=llm_client,
            max_iterations=max_opt_iterations,
            bo_budget_per_iter=bo_budget_per_iter,
        )
        
        # 任务记录
        self.jobs: dict[str, TrainingJob] = {}
    
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
        
        try:
            # Stage 1: 代码生成
            self._log(job, "Stage 1: Generating training program...", progress_callback, "generating")
            job.status = "generating"
            
            program = self.generator.generate(
                intent=intent,
                data_schema=data_schema,
                constraints=constraints,
            )
            job.generated_program = program
            
            # 保存生成的代码
            job_dir = self.output_dir / job_id
            program.save(job_dir / "generated")
            self._log(job, f"Code generated and saved to {job_dir / 'generated'}", progress_callback, "generating")
            
            # Stage 2: QA验证
            self._log(job, "Stage 2: Running QA pipeline...", progress_callback, "validating")
            job.status = "validating"
            
            qa_result = self.qa_pipeline.validate(program, auto_fix=True)
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
            self._log(job, f"QA passed in {qa_result.attempts} attempts", progress_callback, "validating")
            
            # 保存验证后的代码
            validated_program.save(job_dir / "validated")
            
            # Stage 3: 优化训练
            self._log(job, "Stage 3: Starting BO + ReAct optimization...", progress_callback, "optimizing")
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
