from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.compiler import ObjectiveSpec
from backend.data_manager import DataSpec


def _extract_qa_failure_message(qa: dict[str, Any]) -> str | None:
    summary = qa.get("failure_summary")
    if summary:
        return summary

    for attempt in qa.get("stage_results", []):
        for stage in attempt.get("stages", []):
            stage_name = stage.get("name", "qa")

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
class TrainingPlan:
    """Executable training plan derived from existing planning artifacts."""

    modality: str
    executor: str
    domain: str
    strategy: str
    summary: str
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "executor": self.executor,
            "domain": self.domain,
            "strategy": self.strategy,
            "summary": self.summary,
            "inputs": self.inputs,
        }


def _normalize_modality(data_spec: DataSpec) -> str:
    explored = (data_spec.exploration or {}).get("data_type") or data_spec.data_type or "unknown"
    value = str(explored).lower()

    if "tabular" in value:
        return "tabular"
    if "image" in value or "vision" in value:
        return "image"
    if "text" in value or "nlp" in value:
        return "text"
    if "audio" in value or "speech" in value:
        return "audio"
    if "time_series" in value or "timeseries" in value:
        return "time_series"
    return value or "unknown"


def build_training_plan(
    objective_spec: ObjectiveSpec,
    data_spec: DataSpec,
    config: dict[str, Any] | None = None,
) -> TrainingPlan:
    """
    Convert existing planning artifacts into an executable training plan.

    This function is intentionally deterministic: planning is already done by the
    objective compiler and data exploration stages. The router only decides which
    executor should carry out that plan.
    """
    config = config or {}
    modality = _normalize_modality(data_spec)

    if modality == "tabular":
        return TrainingPlan(
            modality=modality,
            executor="tabular_automl",
            domain="general",
            strategy="structured_automl",
            summary="Use tabular AutoML executor with dataframe loading and sklearn-family search.",
            inputs={
                "dataset_id": data_spec.dataset_id,
                "target_column": config.get("target_column") or objective_spec.label.target_column,
                "max_training_time": config.get("max_training_time", objective_spec.max_training_time),
                "max_trials": config.get("max_trials", objective_spec.max_trials),
            },
        )

    domain_map = {
        "image": "vision",
        "text": "nlp",
        "audio": "audio",
        "time_series": "timeseries",
    }

    return TrainingPlan(
        modality=modality,
        executor="agentic_ml_engineer",
        domain=domain_map.get(modality, "general"),
        strategy="plan_then_execute_with_codegen",
        summary=(
            "Use agentic executor to translate existing objective and dataset understanding "
            "into a modality-aware training program instead of forcing dataframe AutoML."
        ),
        inputs={
            "dataset_id": data_spec.dataset_id,
            "data_type": data_spec.data_type,
            "exploration": data_spec.exploration,
            "file_scan": data_spec.file_scan,
            "storage_path": data_spec.storage_path,
            "objective": objective_spec.to_dict(),
        },
    )


def execute_training_plan(
    *,
    job_id: str,
    plan: TrainingPlan,
    objective_spec: ObjectiveSpec,
    progress_callback=None,
):
    """
    Execute a TrainingPlan.

    All tasks go through this router, including tabular workloads.
    """
    if plan.executor == "tabular_automl":
        from backend.data_manager import data_manager
        from backend.trainer import AutoMLTrainer, TrainingConfig

        if progress_callback:
            progress_callback({
                "step": "loading_data",
                "message": "按执行规划加载表格数据...",
                "plan": plan.to_dict(),
            })

        df = data_manager.load_dataframe(plan.inputs["dataset_id"])

        if progress_callback:
            progress_callback({
                "step": "preprocessing",
                "message": f"按规划读取完成，{len(df)} 行 {len(df.columns)} 列",
                "plan": plan.to_dict(),
            })

        config = TrainingConfig(
            max_training_time=plan.inputs.get("max_training_time", 300),
            max_trials=plan.inputs.get("max_trials", 30),
            random_state=42,
        )
        trainer = AutoMLTrainer(config)
        return trainer.train(
            job_id=job_id,
            df=df,
            spec=objective_spec,
            progress_callback=progress_callback,
        )

    if plan.executor == "agentic_ml_engineer":
        from backend.llm_client import get_llm_client
        from backend.ml_engineer import quick_train
        from backend.trainer import TrainingResult

        if progress_callback:
            progress_callback({
                "step": "planning",
                "message": f"按执行规划启动 {plan.modality} agentic executor...",
                "plan": plan.to_dict(),
            })

        llm_client = get_llm_client()
        intent = objective_spec.raw_intent or f"Train a {plan.modality} machine learning model"
        result = quick_train(
            intent=intent,
            domain=plan.domain,
            llm_client=llm_client,
            progress_callback=progress_callback,
            data_schema=plan.inputs,
            constraints=objective_spec.constraints,
            budget=objective_spec.max_trials or 20,
            time_budget_sec=objective_spec.max_training_time or 300,
        )

        status = result.get("status", "failed")
        best_score = (
            (result.get("optimization") or {}).get("best_score")
            or (result.get("output") or {}).get("metrics", {}).get("best_score")
        )
        logs = result.get("logs") or []
        qa = result.get("qa") or {}
        optimization = result.get("optimization") or {}

        error_message = None
        if status != "completed":
            if qa and qa.get("passed") is False:
                qa_failure = _extract_qa_failure_message(qa)
                if qa_failure:
                    error_message = f"QA failed after {qa.get('attempts', '?')} attempts: {qa_failure}"
                else:
                    error_message = f"QA failed after {qa.get('attempts', '?')} attempts"
            elif optimization.get("status") == "failed":
                error_message = "Optimization failed"
            elif logs:
                error_message = logs[-1]
            else:
                error_message = "Agentic executor failed"

        return TrainingResult(
            job_id=job_id,
            status=status,
            best_model_name=f"agentic_{plan.modality}",
            best_metric_score=best_score,
            training_duration=0.0,
            error_message=error_message,
        )

    raise ValueError(f"Unknown training executor: {plan.executor}")
