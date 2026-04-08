"""
V2 API - 简洁的用户流程

三步走:
1. POST /v2/compile - 编译意图，返回三张卡供用户确认
2. POST /v2/train - 用户确认后开始训练，返回观测流
3. POST /v2/finalize - 训练完成，用户选择最终版本
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Any, Literal

from .v2_intent_compiler import (
    AcceptanceCard,
    AugmentationStyle,
    DataConstructionCard,
    ErrorPreference,
    IntentCard,
    IntentCompilerV2,
    QualitySpeedCost,
    TrainingPlanCard,
)
from .v2_training_controller import TrainingControllerV2


router = APIRouter(prefix="/v2")

# 全局状态（生产环境应使用数据库）
compiler = IntentCompilerV2()
controller = TrainingControllerV2()
active_jobs: dict[str, dict] = {}


# ============ 请求/响应模型 ============

class CompileRequest(BaseModel):
    intent: str
    domain: Literal["general", "ai4science", "gnn", "timeseries", "rl"] = "general"
    data_preview: dict | None = None


class CompileResponse(BaseModel):
    session_id: str
    intent_card: dict
    data_card: dict
    plan_card: dict
    summary: str
    can_proceed: bool


class RefineRequest(BaseModel):
    session_id: str
    intent_feedback: dict | None = None      # 修改意图卡
    data_feedback: dict | None = None        # 修改数据卡
    plan_feedback: dict | None = None        # 修改方案卡


class RefineResponse(BaseModel):
    session_id: str
    intent_card: dict
    data_card: dict
    plan_card: dict
    changes_applied: list[str]
    summary: str


class StartTrainingRequest(BaseModel):
    session_id: str
    confirmed_cards: dict | None = None  # 用户确认后的三张卡（可选）
    dataset_id: str | None = None  # 数据集ID


class StartTrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str  # running / paused / completed / failed
    observation: dict | None  # 当前观测
    progress_percent: int


class FeedbackRequest(BaseModel):
    job_id: str
    checkpoint_id: str
    liked: bool
    reason: str | None = None
    aspects: dict | None = None


class FinalizeResponse(BaseModel):
    job_id: str
    acceptance_card: dict
    message: str


class ClarifyRequest(BaseModel):
    """澄清请求"""
    session_id: str
    answers: list[dict]  # [{"question_id": "...", "answer": "..."}]


class ClarifyResponse(BaseModel):
    """澄清响应"""
    session_id: str
    intent_card: dict
    clarification_complete: bool  # 是否已完成澄清
    remaining_questions: list[dict]  # 剩余问题
    message: str


class SelectVersionRequest(BaseModel):
    job_id: str
    checkpoint_id: str


class SelectVersionResponse(BaseModel):
    job_id: str
    selected_checkpoint: str
    model_path: str
    deployment_recommendations: dict


# ============ API端点 ============

@router.post("/compile", response_model=CompileResponse)
async def compile_intent(request: CompileRequest):
    """
    第一步: 编译用户意图
    
    系统解析自然语言，生成三张卡供用户审阅。
    用户可以在后续步骤中修改这些卡。
    """
    # 编译意图
    intent_card, data_card, plan_card = compiler.compile(
        intent=request.intent,
        data_preview=request.data_preview,
        domain=request.domain,
    )
    
    # 生成session_id
    import time
    session_id = f"sess_{int(time.time() * 1000)}"
    
    # 存储
    active_jobs[session_id] = {
        "session_id": session_id,
        "intent_card": intent_card,
        "data_card": data_card,
        "plan_card": plan_card,
        "status": "compiled",
    }
    
    # 生成总结
    summary = compiler.generate_summary((intent_card, data_card, plan_card))
    
    # 根据模糊性检测确定是否可以继续
    can_proceed = not intent_card.ambiguity_detection.is_ambiguous
    
    return CompileResponse(
        session_id=session_id,
        intent_card=intent_card.to_user_friendly_dict(),
        data_card=data_card.to_user_friendly_dict(),
        plan_card=plan_card.to_user_friendly_dict(),
        summary=summary,
        can_proceed=can_proceed,
    )


@router.post("/refine", response_model=RefineResponse)
async def refine_intent(request: RefineRequest):
    """
    修改编译结果
    
    用户可以对系统理解进行修正，这是关键介入点。
    """
    if request.session_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Session not found")
    
    job = active_jobs[request.session_id]
    changes = []
    
    # 应用意图反馈
    if request.intent_feedback:
        job["intent_card"] = compiler.refine_intent(
            job["intent_card"],
            request.intent_feedback,
        )
        changes.append("intent")
    
    # 应用数据反馈
    if request.data_feedback:
        job["data_card"] = compiler.refine_data_construction(
            job["data_card"],
            request.data_feedback,
        )
        changes.append("data")
    
    # 生成新总结
    summary = compiler.generate_summary(
        (job["intent_card"], job["data_card"], job["plan_card"])
    )
    
    return RefineResponse(
        session_id=request.session_id,
        intent_card=job["intent_card"].to_user_friendly_dict(),
        data_card=job["data_card"].to_user_friendly_dict(),
        plan_card=job["plan_card"].to_user_friendly_dict(),
        changes_applied=changes,
        summary=summary,
    )


@router.post("/clarify", response_model=ClarifyResponse)
async def clarify_intent(request: ClarifyRequest):
    """
    意图澄清 - 回答系统生成的追问
    
    当系统检测到意图模糊时，会生成追问问题。
    用户回答后，系统会更新意图卡并重新评估。
    """
    if request.session_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Session not found")
    
    job = active_jobs[request.session_id]
    intent_card = job["intent_card"]
    
    # 应用用户回答
    intent_card = compiler.clarify_with_answers(intent_card, request.answers)
    job["intent_card"] = intent_card
    
    # 检查是否还有未回答的问题
    remaining = [
        {
            "id": q.question_id,
            "text": q.question_text,
            "type": q.question_type,
            "options": q.options,
            "context": q.context,
        }
        for q in intent_card.ambiguity_detection.suggested_questions
        if not q.is_answered
    ]
    
    clarification_complete = not intent_card.ambiguity_detection.is_ambiguous
    
    message = (
        "澄清完成，可以开始训练" if clarification_complete
        else f"还有 {len(remaining)} 个问题需要回答"
    )
    
    return ClarifyResponse(
        session_id=request.session_id,
        intent_card=intent_card.to_user_friendly_dict(),
        clarification_complete=clarification_complete,
        remaining_questions=remaining,
        message=message,
    )


@router.post("/train/start", response_model=StartTrainingResponse)
async def start_training(
    request: StartTrainingRequest,
    background_tasks: BackgroundTasks,
):
    """
    第二步: 开始训练
    
    用户确认卡片后，启动训练流程。
    训练过程中用户可以通过 /train/status 观察进度。
    """
    if request.session_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Session not found")
    
    job = active_jobs[request.session_id]
    
    # 更新为确认后的卡片
    if request.confirmed_cards and request.confirmed_cards.get("intent"):
        # 这里应该解析并更新卡片
        pass
    
    # 启动训练
    def progress_callback(observation):
        # 实际应通过WebSocket推送
        job["current_observation"] = observation
    
    # 获取数据集ID（从请求或session）
    dataset_id = request.dataset_id or job.get("dataset_id")
    
    job_id = controller.start_training(
        intent=job["intent_card"],
        data=job["data_card"],
        plan=job["plan_card"],
        dataset_id=dataset_id,
        progress_callback=progress_callback,
    )
    
    job["job_id"] = job_id
    job["status"] = "training"
    
    return StartTrainingResponse(
        job_id=job_id,
        status="started",
        message="训练已启动，请通过 /train/status 查看进度",
    )


@router.get("/train/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """获取训练状态和当前观测"""
    status = controller.get_current_status()
    
    observation = status.get("current_observation")
    progress = 0
    if observation:
        progress = observation.get("progress", {}).get("percent", 0)
    
    return TrainingStatusResponse(
        job_id=job_id,
        status="running" if status.get("is_running") else "idle",
        observation=observation,
        progress_percent=progress,
    )


@router.post("/train/pause/{job_id}")
async def pause_training(job_id: str):
    """暂停训练"""
    success = controller.pause()
    return {"success": success, "message": "训练已暂停" if success else "无法暂停"}


@router.post("/train/resume/{job_id}")
async def resume_training(job_id: str):
    """恢复训练"""
    success = controller.resume()
    return {"success": success, "message": "训练已恢复" if success else "无法恢复"}


@router.post("/train/feedback")
async def provide_feedback(request: FeedbackRequest):
    """
    对中间结果提供反馈
    
    例如：{"liked": false, "reason": "风格不一致"}
    系统会据此调整训练策略。
    """
    success = controller.provide_feedback(
        checkpoint_id=request.checkpoint_id,
        feedback={
            "liked": request.liked,
            "reason": request.reason,
            "aspects": request.aspects,
        },
    )
    
    return {
        "success": success,
        "message": "反馈已记录，将用于优化训练" if success else "无法记录反馈",
    }


@router.post("/train/finalize/{job_id}", response_model=FinalizeResponse)
async def finalize_training(job_id: str):
    """
    第三步: 完成训练，生成候选版本
    
    系统生成多个checkpoint供用户选择。
    """
    # 停止训练（如果还在运行）
    controller.stop()
    
    # 生成验收卡
    acceptance = controller.finalize()
    
    return FinalizeResponse(
        job_id=job_id,
        acceptance_card=acceptance.to_user_friendly_dict(),
        message="训练完成，请选择最终版本",
    )


@router.post("/train/select", response_model=SelectVersionResponse)
async def select_final_version(request: SelectVersionRequest):
    """
    选择最终版本
    
    用户从候选checkpoint中选择最合适的版本。
    """
    # 获取当前验收卡
    acceptance = controller.finalize()
    
    result = controller.confirm_final_version(acceptance, request.checkpoint_id)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return SelectVersionResponse(
        job_id=request.job_id,
        selected_checkpoint=request.checkpoint_id,
        model_path=result.get("model_path", ""),
        deployment_recommendations=result.get("deployment_recommendations", {}),
    )


# ============ Checkpoint 管理端点 ============

@router.get("/train/checkpoints/{job_id}")
async def list_checkpoints(job_id: str):
    """
    获取训练过程中可用的 checkpoint 列表
    
    用户可以在训练过程中随时查看和下载中间 checkpoint 进行测试。
    """
    checkpoints = controller.get_available_checkpoints()
    return {
        "job_id": job_id,
        "checkpoints": checkpoints,
        "count": len(checkpoints),
    }


@router.get("/train/checkpoints/{job_id}/{checkpoint_id}/download")
async def download_checkpoint(job_id: str, checkpoint_id: str):
    """
    下载指定 checkpoint 进行测试
    
    用户可以暂停训练，下载当前 checkpoint 进行验证，
    如果效果满意可以提前结束训练。
    """
    from fastapi.responses import FileResponse
    
    checkpoint_path = controller.get_checkpoint_path(checkpoint_id)
    
    if not checkpoint_path:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    import os
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail="Checkpoint file not found")
    
    return FileResponse(
        path=checkpoint_path,
        filename=f"{checkpoint_id}.pkl",
        media_type="application/octet-stream",
    )


@router.post("/train/pause-and-download/{job_id}")
async def pause_and_download(job_id: str):
    """
    暂停训练并返回最新的 checkpoint 信息
    
    方便用户暂停后立即下载测试。
    """
    # 暂停训练
    controller.pause()
    
    # 获取最新 checkpoint
    checkpoints = controller.get_available_checkpoints()
    
    if not checkpoints:
        return {
            "success": True,
            "message": "训练已暂停，暂无可用 checkpoint",
            "checkpoints": [],
        }
    
    # 返回最好的 checkpoint
    best = max(checkpoints, key=lambda x: x.get("metric", 0))
    
    return {
        "success": True,
        "message": "训练已暂停，可以下载 checkpoint 进行测试",
        "checkpoints": checkpoints,
        "best_checkpoint": best,
        "download_url": f"/v2/train/checkpoints/{job_id}/{best['id']}/download",
    }


# ============ 简化流程端点 ============

class QuickTrainRequest(BaseModel):
    """快速训练请求 - 最小介入版本"""
    intent: str
    quality_speed_cost: QualitySpeedCost = QualitySpeedCost.BALANCED
    augmentation_style: AugmentationStyle = AugmentationStyle.STANDARD
    domain: Literal["general", "ai4science", "gnn", "timeseries", "rl"] = "general"
    dataset_id: str | None = None


@router.post("/quick-train")
async def quick_train(request: QuickTrainRequest):
    """
    快速训练 - 最小介入版本
    
    用户只提供意图和几个高层选择，其他全自动。
    """
    # 1. 编译
    intent_card, data_card, plan_card = compiler.compile(
        intent=request.intent,
        domain=request.domain,
    )
    
    # 应用用户选择
    intent_card.quality_speed_cost = request.quality_speed_cost
    data_card.augmentation_style = request.augmentation_style
    plan_card.quality_speed_cost = request.quality_speed_cost
    
    # 2. 自动开始训练（跳过确认步骤）
    job_id = controller.start_training(
        intent=intent_card,
        data=data_card,
        plan=plan_card,
        dataset_id=request.dataset_id,
    )
    
    return {
        "job_id": job_id,
        "message": "快速训练已启动",
        "status_endpoint": f"/v2/train/status/{job_id}",
    }