"""
VibeML Agent - 完整后端实现

端到端流程：
1. 数据上传 -> DataSpec
2. 意图澄清 -> ObjectiveSpec (chat2objective)
3. 训练执行 -> 真实模型权重 (chat2model)
4. 模型下载 -> 可部署产物
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 导入新模块
from backend.compiler import compile_objective, ObjectiveSpec
from backend.data_manager import data_manager, DataSpec, DataManager
from backend.trainer import AutoMLTrainer, TrainingConfig, TrainingResult, CHECKPOINT_DIR

# 导入 V2 API
from backend.v2_api import router as v2_router

# 目录设置
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

Priority = Literal["quality", "latency", "cost"]
Mode = Literal["conservative", "standard", "aggressive"]


# ============ Pydantic 模型 ============

class ClarifyRequest(BaseModel):
    user_goal: str = Field(min_length=5)
    must_keep: list[str] = Field(default_factory=list)
    can_change: list[str] = Field(default_factory=list)
    worst_errors: list[str] = Field(default_factory=list)
    priority: Priority = "quality"
    dataset_id: str | None = None


class ClarifyResponse(BaseModel):
    is_ambiguous: bool
    reasons: list[str]
    follow_up_questions: list[str]
    compiled_intent: dict[str, Any]
    objective_spec: dict[str, Any] | None = None


class StartTrainingRequest(BaseModel):
    objective: str = Field(min_length=5)
    must_keep: list[str] = Field(default_factory=list)
    can_change: list[str] = Field(default_factory=list)
    worst_errors: list[str] = Field(default_factory=list)
    priority: Priority = "quality"
    mode: Mode = "standard"
    sample_notes: str = ""
    dataset_id: str | None = None
    target_column: str | None = None
    max_training_time: int = 300  # 秒
    max_trials: int = 30


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # running, completed, failed, stopped, paused
    progress: float
    current_step: str
    message: str
    train_metrics: dict[str, float] = Field(default_factory=dict)
    val_metrics: dict[str, float] = Field(default_factory=dict)
    best_score: float | None = None
    elapsed_time: float = 0


# ============ 任务状态管理 ============

@dataclass
class JobState:
    """任务状态"""
    job_id: str
    created_at: str
    status: str = "running"
    current_step: str = "initializing"
    progress: float = 0.0
    message: str = ""
    
    # 训练配置
    config: dict[str, Any] = field(default_factory=dict)
    objective_spec: ObjectiveSpec | None = None
    dataset_id: str | None = None
    
    # 训练指标
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    
    # 训练结果
    training_result: TrainingResult | None = None
    
    # 时间记录
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    
    # 线程控制
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_requested: bool = False


class JobManager:
    """任务管理器"""
    
    def __init__(self):
        self.jobs: dict[str, JobState] = {}
        self.trainer = AutoMLTrainer()
    
    def create_job(self, cfg: dict[str, Any], spec: ObjectiveSpec | None = None) -> JobState:
        """创建新任务"""
        job_id = f"vml_{uuid.uuid4().hex[:8]}"
        job = JobState(
            job_id=job_id,
            created_at=_utc_now(),
            config=cfg,
            objective_spec=spec,
            dataset_id=cfg.get("dataset_id"),
        )
        self.jobs[job_id] = job
        
        # 启动训练线程
        thread = threading.Thread(target=self._training_worker, args=(job,), daemon=True)
        thread.start()
        
        return job
    
    def get(self, job_id: str) -> JobState:
        """获取任务状态"""
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
        return job
    
    def _training_worker(self, job: JobState):
        """训练工作线程"""
        try:
            with job.lock:
                if job.stop_requested:
                    job.status = "stopped"
                    return
                job.current_step = "loading_data"
                job.message = "加载数据中..."
            
            # 1. 加载数据
            if not job.dataset_id:
                raise ValueError("未指定数据集")
            
            df = data_manager.load_dataframe(job.dataset_id)
            
            with job.lock:
                job.progress = 10.0
                job.current_step = "preprocessing"
                job.message = f"数据加载完成，{len(df)} 行 {len(df.columns)} 列"
            
            # 2. 更新目标列（如果用户指定）
            if job.config.get("target_column"):
                if job.objective_spec:
                    job.objective_spec.label.target_column = job.config["target_column"]
            
            # 3. 执行训练
            config = TrainingConfig(
                max_training_time=job.config.get("max_training_time", 300),
                max_trials=job.config.get("max_trials", 30),
                random_state=42,
            )
            
            trainer = AutoMLTrainer(config)
            
            def progress_callback(progress: dict):
                with job.lock:
                    if job.stop_requested:
                        raise InterruptedError("训练被中断")
                    
                    step = progress.get("step", "")
                    if step == "preprocessing":
                        job.progress = 15.0
                        job.message = progress.get("message", "预处理中...")
                    elif step == "search":
                        trial = progress.get("trial", 0)
                        best = progress.get("best_score", 0)
                        job.progress = 15.0 + min(trial * 2, 70.0)
                        job.message = f"超参数搜索中... Trial {trial}, 最佳得分: {best:.4f}"
                        job.train_metrics["best_cv_score"] = best
                    elif step == "final_training":
                        job.progress = 90.0
                        job.message = "训练最终模型..."
            
            with job.lock:
                job.current_step = "training"
                job.message = "开始训练..."
            
            result = trainer.train(
                job_id=job.job_id,
                df=df,
                spec=job.objective_spec or ObjectiveSpec(),
                progress_callback=progress_callback,
            )
            
            with job.lock:
                job.training_result = result
                job.status = result.status
                job.progress = 100.0
                job.end_time = time.time()
                
                if result.status == "completed":
                    job.message = f"训练完成！最佳模型: {result.best_model_name}, 验证得分: {result.best_metric_score:.4f}"
                    job.train_metrics.update(result.train_metrics)
                    job.val_metrics.update(result.val_metrics)
                else:
                    job.message = f"训练失败: {result.error_message}"
            
        except InterruptedError:
            with job.lock:
                job.status = "stopped"
                job.message = "训练已停止"
                job.end_time = time.time()
        except Exception as e:
            with job.lock:
                job.status = "failed"
                job.message = f"训练失败: {str(e)}"
                job.end_time = time.time()
    
    def pause_job(self, job_id: str) -> JobState:
        """暂停任务（实际上是保存当前状态，训练继续到下一个检查点）"""
        job = self.get(job_id)
        with job.lock:
            if job.status == "running":
                job.status = "paused"
        return job
    
    def resume_job(self, job_id: str) -> JobState:
        """恢复任务"""
        job = self.get(job_id)
        with job.lock:
            if job.status == "paused":
                job.status = "running"
        return job
    
    def stop_job(self, job_id: str) -> JobState:
        """停止任务"""
        job = self.get(job_id)
        with job.lock:
            job.stop_requested = True
            job.status = "stopped"
            job.end_time = time.time()
        return job


# 全局任务管理器
job_manager = JobManager()


# ============ FastAPI 应用 ============

app = FastAPI(
    title="VibeML Agent",
    version="1.0.0",
    description="从自然语言需求到可部署模型权重的全自动 ML 平台",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 V2 API
app.include_router(v2_router, prefix="/api")

# 挂载前端
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="app")


# ============ 辅助函数 ============

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_ambiguity(req: ClarifyRequest) -> tuple[bool, list[str], list[str]]:
    """检测需求模糊性"""
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
        follow_ups.append("请补充：谁在什么场景使用、怎样才算成功。")
    
    if not req.dataset_id:
        reasons.append("未上传数据集")
        follow_ups.append("请上传包含训练数据的 CSV 文件。")
    
    return len(reasons) > 0, reasons, follow_ups


# ============ API 端点 ============

@app.get("/")
def root() -> FileResponse:
    """根路径 - 返回前端"""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return JSONResponse({
            "message": "VibeML Agent API",
            "version": "1.0.0",
            "docs": "/docs",
        })
    return FileResponse(index)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """健康检查"""
    # 检查 LLM 配置
    llm_configured = False
    llm_model = None
    try:
        from backend.llm_client import LLMConfig
        config = LLMConfig()
        llm_configured = True
        llm_model = config.model_name
    except:
        pass
    
    return {
        "status": "ok", 
        "version": "1.0.0",
        "llm_configured": llm_configured,
        "llm_model": llm_model,
        "features": {
            "llm_intent_parsing": llm_configured,
            "rule_fallback": True,
            "automL_training": True,
        }
    }


# ---- 数据管理 API ----

@app.post("/api/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    target_hint: str = Form(""),
) -> dict[str, Any]:
    """
    上传数据文件
    
    支持格式：
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - ZIP (.zip) - 包含上述格式的压缩包
    - 7Z (.7z) - 包含上述格式的压缩包
    """
    # 检查文件扩展名
    allowed_extensions = ['.csv', '.xlsx', '.xls', '.zip', '.7z']
    filename_lower = file.filename.lower()
    
    if not any(filename_lower.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(400, f"不支持的文件格式。支持: {', '.join(allowed_extensions)}")
    
    content = await file.read()
    
    try:
        spec = data_manager.upload_file(
            content=content,
            filename=file.filename,
            target_hint=target_hint if target_hint else None,
        )
        return {
            "success": True,
            "dataset_id": spec.dataset_id,
            "filename": spec.filename,
            "n_rows": spec.n_rows,
            "n_cols": spec.n_cols,
            "target_column": spec.target_column,
            "feature_columns": spec.feature_columns,
            "columns": [c.to_dict() for c in spec.columns],
        }
    except Exception as e:
        raise HTTPException(400, f"数据处理失败: {str(e)}")


# ---- Google Drive 集成 API ----

@app.get("/api/drive/status")
def google_drive_status() -> dict[str, Any]:
    """检查 Google Drive 集成状态"""
    # 检查是否配置了 Google Drive API
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    
    return {
        "enabled": bool(client_id and client_secret),
        "client_id_configured": bool(client_id),
        "auth_url": "/api/drive/auth" if client_id else None,
    }


@app.get("/api/drive/auth")
def google_drive_auth():
    """获取 Google Drive 授权 URL"""
    # 预留：实际实现需要 Google OAuth 流程
    raise HTTPException(501, "Google Drive 集成需要配置 OAuth 凭证。请联系管理员。")


@app.post("/api/drive/import")
async def import_from_drive(
    file_id: str = Form(...),
    file_name: str = Form(...),
    target_hint: str = Form(""),
) -> dict[str, Any]:
    """
    从 Google Drive 导入文件
    
    需要：
    1. 用户已完成 Google OAuth 授权
    2. 有有效的 access_token
    """
    # 预留：实际实现需要：
    # 1. 验证用户 access_token
    # 2. 调用 Google Drive API 下载文件
    # 3. 保存并解析文件
    raise HTTPException(501, "Google Drive 导入功能开发中。请直接上传文件。")


@app.get("/api/data/list")
def list_datasets() -> dict[str, Any]:
    """列出所有数据集"""
    datasets = data_manager.list_datasets()
    return {
        "datasets": [d.to_dict() for d in datasets],
    }


@app.get("/api/data/{dataset_id}")
def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    """获取数据集信息"""
    try:
        spec = data_manager.get_dataset(dataset_id)
        return spec.to_dict()
    except ValueError as e:
        raise HTTPException(404, str(e))


# ---- 意图编译 API ----

@app.post("/api/intent/clarify", response_model=ClarifyResponse)
def clarify_intent(req: ClarifyRequest) -> ClarifyResponse:
    """
    意图澄清 API
    
    将自然语言需求编译为 ObjectiveSpec
    """
    is_ambiguous, reasons, follow_ups = detect_ambiguity(req)
    
    # 即使模糊，也尝试编译
    spec = compile_objective(
        user_goal=req.user_goal,
        must_keep=req.must_keep,
        can_change=req.can_change,
        worst_errors=req.worst_errors,
        priority=req.priority,
    )
    
    compiled = {
        "objective": req.user_goal,
        "must_keep": req.must_keep,
        "can_change": req.can_change,
        "worst_errors": req.worst_errors,
        "priority": req.priority,
        "recommended_mode": "conservative" if is_ambiguous else "standard",
    }
    
    response = ClarifyResponse(
        is_ambiguous=is_ambiguous,
        reasons=reasons,
        follow_up_questions=follow_ups,
        compiled_intent=compiled,
        objective_spec=spec.to_dict(),
    )
    
    return response


@app.post("/api/intent/compile")
def compile_intent(req: ClarifyRequest) -> dict[str, Any]:
    """
    编译意图为 ObjectiveSpec
    
    这是 chat2objective 的核心实现，使用 LLM 进行意图解析
    """
    # 如果有数据集，获取 schema 辅助理解
    data_schema = None
    if req.dataset_id:
        try:
            ds = data_manager.get_dataset(req.dataset_id)
            data_schema = {
                "columns": [
                    {
                        "name": c.name,
                        "type": c.column_type.value,
                        "sample": c.sample_values[:3] if c.sample_values else [],
                    }
                    for c in ds.columns
                ],
                "target_column": ds.target_column,
                "n_rows": ds.n_rows,
            }
        except:
            pass
    
    spec = compile_objective(
        user_goal=req.user_goal,
        must_keep=req.must_keep,
        can_change=req.can_change,
        worst_errors=req.worst_errors,
        priority=req.priority,
        data_schema=data_schema,
    )
    
    return {
        "success": True,
        "objective_spec": spec.to_dict(),
        "interpretation": {
            "task_type": spec.task_family.value,
            "target_column": spec.label.target_column,
            "primary_metric": spec.primary_metric.value,
            "validation_strategy": spec.validation_strategy,
            "recommended_models": spec.recommended_models,
            "reasoning": spec.interpretation,
        }
    }


# ---- 训练 API ----

@app.post("/api/training/start")
def start_training(req: StartTrainingRequest) -> dict[str, Any]:
    """
    启动训练任务
    
    从 ObjectiveSpec 到模型权重 (chat2model)
    """
    # 如果有数据集，获取 schema 辅助理解
    data_schema = None
    if req.dataset_id:
        try:
            ds = data_manager.get_dataset(req.dataset_id)
            data_schema = {
                "columns": [
                    {"name": c.name, "type": c.column_type.value}
                    for c in ds.columns
                ],
                "target_column": ds.target_column,
                "n_rows": ds.n_rows,
            }
        except:
            pass
    
    # 编译意图（使用LLM）
    spec = compile_objective(
        user_goal=req.objective,
        must_keep=req.must_keep,
        can_change=req.can_change,
        worst_errors=req.worst_errors,
        priority=req.priority,
        data_schema=data_schema,
    )
    
    # 使用用户指定的目标列
    if req.target_column:
        spec.label.target_column = req.target_column
    
    # 创建任务
    cfg = {
        "objective": req.objective,
        "must_keep": req.must_keep,
        "can_change": req.can_change,
        "worst_errors": req.worst_errors,
        "priority": req.priority,
        "mode": req.mode,
        "sample_notes": req.sample_notes,
        "dataset_id": req.dataset_id,
        "target_column": req.target_column,
        "max_training_time": req.max_training_time,
        "max_trials": req.max_trials,
    }
    
    job = job_manager.create_job(cfg, spec)
    
    return {
        "success": True,
        "job_id": job.job_id,
        "status": job.status,
        "status_url": f"/api/training/{job.job_id}",
        "objective_spec": spec.to_dict(),
    }


@app.get("/api/training/{job_id}")
def get_training_status(job_id: str) -> JobStatusResponse:
    """获取训练状态"""
    job = job_manager.get(job_id)
    
    with job.lock:
        elapsed = time.time() - job.start_time
        if job.end_time:
            elapsed = job.end_time - job.start_time
        
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=round(job.progress, 1),
            current_step=job.current_step,
            message=job.message,
            train_metrics=job.train_metrics,
            val_metrics=job.val_metrics,
            best_score=job.training_result.best_metric_score if job.training_result else None,
            elapsed_time=round(elapsed, 1),
        )


@app.post("/api/training/{job_id}/pause")
def pause_training(job_id: str) -> dict[str, Any]:
    """暂停训练"""
    job = job_manager.pause_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "message": "训练已暂停",
    }


@app.post("/api/training/{job_id}/resume")
def resume_training(job_id: str) -> dict[str, Any]:
    """恢复训练"""
    job = job_manager.resume_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "message": "训练已恢复",
    }


@app.post("/api/training/{job_id}/stop")
def stop_training(job_id: str) -> dict[str, Any]:
    """停止训练"""
    job = job_manager.stop_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "message": "训练已停止",
    }


# ---- Checkpoint / 模型下载 API ----

@app.get("/api/training/{job_id}/checkpoints")
def list_checkpoints(job_id: str) -> dict[str, Any]:
    """列出所有 checkpoint"""
    job = job_manager.get(job_id)
    
    checkpoints = []
    
    # 查找模型文件
    model_path = CHECKPOINT_DIR / f"{job_id}_model.pkl"
    if model_path.exists():
        checkpoints.append({
            "type": "model",
            "file": f"{job_id}_model.pkl",
            "size": model_path.stat().st_size,
            "created": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        })
    
    # 查找预处理器
    preprocessor_path = CHECKPOINT_DIR / f"{job_id}_preprocessor.pkl"
    if preprocessor_path.exists():
        checkpoints.append({
            "type": "preprocessor",
            "file": f"{job_id}_preprocessor.pkl",
            "size": preprocessor_path.stat().st_size,
            "created": datetime.fromtimestamp(preprocessor_path.stat().st_mtime).isoformat(),
        })
    
    # 查找结果元数据
    result_path = CHECKPOINT_DIR / f"{job_id}_result.json"
    if result_path.exists():
        checkpoints.append({
            "type": "metadata",
            "file": f"{job_id}_result.json",
            "size": result_path.stat().st_size,
            "created": datetime.fromtimestamp(result_path.stat().st_mtime).isoformat(),
        })
    
    return {
        "job_id": job_id,
        "status": job.status,
        "checkpoints": checkpoints,
    }


@app.get("/api/training/{job_id}/download/{file_type}")
def download_artifact(job_id: str, file_type: str) -> FileResponse:
    """
    下载训练产物
    
    file_type: model, preprocessor, metadata
    """
    job = job_manager.get(job_id)
    
    filename_map = {
        "model": f"{job_id}_model.pkl",
        "preprocessor": f"{job_id}_preprocessor.pkl",
        "metadata": f"{job_id}_result.json",
    }
    
    if file_type not in filename_map:
        raise HTTPException(400, f"不支持的文件类型: {file_type}")
    
    filename = filename_map[file_type]
    path = CHECKPOINT_DIR / filename
    
    if not path.exists():
        raise HTTPException(404, f"文件不存在: {filename}")
    
    # 根据类型设置 media_type
    media_types = {
        "model": "application/octet-stream",
        "preprocessor": "application/octet-stream",
        "metadata": "application/json",
    }
    
    return FileResponse(
        path=path,
        filename=filename,
        media_type=media_types[file_type],
    )


@app.get("/api/training/{job_id}/result")
def get_training_result(job_id: str) -> dict[str, Any]:
    """获取完整训练结果"""
    job = job_manager.get(job_id)
    
    if not job.training_result:
        raise HTTPException(404, "训练结果尚未生成")
    
    result = job.training_result
    
    return {
        "job_id": job_id,
        "status": result.status,
        "best_model_name": result.best_model_name,
        "best_metric_score": result.best_metric_score,
        "best_hyperparameters": result.best_hyperparameters,
        "train_metrics": result.train_metrics,
        "val_metrics": result.val_metrics,
        "feature_importance": result.feature_importance,
        "training_duration": result.training_duration,
        "n_trials": len(result.trials),
        "trials": [
            {
                "trial_id": t.trial_id,
                "model_name": t.model_name,
                "metric_score": t.metric_score,
                "hyperparameters": t.hyperparameters,
            }
            for t in result.trials[:10]  # 只返回前10个
        ],
        "downloads": {
            "model": f"/api/training/{job_id}/download/model",
            "preprocessor": f"/api/training/{job_id}/download/preprocessor",
            "metadata": f"/api/training/{job_id}/download/metadata",
        }
    }


# ---- 预测 API ----

@app.post("/api/predict/{job_id}")
async def predict(job_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """使用训练好的模型进行预测"""
    job = job_manager.get(job_id)
    
    if job.status != "completed":
        raise HTTPException(400, "模型训练尚未完成")
    
    if not job.training_result or not job.training_result.final_model_path:
        raise HTTPException(404, "模型文件不存在")
    
    # 读取预测数据
    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"无法解析 CSV: {str(e)}")
    
    # 加载模型和预处理器
    from backend.trainer import ModelLoader
    
    loader = ModelLoader(
        job.training_result.final_model_path,
        job.training_result.preprocessor_path,
    )
    
    # 预测
    try:
        predictions = loader.predict(df)
        
        # 尝试获取概率（分类任务）
        probabilities = None
        try:
            probabilities = loader.predict_proba(df)
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                # 二分类，返回正类概率
                probabilities = probabilities[:, 1].tolist()
            else:
                probabilities = probabilities.tolist()
        except:
            pass
        
        return {
            "success": True,
            "n_predictions": len(predictions),
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "model_used": job.training_result.best_model_name,
        }
        
    except Exception as e:
        raise HTTPException(500, f"预测失败: {str(e)}")
