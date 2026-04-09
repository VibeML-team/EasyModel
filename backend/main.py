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
import mimetypes
import os
import pickle
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 导入新模块
from backend.compiler import compile_objective, ObjectiveSpec, detect_ambiguity as _compiler_detect_ambiguity
from backend.data_manager import data_manager, DataSpec, DataManager, DATA_DIR
from backend.trainer import TrainingResult, CHECKPOINT_DIR
from backend.training_router import build_training_plan, execute_training_plan

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
    user_goal: str = Field(min_length=1)
    must_keep: list[str] = Field(default_factory=list)
    can_change: list[str] = Field(default_factory=list)
    worst_errors: list[str] = Field(default_factory=list)
    priority: Priority = "quality"
    dataset_id: str | None = None


class ClarifyResponse(BaseModel):
    is_ambiguous: bool
    reasons: list[str]
    follow_up_questions: list[str]
    compiled_intent: dict[str, Any] = {}
    objective_spec: dict[str, Any] | None = None
    # 新增：个性化方案（双层结构）
    personalized_plan: dict[str, Any] | None = None


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
    error_message: str | None = None
    recent_logs: list[str] = Field(default_factory=list)
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
    recent_logs: list[str] = field(default_factory=list)
    
    # 训练配置
    config: dict[str, Any] = field(default_factory=dict)
    objective_spec: ObjectiveSpec | None = None
    dataset_id: str | None = None
    
    # 训练指标
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    
    # 训练结果
    training_result: TrainingResult | None = None
    training_plan: dict[str, Any] = field(default_factory=dict)
    
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

            data_spec = data_manager.get_dataset(job.dataset_id)

            # 2. 更新目标列（如果用户指定）
            if job.config.get("target_column"):
                if job.objective_spec:
                    job.objective_spec.label.target_column = job.config["target_column"]

            objective_spec = job.objective_spec or ObjectiveSpec()
            if not objective_spec.raw_intent:
                objective_spec.raw_intent = job.config.get("objective", "")

            plan = build_training_plan(
                objective_spec=objective_spec,
                data_spec=data_spec,
                config=job.config,
            )

            with job.lock:
                job.training_plan = plan.to_dict()
                job.progress = 5.0
                job.current_step = "planning"
                job.message = f"执行规划已生成: {plan.executor} ({plan.modality})"

            def progress_callback(progress: dict):
                with job.lock:
                    if job.stop_requested:
                        raise InterruptedError("训练被中断")
                    
                    step = progress.get("step", "")
                    if step == "planning":
                        job.progress = 8.0
                        job.current_step = "planning"
                        job.message = progress.get("message", "生成执行规划中...")
                    elif step == "loading_data":
                        job.progress = 10.0
                        job.current_step = "loading_data"
                        job.message = progress.get("message", "按规划加载数据中...")
                    elif step == "preprocessing":
                        job.progress = 15.0
                        job.current_step = "preprocessing"
                        job.message = progress.get("message", "预处理中...")
                    elif step == "search":
                        trial = progress.get("trial", 0)
                        best = progress.get("best_score", 0)
                        job.progress = 15.0 + min(trial * 2, 70.0)
                        job.current_step = "training"
                        job.message = f"超参数搜索中... Trial {trial}, 最佳得分: {best:.4f}"
                        job.train_metrics["best_cv_score"] = best
                    elif step == "final_training":
                        job.progress = 90.0
                        job.current_step = "final_training"
                        job.message = "训练最终模型..."
                    elif step == "agentic_log":
                        stage = progress.get("agent_stage") or "agentic"
                        stage_progress = {
                            "generating": 18.0,
                            "validating": 38.0,
                            "optimizing": 65.0,
                            "completed": 95.0,
                            "failed": job.progress,
                        }
                        job.progress = max(job.progress, stage_progress.get(stage, job.progress))
                        job.current_step = stage
                        job.message = progress.get("message", "Agent 正在执行...")
                        log_entry = progress.get("log_entry") or job.message
                        if not job.recent_logs or job.recent_logs[-1] != log_entry:
                            job.recent_logs.append(log_entry)
                            job.recent_logs = job.recent_logs[-8:]

            with job.lock:
                job.current_step = "training"
                job.message = f"按规划开始执行: {plan.strategy}"

            result = execute_training_plan(
                job_id=job.job_id,
                plan=plan,
                objective_spec=objective_spec,
                progress_callback=progress_callback,
            )
            
            with job.lock:
                job.training_result = result
                job.status = result.status
                job.progress = 100.0
                job.end_time = time.time()
                
                if result.status == "completed":
                    score_text = (
                        f"{result.best_metric_score:.4f}"
                        if result.best_metric_score is not None
                        else "N/A"
                    )
                    job.message = f"训练完成！最佳模型: {result.best_model_name}, 验证得分: {score_text}"
                    job.train_metrics.update(result.train_metrics)
                    job.val_metrics.update(result.val_metrics)
                else:
                    job.message = result.error_message or "训练失败"
            
        except InterruptedError:
            with job.lock:
                job.status = "stopped"
                job.message = "训练已停止"
                job.end_time = time.time()
        except Exception as e:
            with job.lock:
                job.status = "failed"
                job.message = str(e)
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


def _build_training_result_payload(job: JobState) -> dict[str, Any]:
    """构造训练结果元数据，供 API 返回和下载复用。"""
    if not job.training_result:
        raise HTTPException(404, "训练结果尚未生成")

    result = job.training_result
    return {
        "job_id": job.job_id,
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
            for t in result.trials[:10]
        ],
        "downloads": {
            "model": f"/api/training/{job.job_id}/download/model",
            "preprocessor": f"/api/training/{job.job_id}/download/preprocessor",
            "metadata": f"/api/training/{job.job_id}/download/metadata",
        },
    }


def _resolve_artifact(job: JobState, file_type: str) -> tuple[Path | None, str | None]:
    """按真实训练结果路径解析下载产物，而不是依赖固定文件名。"""
    if file_type == "model":
        candidates = [
            (CHECKPOINT_DIR / f"{job.job_id}_model.pkl", f"{job.job_id}_model.pkl"),
        ]
        if job.training_result and job.training_result.final_model_path:
            model_path = Path(job.training_result.final_model_path)
            candidates.insert(0, (model_path, model_path.name))
    elif file_type == "preprocessor":
        candidates = [
            (CHECKPOINT_DIR / f"{job.job_id}_preprocessor.pkl", f"{job.job_id}_preprocessor.pkl"),
        ]
        if job.training_result and job.training_result.preprocessor_path:
            preprocessor_path = Path(job.training_result.preprocessor_path)
            candidates.insert(0, (preprocessor_path, preprocessor_path.name))
    elif file_type == "metadata":
        metadata_path = CHECKPOINT_DIR / f"{job.job_id}_result.json"
        if not metadata_path.exists() and job.training_result:
            metadata_path.write_text(
                json.dumps(_build_training_result_payload(job), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        candidates = [(metadata_path, metadata_path.name)]
    else:
        raise HTTPException(400, f"不支持的文件类型: {file_type}")

    for path, filename in candidates:
        if path and path.exists() and path.is_file():
            return path, filename
    return None, None


# ============ FastAPI 应用 ============

app = FastAPI(
    title="VibeML Agent",
    version="1.0.0",
    description="从自然语言需求到可部署模型权重的全自动 ML 平台",
)

# 大文件上传支持
from starlette.formparsers import MultiPartParser
MultiPartParser.spool_max_size = 1024 * 1024 * 2048  # 2 GB spool to disk threshold
MultiPartParser.max_part_size = 1024 * 1024 * 2048   # 2 GB max part size


async def _stream_upload_to_disk(file: UploadFile, dest: Path) -> int:
    """流式写入磁盘，不把文件全部读入内存"""
    total = 0
    with open(dest, 'wb') as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total

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


def detect_ambiguity(req):
    """检测需求模糊性 — 委托给 compiler.py（LLM 驱动，无硬编码规则）"""
    return _compiler_detect_ambiguity(req)


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


@app.get("/chat")
def chat() -> FileResponse:
    """Chat 页面"""
    chat_file = FRONTEND_DIR / "chat.html"
    if chat_file.exists():
        return FileResponse(chat_file)
    return JSONResponse({"error": "Chat page not found"}, status_code=404)


@app.get("/chat.html")
def chat_html() -> FileResponse:
    """Chat 页面 (.html 后缀)"""
    return chat()


@app.get("/dashboard")
def dashboard() -> FileResponse:
    """Dashboard 页面"""
    dashboard_file = FRONTEND_DIR / "dashboard.html"
    if dashboard_file.exists():
        return FileResponse(dashboard_file)
    return JSONResponse({"error": "Dashboard page not found"}, status_code=404)


@app.get("/dashboard.html")
def dashboard_html() -> FileResponse:
    """Dashboard 页面 (.html 后缀)"""
    return dashboard()


@app.get("/api/health")
async def health() -> dict[str, Any]:
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
        "version": "1.1.0-fix-ambiguity",
        "deploy_tag": "20260408-v3",
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
    上传任意格式的数据文件。
    流式写入磁盘，支持大文件（几百MB ~ 几GB）。
    """
    import uuid as _uuid
    dataset_id = f"ds_{_uuid.uuid4().hex[:8]}"
    
    # 流式写入磁盘（不读入内存）
    upload_dir = DATA_DIR / dataset_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    file_size = await _stream_upload_to_disk(file, dest)
    
    # 从磁盘读取交给 data_manager
    content = dest.read_bytes() if file_size < 500 * 1024 * 1024 else None  # >500MB 不读入内存
    
    try:
        if content is not None:
            spec = data_manager.upload_file(
                content=content,
                filename=file.filename,
                dataset_id=dataset_id,
                target_hint=target_hint if target_hint else None,
            )
        else:
            # 大文件：直接让 data_manager 从磁盘处理
            spec = data_manager.upload_from_disk(
                file_path=dest,
                filename=file.filename,
                dataset_id=dataset_id,
            target_hint=target_hint if target_hint else None,
        )
        return {
            "success": True,
            "dataset_id": spec.dataset_id,
            "filename": spec.filename,
            "n_rows": spec.n_rows,
            "n_cols": spec.n_cols,
            "data_type": spec.data_type,
            "target_column": spec.target_column,
            "feature_columns": spec.feature_columns,
            "file_scan": {
                "total_files": spec.file_scan.get("total_files", 0),
                "total_size_human": spec.file_scan.get("total_size_human", ""),
                "extensions": spec.file_scan.get("extensions", {}),
                "has_images": spec.file_scan.get("has_images", False),
                "has_tabular": spec.file_scan.get("has_tabular", False),
            },
            "columns": [c.to_dict() for c in spec.columns],
        }
    except Exception as e:
        raise HTTPException(400, f"数据处理失败: {str(e)}")


from fastapi.responses import StreamingResponse

# ---- 分块上传 API（绕过 proxy body size 限制）----

_chunk_uploads: dict[str, dict] = {}  # upload_id → {dir, filename, received_chunks, total_chunks}

class ChunkInitRequest(BaseModel):
    filename: str
    total_size: int
    total_chunks: int

@app.post("/api/data/upload-init")
async def upload_init(req: ChunkInitRequest) -> dict[str, Any]:
    """初始化分块上传"""
    import uuid as _uuid
    upload_id = _uuid.uuid4().hex[:12]
    dataset_id = f"ds_{_uuid.uuid4().hex[:8]}"
    
    upload_dir = DATA_DIR / dataset_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    _chunk_uploads[upload_id] = {
        "dataset_id": dataset_id,
        "dir": str(upload_dir),
        "filename": req.filename,
        "total_size": req.total_size,
        "total_chunks": req.total_chunks,
        "received": set(),
    }
    
    return {"upload_id": upload_id, "dataset_id": dataset_id}


@app.post("/api/data/upload-chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
) -> dict[str, Any]:
    """上传单个分块"""
    if upload_id not in _chunk_uploads:
        raise HTTPException(404, "Upload session not found")
    
    info = _chunk_uploads[upload_id]
    chunk_dir = Path(info["dir"]) / "_chunks"
    chunk_dir.mkdir(exist_ok=True)
    
    # 写入分块
    chunk_path = chunk_dir / f"chunk_{chunk_index:05d}"
    content = await chunk.read()
    with open(chunk_path, 'wb') as f:
        f.write(content)
    
    info["received"].add(chunk_index)
    
    return {
        "received": chunk_index,
        "total_received": len(info["received"]),
        "total_chunks": info["total_chunks"],
        "complete": len(info["received"]) >= info["total_chunks"],
    }


@app.post("/api/data/upload-complete")
async def upload_complete(
    upload_id: str = Form(""),
    target_hint: str = Form(""),
    user_goal: str = Form(""),
) -> dict[str, Any]:
    """分块上传完成，拼接文件并触发分析"""
    if upload_id not in _chunk_uploads:
        raise HTTPException(404, "Upload session not found")
    
    info = _chunk_uploads[upload_id]
    
    if len(info["received"]) < info["total_chunks"]:
        raise HTTPException(400, f"Missing chunks: received {len(info['received'])}/{info['total_chunks']}")
    
    # 拼接文件
    chunk_dir = Path(info["dir"]) / "_chunks"
    dest = Path(info["dir"]) / info["filename"]
    
    with open(dest, 'wb') as out:
        for i in range(info["total_chunks"]):
            chunk_path = chunk_dir / f"chunk_{i:05d}"
            with open(chunk_path, 'rb') as inp:
                while True:
                    block = inp.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
    
    # 清理分块
    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)
    
    file_size = dest.stat().st_size
    
    # 用 upload_from_disk 处理
    try:
        spec = data_manager.upload_from_disk(
            file_path=dest,
            filename=info["filename"],
            dataset_id=info["dataset_id"],
            target_hint=target_hint if target_hint else None,
        )
        
        del _chunk_uploads[upload_id]
        
        return {
            "success": True,
            "dataset_id": spec.dataset_id,
            "filename": spec.filename,
            "file_size": file_size,
            "n_rows": spec.n_rows,
            "data_type": spec.data_type,
            "file_scan": {
                "total_files": spec.file_scan.get("total_files", 0),
                "total_size_human": spec.file_scan.get("total_size_human", ""),
                "extensions": spec.file_scan.get("extensions", {}),
            },
        }
    except Exception as e:
        raise HTTPException(400, f"数据处理失败: {str(e)}")

@app.post("/api/data/upload-stream")
async def upload_and_analyze_stream(
    file: UploadFile = File(...),
    target_hint: str = Form(""),
    user_goal: str = Form(""),
):
    """
    上传 + 扫描 + AI Agent 自主探索，全流程 SSE 流式推送。
    流式写入磁盘，支持大文件。
    """
    import uuid as _uuid
    filename = file.filename
    dataset_id = f"ds_{_uuid.uuid4().hex[:8]}"
    
    # 流式写入磁盘
    upload_dir = DATA_DIR / dataset_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / filename
    file_size = await _stream_upload_to_disk(file, dest)
    
    async def event_stream():
        progress_events = []
        
        def on_progress(step: str, message: str):
            progress_events.append({"step": step, "message": message, "done": False})
        
        on_progress("save", f"📥 文件已接收: {filename} ({file_size / 1024 / 1024:.1f} MB)")
        
        # 从磁盘处理
        try:
            content = dest.read_bytes() if file_size < 500 * 1024 * 1024 else None
            if content is not None:
                spec = data_manager.upload_file(
                    content=content,
                    filename=filename,
                    dataset_id=dataset_id,
                    target_hint=target_hint if target_hint else None,
                    progress_callback=on_progress,
                )
            else:
                spec = data_manager.upload_from_disk(
                    file_path=dest,
                    filename=filename,
                    dataset_id=dataset_id,
                    target_hint=target_hint if target_hint else None,
                    progress_callback=on_progress,
                )
        except Exception as e:
            yield f"data: {json.dumps({'step': 'error', 'message': f'❌ 处理失败: {str(e)}', 'done': True}, ensure_ascii=False)}\n\n"
            return
        
        # 推送所有进度事件
        for evt in progress_events:
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        
        # === ReAct Agent 自主探索数据 ===
        exploration = None
        from backend.compiler import _use_llm_compiler
        if _use_llm_compiler():
            try:
                from backend.llm_client import get_llm_client
                from backend.data_exploration_agent import run_exploration_agent
                client = get_llm_client()
                
                # 确保 httpx 超时足够
                import httpx
                if hasattr(client, 'client'):
                    client.client = httpx.Client(
                        base_url=client.config.base_url,
                        headers={"Authorization": f"Bearer {client.config.api_key}", "Content-Type": "application/json"},
                        timeout=600.0,
                    )
                
                # 确定 Agent 工作目录（解压后的目录）
                agent_cwd = str(Path(spec.storage_path) / "extracted") if spec.storage_path else str(DATA_DIR / spec.dataset_id)
                if not Path(agent_cwd).exists():
                    agent_cwd = spec.storage_path or str(DATA_DIR / spec.dataset_id)
                
                file_size_human = spec.file_scan.get("total_size_human", "unknown")
                
                # 运行 ReAct Agent，实时推送每一步
                for event in run_exploration_agent(
                    dataset_dir=agent_cwd,
                    filename=filename,
                    file_size_human=file_size_human,
                    user_goal=user_goal if user_goal else None,
                    llm_client=client,
                ):
                    etype = event["type"]
                    content = event["content"]
                    
                    if etype == "thought":
                        yield f"data: {json.dumps({'step': 'think', 'message': f'💭 {content}', 'done': False}, ensure_ascii=False)}\n\n"
                    elif etype == "action":
                        yield f"data: {json.dumps({'step': 'action', 'message': f'⚡ 执行: {content}', 'done': False}, ensure_ascii=False)}\n\n"
                    elif etype == "observation":
                        # 截断过长的输出
                        obs_preview = content[:300] + ('...' if len(content) > 300 else '')
                        yield f"data: {json.dumps({'step': 'observe', 'message': f'👁 {obs_preview}', 'done': False}, ensure_ascii=False)}\n\n"
                    elif etype == "insight":
                        exploration = content
                        yield f"data: {json.dumps({'step': 'think', 'message': '✅ 数据探索完成', 'done': False}, ensure_ascii=False)}\n\n"
                    elif etype == "error":
                        yield f"data: {json.dumps({'step': 'think', 'message': f'⚠️ {content}', 'done': False}, ensure_ascii=False)}\n\n"
                
                # 保存探索结果
                if exploration:
                    spec.exploration = exploration
                    # 如果 Agent 识别了数据类型，更新 spec
                    if exploration.get("data_type"):
                        spec.data_type = exploration["data_type"]
                    meta_path = DATA_DIR / f"{spec.dataset_id}_meta.json"
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
                    data_manager.datasets[spec.dataset_id] = spec
                
            except Exception as e:
                yield f"data: {json.dumps({'step': 'think', 'message': f'⚠️ Agent 遇到问题: {str(e)[:150]}', 'done': False}, ensure_ascii=False)}\n\n"
        
        # 最终结果
        result = {
            "step": "result",
            "done": True,
            "data": {
                "dataset_id": spec.dataset_id,
                "filename": spec.filename,
                "n_rows": spec.n_rows,
                "n_cols": spec.n_cols,
                "data_type": spec.data_type,
                "target_column": spec.target_column,
                "file_scan": {
                    "total_files": spec.file_scan.get("total_files", 0),
                    "total_size_human": spec.file_scan.get("total_size_human", ""),
                    "extensions": spec.file_scan.get("extensions", {}),
                    "directory_tree": spec.file_scan.get("directory_tree", ""),
                },
                "exploration": exploration,
            }
        }
        yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ---- Google Drive / 链接导入 API ----

class LinkImportRequest(BaseModel):
    """通过链接导入数据"""
    url: str = Field(min_length=5, description="Google Drive 分享链接、Google Sheets 链接或任何公开的 CSV/Excel 下载链接")
    target_hint: str = ""


def _parse_google_drive_url(url: str) -> tuple[str | None, str]:
    """
    解析 Google Drive/Sheets URL，返回 (direct_download_url, filename)
    
    支持格式：
    - https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing
    - https://drive.google.com/open?id={FILE_ID}
    - https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit#gid=0
    - 直接下载链接
    """
    import re
    
    # Google Drive file link
    match = re.search(r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}", f"gdrive_{file_id}.csv"
    
    # Google Drive open link
    match = re.search(r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}", f"gdrive_{file_id}.csv"
    
    # Google Sheets link → export as CSV
    match = re.search(r'docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)', url)
    if match:
        sheet_id = match.group(1)
        # 提取 gid（子表ID），默认 0
        gid_match = re.search(r'gid=(\d+)', url)
        gid = gid_match.group(1) if gid_match else "0"
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}", f"gsheet_{sheet_id}.csv"
    
    # 普通链接（直接下载）
    return url, url.split('/')[-1].split('?')[0] or "downloaded_data.csv"


@app.post("/api/data/import-link")
async def import_from_link(req: LinkImportRequest) -> dict[str, Any]:
    """
    通过链接导入数据
    
    支持：
    - Google Drive 分享链接（文件需设为"知道链接的人可查看"）
    - Google Sheets 链接（自动导出为 CSV）
    - 任何公开的 CSV/Excel 下载链接
    """
    import httpx
    
    download_url, filename = _parse_google_drive_url(req.url)
    
    if not download_url:
        raise HTTPException(400, "无法解析链接。请确保是 Google Drive 分享链接、Google Sheets 链接或直接下载链接。")
    
    # 下载文件
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(download_url)
            response.raise_for_status()
            content = response.content
    except httpx.TimeoutException:
        raise HTTPException(408, "下载超时。文件可能过大或网络不稳定，请稍后重试。")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(404, "文件不存在或无访问权限。请检查链接是否正确，且文件已设为公开分享。")
        raise HTTPException(502, f"下载失败 (HTTP {e.response.status_code})。请检查链接权限。")
    except Exception as e:
        raise HTTPException(502, f"下载失败: {str(e)}")
    
    if len(content) < 10:
        raise HTTPException(400, "下载的文件为空。请检查链接权限，确保文件已设为'知道链接的人可查看'。")
    
    # 检测文件格式（如果 filename 没有扩展名，尝试从 content 推断）
    if not any(filename.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.xls', '.zip', '.7z']):
        # 尝试检测内容格式
        content_type = response.headers.get('content-type', '')
        if 'spreadsheet' in content_type or 'excel' in content_type:
            filename = filename + '.xlsx'
        elif 'zip' in content_type:
            filename = filename + '.zip'
        else:
            filename = filename + '.csv'  # 默认当 CSV 处理
    
    # 交给 data_manager 统一处理
    try:
        spec = data_manager.upload_file(
            content=content,
            filename=filename,
            target_hint=req.target_hint if req.target_hint else None,
        )
        return {
            "success": True,
            "source": "link_import",
            "original_url": req.url,
            "dataset_id": spec.dataset_id,
            "filename": spec.filename,
            "n_rows": spec.n_rows,
            "n_cols": spec.n_cols,
            "target_column": spec.target_column,
            "feature_columns": spec.feature_columns,
            "columns": [c.to_dict() for c in spec.columns],
        }
    except Exception as e:
        raise HTTPException(400, f"数据解析失败: {str(e)}")


class AgentFindDataRequest(BaseModel):
    user_goal: str = Field(min_length=3)


# Agent 任务状态存储（后台线程 + 轮询）
_agent_tasks: dict[str, dict] = {}


def _try_prepare_known_dataset(task: dict[str, Any], dataset_id: str, dataset_dir: str, user_goal: str) -> dict[str, Any] | None:
    """对少数知名公开数据集走确定性下载路径，避免让 Agent 临场拼命令。"""
    goal = (user_goal or "").lower()
    if "mnist" not in goal:
        return None

    task["log"].append({"type": "thought", "content": "识别到用户明确需要 MNIST，优先走后端内置下载器，而不是让 Agent 现场拼 shell 命令。"})
    task["log"].append({"type": "action", "content": "download_builtin_dataset('mnist')"})

    from collections import defaultdict

    import numpy as np
    from PIL import Image
    from sklearn.datasets import fetch_openml

    root = Path(dataset_dir)
    train_root = root / "train"
    test_root = root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    except TypeError:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data
    y = mnist.target.astype(str)

    train_limits = defaultdict(int)
    test_limits = defaultdict(int)
    train_cap = 200
    test_cap = 50

    train_saved = 0
    test_saved = 0

    for idx, (pixels, label) in enumerate(zip(X, y)):
        arr = pixels.reshape(28, 28).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        if train_saved < 2000 and train_limits[label] < train_cap:
            out_dir = train_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / f"{label}_{train_limits[label]:04d}.png")
            train_limits[label] += 1
            train_saved += 1
            continue

        if test_saved < 500 and test_limits[label] < test_cap:
            out_dir = test_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            img.save(out_dir / f"{label}_{test_limits[label]:04d}.png")
            test_limits[label] += 1
            test_saved += 1

        if train_saved >= 2000 and test_saved >= 500:
            break

    task["log"].append({
        "type": "observation",
        "content": f"MNIST 已下载并整理为图像目录。train={train_saved} 张, test={test_saved} 张, classes={sorted(set(y.tolist()))}",
    })

    return {
        "data_type": "image_classification",
        "format_details": "directory_split/train_label_png",
        "business_summary": "MNIST 手写数字图像分类数据集，已整理为 train/test 按类别分目录的 PNG 文件。",
        "data_understanding": {
            "summary": "28x28 灰度手写数字图像，共 10 个类别。",
            "organization": "train/<label>/*.png, test/<label>/*.png",
            "key_files": ["train/0-9/*.png", "test/0-9/*.png"],
        },
        "statistics": {
            "total_samples": train_saved + test_saved,
            "classes": [str(i) for i in range(10)],
            "class_distribution": {str(i): train_limits[str(i)] + test_limits[str(i)] for i in range(10)},
            "splits": {"train": train_saved, "test": test_saved},
        },
        "quality_issues": [],
        "training_implications": [
            "这是标准图像分类任务，适合使用 CNN 或轻量视觉模型。",
            "输入是 28x28 灰度图像，需要在数据管道里显式处理单通道。",
            "当前为了快速可用只落盘了一个可训练子集，而不是完整 7 万张样本。",
        ],
        "suggested_next_steps": [
            "按图像分类任务启动训练。",
            "如果需要更高精度，可扩展为完整 MNIST 全量落盘。",
        ],
    }


def _run_agent_background(task_id: str, dataset_id: str, dataset_dir: str, user_goal: str):
    """后台线程运行 ReAct Agent"""
    task = _agent_tasks[task_id]
    
    try:
        exploration = _try_prepare_known_dataset(task, dataset_id, dataset_dir, user_goal)
        if exploration is None:
            from backend.llm_client import get_llm_client
            from backend.data_exploration_agent import run_exploration_agent
            client = get_llm_client()
            
            import httpx
            if hasattr(client, 'client'):
                client.client = httpx.Client(
                    base_url=client.config.base_url,
                    headers={"Authorization": f"Bearer {client.config.api_key}", "Content-Type": "application/json"},
                    timeout=600.0,
                )
            
            for event in run_exploration_agent(
                dataset_dir=dataset_dir,
                filename="agent_requested_data",
                file_size_human="0 B (待下载)",
                user_goal=user_goal,
                llm_client=client,
            ):
                task["log"].append(event)
                if event["type"] == "insight":
                    exploration = event["content"]
        
        # 扫描结果
        from backend.data_manager import FileScanner
        file_scan = {}
        try:
            file_scan = FileScanner.scan_directory(dataset_dir)
        except Exception:
            pass
        
        # 注册数据集
        spec = DataSpec(
            dataset_id=dataset_id,
            filename="agent_acquired",
            n_rows=file_scan.get("total_files", 0),
            data_type=exploration.get("data_type", "unknown") if exploration else "unknown",
            file_scan=file_scan,
            exploration=exploration or {},
            storage_path=dataset_dir,
        )
        meta_path = DATA_DIR / f"{dataset_id}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
        data_manager.datasets[dataset_id] = spec
        
        task["status"] = "completed"
        task["result"] = {
            "dataset_id": dataset_id,
            "data_type": spec.data_type,
            "exploration": exploration,
            "file_scan": {"total_files": file_scan.get("total_files", 0), "total_size_human": file_scan.get("total_size_human", "")},
        }
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)[:500]
        task["log"].append({"type": "error", "content": str(e)[:300]})


@app.post("/api/data/agent-find")
def agent_find_data(req: AgentFindDataRequest) -> dict[str, Any]:
    """
    启动 ReAct Agent 后台获取数据。
    立即返回 task_id，前端轮询 /agent-status 获取进度。
    """
    from backend.compiler import _use_llm_compiler
    if not _use_llm_compiler():
        raise HTTPException(501, "需要配置 LLM 才能使用 Agent")
    
    import uuid as _uuid
    task_id = _uuid.uuid4().hex[:12]
    dataset_id = f"ds_{_uuid.uuid4().hex[:8]}"
    dataset_dir = DATA_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    _agent_tasks[task_id] = {
        "status": "running",
        "dataset_id": dataset_id,
        "log": [],
        "result": None,
        "error": None,
    }
    
    # 后台线程启动 Agent
    t = threading.Thread(
        target=_run_agent_background,
        args=(task_id, dataset_id, str(dataset_dir), req.user_goal),
        daemon=True,
    )
    t.start()
    
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "status": "running",
    }


@app.get("/api/data/agent-status/{task_id}")
def agent_status(task_id: str, since: int = 0) -> dict[str, Any]:
    """
    轮询 Agent 进度。
    
    since: 上次拿到的 log 条目数，只返回新增的。
    前端每 2 秒调一次。
    """
    if task_id not in _agent_tasks:
        raise HTTPException(404, "Agent 任务不存在")
    
    task = _agent_tasks[task_id]
    
    # 只返回 since 之后的新日志
    new_logs = task["log"][since:]
    formatted = []
    for evt in new_logs:
        etype = evt["type"]
        content = str(evt.get("content", ""))
        if etype == "thought":
            formatted.append({"step": "think", "message": f"💭 {content[:500]}"})
        elif etype == "action":
            formatted.append({"step": "action", "message": f"⚡ {content[:500]}"})
        elif etype == "observation":
            formatted.append({"step": "observe", "message": f"👁 {content[:500]}"})
        elif etype == "insight":
            formatted.append({"step": "think", "message": "✅ 数据获取和分析完成"})
        elif etype == "error":
            formatted.append({"step": "error", "message": f"⚠️ {content[:300]}"})
    
    resp = {
        "status": task["status"],
        "dataset_id": task["dataset_id"],
        "log_total": len(task["log"]),
        "new_logs": formatted,
    }
    
    if task["status"] == "completed":
        resp["result"] = task["result"]
    elif task["status"] == "failed":
        resp["error"] = task["error"]
    
    return resp


@app.get("/api/data/agent-stream/{task_id}")
async def agent_stream(task_id: str):
    """
    SSE 流式输出 Agent 进度。
    
    后台线程执行 Agent，这里每 1.5 秒轮询内部状态并推送新事件。
    发心跳防止代理超时。
    """
    import asyncio

    if task_id not in _agent_tasks:
        raise HTTPException(404, "Agent 任务不存在")

    def _fmt(evt: dict) -> dict:
        etype = evt["type"]
        content = str(evt.get("content", ""))
        if etype == "thought":
            return {"step": "think", "message": f"💭 {content[:600]}"}
        elif etype == "action":
            return {"step": "action", "message": f"⚡ {content[:600]}"}
        elif etype == "observation":
            return {"step": "observe", "message": f"👁 {content[:600]}"}
        elif etype == "insight":
            return {"step": "insight", "message": "✅ 数据获取和分析完成"}
        elif etype == "error":
            return {"step": "error", "message": f"⚠️ {content[:400]}"}
        return {"step": "info", "message": content[:300]}

    async def generate():
        since = 0
        max_wait = 6000  # 100 分钟
        elapsed = 0

        while elapsed < max_wait:
            task = _agent_tasks.get(task_id)
            if not task:
                yield f"data: {json.dumps({'step': 'error', 'message': 'Task not found', 'done': True}, ensure_ascii=False)}\n\n"
                break

            # 推送新日志
            new_logs = task["log"][since:]
            for evt in new_logs:
                yield f"data: {json.dumps(_fmt(evt), ensure_ascii=False)}\n\n"
            since = len(task["log"])

            # 完成
            if task["status"] == "completed":
                result_data = task.get("result", {})
                yield f"data: {json.dumps({'step': 'result', 'done': True, 'data': result_data}, ensure_ascii=False)}\n\n"
                break

            # 失败
            if task["status"] == "failed":
                err_msg = task.get("error", "未知错误")
                yield f"data: {json.dumps({'step': 'error', 'message': f'❌ {err_msg}', 'done': True}, ensure_ascii=False)}\n\n"
                break

            # 心跳（防代理超时）
            if not new_logs:
                yield f"data: {json.dumps({'step': 'heartbeat'}, ensure_ascii=False)}\n\n"

            await asyncio.sleep(1.5)
            elapsed += 1.5

        if elapsed >= max_wait:
            yield f"data: {json.dumps({'step': 'error', 'message': '⏰ Agent 超时', 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/data/{dataset_id}/explore")
def explore_dataset(dataset_id: str, user_goal: str = "") -> dict[str, Any]:
    """
    ReAct Agent 驱动的数据探索
    
    Agent 自主决定如何探索数据：
    - 打开文件看格式
    - 如果读不出来就自己想办法（转格式、换库、重新下载等）
    - 理解数据结构和内容
    - 生成结构化 insights
    """
    try:
        spec = data_manager.get_dataset(dataset_id)
    except ValueError:
        raise HTTPException(404, f"数据集不存在: {dataset_id}")
    
    # 确定 Agent 工作目录
    agent_cwd = spec.storage_path or str(DATA_DIR / dataset_id)
    extracted = Path(agent_cwd) / "extracted"
    if extracted.exists():
        agent_cwd = str(extracted)
    
    from backend.compiler import _use_llm_compiler
    if _use_llm_compiler():
        try:
            from backend.llm_client import get_llm_client
            from backend.data_exploration_agent import run_exploration_agent
            client = get_llm_client()
            
            # 增加超时
            import httpx
            if hasattr(client, 'client'):
                client.client = httpx.Client(
                    base_url=client.config.base_url,
                    headers={"Authorization": f"Bearer {client.config.api_key}", "Content-Type": "application/json"},
                    timeout=600.0,
                )
            
            file_size_human = spec.file_scan.get("total_size_human", "unknown") if spec.file_scan else "unknown"
            
            # 运行 ReAct Agent（同步收集所有步骤）
            exploration = None
            agent_log = []
            for event in run_exploration_agent(
                dataset_dir=agent_cwd,
                filename=spec.filename,
                file_size_human=file_size_human,
                user_goal=user_goal if user_goal else None,
                llm_client=client,
            ):
                agent_log.append(event)
                if event["type"] == "insight":
                    exploration = event["content"]
            
            if exploration:
                spec.exploration = exploration
                if exploration.get("data_type"):
                    spec.data_type = exploration["data_type"]
                meta_path = DATA_DIR / f"{dataset_id}_meta.json"
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
                data_manager.datasets[dataset_id] = spec
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "exploration": exploration or {"business_summary": "Agent 探索完成但未生成结构化结论"},
                "agent_log": [
                    {"type": e["type"], "content": str(e["content"])[:300]}
                    for e in agent_log
                ],
                "updated_spec": {
                    "target_column": spec.target_column,
                    "feature_columns": spec.feature_columns,
                    "data_type": spec.data_type,
                },
            }
        except Exception as e:
            return {
                "success": False,
                "dataset_id": dataset_id,
                "exploration": {"business_summary": f"Agent 探索遇到问题: {str(e)[:200]}"},
                "updated_spec": {},
            }
    else:
        return {
            "success": True,
            "dataset_id": dataset_id,
            "exploration": {
                "business_summary": f"数据集 '{spec.filename}' 包含 {spec.n_rows} 行 × {spec.n_cols} 列。配置 LLM 后可获得智能数据分析。",
                "data_understanding": {
                    "summary": f"{spec.n_rows} 行, {spec.n_cols} 列",
                    "columns_analysis": [
                        {"name": c.to_dict()["name"], "role": "target" if c.name == spec.target_column else "feature"}
                        for c in (spec.columns or [])
                    ],
                },
                "target_recommendation": {
                    "column": spec.target_column,
                    "confidence": 0.5,
                    "reasoning": "基于规则推断（未使用 LLM）",
                },
            },
            "updated_spec": {
                "target_column": spec.target_column,
                "feature_columns": spec.feature_columns,
            },
        }


class DataSearchRequest(BaseModel):
    """搜索公开数据集"""
    user_goal: str = Field(min_length=3)
    task_type: str | None = None


class SyntheticDataRequest(BaseModel):
    """生成合成数据"""
    user_goal: str = Field(min_length=3)
    schema: list[dict] = Field(default_factory=list)
    target_column: str = "target"
    n_rows: int = Field(default=500, ge=50, le=5000)


class DownloadPublicDatasetRequest(BaseModel):
    """下载公开数据集"""
    url: str = Field(min_length=5)
    dataset_name: str = ""
    target_hint: str = ""


class _DatasetLinkHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


@app.post("/api/data/search-public")
def search_public_datasets(req: DataSearchRequest) -> dict[str, Any]:
    """
    搜索公开数据集
    
    1. LLM 推荐最匹配的公开数据集（HuggingFace, GitHub, ModelScope, Kaggle 等）
    2. 实际查询 HuggingFace API 验证数据集存在性
    3. 提供合成数据方案作为备选
    """
    from backend.compiler import _use_llm_compiler

    # LLM 推荐
    llm_suggestions = None
    if _use_llm_compiler():
        try:
            from backend.llm_client import get_llm_client
            client = get_llm_client()
            llm_suggestions = client.suggest_data_sources(
                user_goal=req.user_goal,
                task_type=req.task_type,
            )
        except Exception as e:
            llm_suggestions = {"error": str(e)[:200]}

    # 实际查询 HuggingFace Datasets API
    hf_results = []
    try:
        import httpx
        keywords = []
        if llm_suggestions and llm_suggestions.get("search_keywords"):
            keywords = llm_suggestions["search_keywords"].get("en", [])[:3]
        if not keywords:
            keywords = [req.user_goal[:50]]

        for kw in keywords[:2]:  # 最多查 2 个关键词
            resp = httpx.get(
                "https://huggingface.co/api/datasets",
                params={"search": kw, "limit": 5, "sort": "downloads", "direction": -1},
                timeout=10.0,
            )
            if resp.status_code == 200:
                for ds in resp.json():
                    hf_results.append({
                        "name": ds.get("id", ""),
                        "source": "huggingface",
                        "url": f"https://huggingface.co/datasets/{ds.get('id', '')}",
                        "description": ds.get("description", "")[:200] if ds.get("description") else "",
                        "downloads": ds.get("downloads", 0),
                        "likes": ds.get("likes", 0),
                        "tags": ds.get("tags", [])[:5],
                    })
    except Exception:
        pass  # HuggingFace API 不可用时静默失败

    # 去重
    seen = set()
    unique_hf = []
    for r in hf_results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique_hf.append(r)

    return {
        "success": True,
        "llm_suggestions": llm_suggestions,
        "huggingface_results": unique_hf[:8],
        "has_synthetic_plan": bool(llm_suggestions and llm_suggestions.get("synthetic_data_plan")),
    }


@app.post("/api/data/generate-synthetic")
async def generate_synthetic_data(req: SyntheticDataRequest) -> dict[str, Any]:
    """
    用 AI 生成合成训练数据
    
    基于用户的业务描述和 schema，让 LLM 生成符合业务逻辑的合成数据。
    生成后自动注册为数据集，可直接用于训练。
    """
    from backend.compiler import _use_llm_compiler

    if not _use_llm_compiler():
        raise HTTPException(501, "合成数据生成需要配置 LLM。请设置 LLM_API_KEY 环境变量。")

    try:
        from backend.llm_client import get_llm_client
        client = get_llm_client()

        # 如果没有提供 schema，让 LLM 先设计 schema
        schema = req.schema
        target_column = req.target_column
        if not schema:
            suggestions = client.suggest_data_sources(
                user_goal=req.user_goal,
                task_type=None,
            )
            plan = suggestions.get("synthetic_data_plan", {})
            schema = plan.get("schema", [])
            target_column = plan.get("target_column", "target")

            if not schema:
                raise ValueError("无法自动设计数据 schema，请提供列定义。")

        # 生成合成数据 CSV
        csv_text = client.generate_synthetic_data(
            user_goal=req.user_goal,
            schema=schema,
            target_column=target_column,
            n_rows=min(req.n_rows, 2000),  # LLM 单次上限
        )

        if not csv_text or len(csv_text.strip()) < 20:
            raise ValueError("生成的数据为空，请重试。")

        # 注册为数据集
        content = csv_text.encode('utf-8')
        spec = data_manager.upload_file(
            content=content,
            filename=f"synthetic_{int(time.time())}.csv",
            target_hint=target_column,
        )

        return {
            "success": True,
            "source": "synthetic",
            "dataset_id": spec.dataset_id,
            "filename": spec.filename,
            "n_rows": spec.n_rows,
            "n_cols": spec.n_cols,
            "target_column": spec.target_column,
            "feature_columns": spec.feature_columns,
            "columns": [c.to_dict() for c in spec.columns],
            "caveat": "这是 AI 生成的合成数据，适合验证模型流程和原型测试。建议后续替换为真实业务数据以获得生产级效果。",
        }

    except Exception as e:
        raise HTTPException(400, f"合成数据生成失败: {str(e)}")


async def _resolve_hf_dataset_url(ds_id: str) -> tuple[str, str] | None:
    """
    通过 HuggingFace API 获取数据集的真实 parquet 下载链接。
    处理各种 ID 格式：'ylecun/mnist', 'mnist', 'MNIST' 等。
    """
    import httpx
    
    async def _try_parquet_api(dataset_id: str) -> tuple[str, str] | None:
        try:
            api_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_id}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(api_url)
                if resp.status_code == 200:
                    data = resp.json()
                    files = data.get("parquet_files", [])
                    train = [f for f in files if f.get("split") == "train"]
                    pick = train[0] if train else (files[0] if files else None)
                    if pick and pick.get("url"):
                        safe_name = dataset_id.replace('/', '_')
                        return pick["url"], f"hf_{safe_name}.parquet"
        except Exception:
            pass
        return None
    
    # 1. 直接试（完整 ID 如 ylecun/mnist）
    result = await _try_parquet_api(ds_id)
    if result:
        return result
    
    # 2. 如果没有 /，可能是简写（如 "mnist"）→ 搜索找完整 ID
    if '/' not in ds_id:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                search_resp = await client.get(
                    "https://huggingface.co/api/datasets",
                    params={"search": ds_id, "limit": 5, "sort": "downloads", "direction": -1},
                )
                if search_resp.status_code == 200:
                    for ds in search_resp.json():
                        full_id = ds.get("id", "")
                        if ds_id.lower() in full_id.lower():
                            result = await _try_parquet_api(full_id)
                            if result:
                                return result
        except Exception:
            pass
    
    return None


async def _crawl_dataset_download_link(page_url: str) -> tuple[str, str] | None:
    import httpx
    from urllib.parse import urljoin

    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        response = await client.get(page_url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "text/html" not in response.text[:120].lower():
            return None

        parser = _DatasetLinkHTMLParser()
        parser.feed(response.text)
        candidates: list[str] = []
        for href in parser.links:
            absolute = urljoin(str(response.url), href)
            lower = absolute.lower()
            if any(lower.endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".xls", ".zip", ".7z", ".parquet", ".json", ".jsonl")):
                candidates.append(absolute)

        if not candidates:
            return None

        picked = candidates[0]
        filename = picked.split("/")[-1].split("?")[0] or "crawled_dataset"
        return picked, filename


@app.post("/api/data/download-public")
async def download_public_dataset(req: DownloadPublicDatasetRequest) -> dict[str, Any]:
    """
    下载公开数据集并注册
    
    支持 HuggingFace datasets、GitHub raw 文件链接、
    以及任何返回 CSV/Excel 的公开 URL。
    """
    import httpx

    url = req.url
    filename = req.dataset_name or url.split('/')[-1].split('?')[0] or "public_dataset"

    # HuggingFace datasets 特殊处理
    if "huggingface.co/datasets/" in url and "/resolve/" not in url:
        match = re.search(r'huggingface\.co/datasets/([^/?#]+(?:/[^/?#]+)?)', url)
        if match:
            ds_id = match.group(1)
            resolved = await _resolve_hf_dataset_url(ds_id)
            if resolved:
                url, filename = resolved
    elif not any(url.lower().endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".xls", ".zip", ".7z", ".parquet", ".json", ".jsonl")):
        try:
            crawled = await _crawl_dataset_download_link(url)
            if crawled:
                url, filename = crawled
        except Exception:
            pass

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
    except Exception as e:
        raise HTTPException(502, f"下载失败: {str(e)[:200]}。请检查链接是否可公开访问。")

    if len(content) < 100:
        raise HTTPException(400, "下载的文件为空或太小，可能 URL 不正确。")
    
    # 检测是否下载到了 HTML 错误页而非数据
    content_start = content[:200].decode('utf-8', errors='replace').lower()
    if '<html' in content_start or '<!doctype' in content_start:
        try:
            crawled = await _crawl_dataset_download_link(req.url)
            if crawled and crawled[0] != url:
                async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                    response = await client.get(crawled[0])
                    response.raise_for_status()
                    content = response.content
                    url, filename = crawled
            else:
                raise HTTPException(400, "下载到了 HTML 页面而非数据文件。数据集可能需要认证或 URL 不正确。")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "下载到了 HTML 页面而非数据文件。数据集可能需要认证或 URL 不正确。")

    # 确保有文件扩展名
    if not any(filename.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.xls', '.zip', '.7z', '.parquet', '.json', '.jsonl', '.tsv']):
        content_type = response.headers.get('content-type', '')
        if 'parquet' in content_type:
            filename += '.parquet'
        elif 'excel' in content_type or 'spreadsheet' in content_type:
            filename += '.xlsx'
        else:
            filename += '.csv'

    # 保存文件到磁盘，不管解析成不成功都返回 dataset_id
    # 解析/格式转换 全部交给 ReAct Agent 自己处理
    import uuid as _uuid
    dataset_id = f"ds_{_uuid.uuid4().hex[:8]}"
    dataset_dir = DATA_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dest = dataset_dir / filename
    with open(dest, 'wb') as f:
        f.write(content)
    
    # 尝试让 data_manager 处理，但失败不报错——Agent 会接手
    try:
        spec = data_manager.upload_from_disk(
            file_path=dest,
            filename=filename,
            dataset_id=dataset_id,
            target_hint=req.target_hint if req.target_hint else None,
        )
        return {
            "success": True,
            "source": "public_download",
            "original_url": req.url,
            "dataset_id": spec.dataset_id,
            "filename": spec.filename,
            "n_rows": spec.n_rows,
            "data_type": spec.data_type,
            "file_scan": {
                "total_files": spec.file_scan.get("total_files", 0),
                "total_size_human": spec.file_scan.get("total_size_human", ""),
                "extensions": spec.file_scan.get("extensions", {}),
            },
        }
    except Exception:
        # 解析失败没关系——文件已在磁盘上，返回 dataset_id 让 Agent 去探索
        from backend.data_manager import FileScanner
        file_size = dest.stat().st_size
        return {
            "success": True,
            "source": "public_download",
            "original_url": req.url,
            "dataset_id": dataset_id,
            "filename": filename,
            "n_rows": 0,
            "data_type": "unknown",
            "needs_agent_exploration": True,
            "file_scan": {
                "total_files": 1,
                "total_size_human": FileScanner._human_size(file_size),
            },
        }


@app.get("/api/data/list")
async def list_datasets() -> dict[str, Any]:
    """列出所有数据集"""
    datasets = data_manager.list_datasets()
    return {
        "datasets": [d.to_dict() for d in datasets],
    }


@app.get("/api/data/{dataset_id}")
async def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    """获取数据集信息"""
    try:
        spec = data_manager.get_dataset(dataset_id)
        return spec.to_dict()
    except ValueError as e:
        raise HTTPException(404, str(e))


# ---- 意图编译 API（流式思考版）----

_plan_tasks: dict[str, dict] = {}


def _run_plan_background(task_id: str, user_goal: str, dataset_id: str | None, priority: str):
    """后台线程：流式生成训练方案（思考过程实时输出）"""
    task = _plan_tasks[task_id]
    
    try:
        from backend.compiler import _use_llm_compiler, detect_ambiguity
        from backend.llm_client import get_llm_client
        
        if not _use_llm_compiler():
            task["log"].append({"type": "error", "content": "LLM 未配置"})
            task["status"] = "failed"
            return
        
        client = get_llm_client()
        import httpx
        if hasattr(client, 'client'):
            client.client = httpx.Client(
                base_url=client.config.base_url,
                headers={"Authorization": f"Bearer {client.config.api_key}", "Content-Type": "application/json"},
                timeout=600.0,
            )
        
        task["log"].append({"type": "thinking", "content": ""})
        thinking_idx = len(task["log"]) - 1
        
        # 获取数据 insights（如果有）
        enriched_goal = user_goal
        if dataset_id:
            try:
                ds = data_manager.get_dataset(dataset_id)
                if ds.exploration and ds.exploration.get("training_implications"):
                    implications = "\n".join(f"- {imp}" for imp in ds.exploration["training_implications"])
                    enriched_goal += f"\n\n[数据探索 Agent 的发现]\n{implications}"
                if ds.exploration and ds.exploration.get("business_summary"):
                    enriched_goal += f"\n[数据概况] {ds.exploration['business_summary']}"
            except Exception:
                pass
        
        task["log"].append({"type": "status", "content": "🧠 正在生成训练方案..."})

        response_text = client.chat_completion(
            messages=[
                {"role": "system", "content": _get_plan_system_prompt()},
                {"role": "user", "content": _get_plan_user_prompt(enriched_goal, priority)},
            ],
            temperature=0.2,
            max_tokens=1800,
        )
        task["log"][thinking_idx]["content"] = _truncate_plan_reasoning(response_text)
        
        # 解析 JSON 结果
        try:
            json_str = client._extract_json(response_text)
            plan = json.loads(json_str)
        except Exception:
            plan = {
                "business_layer": {
                    "understanding": response_text[:300],
                    "personalized_approach": "",
                    "core_promises": [],
                    "vs_standard": "",
                },
                "technical_layer": {},
                "confidence": 0.5,
            }
        
        task["status"] = "completed"
        task["result"] = plan
        task["log"].append({"type": "done", "content": "✅ 方案生成完成"})
        
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)[:300]
        task["log"].append({"type": "error", "content": str(e)[:200]})


def _get_plan_system_prompt():
    """方案生成的 system prompt（和 compile_personalized_plan 一致）"""
    return """你是 VibeML 的核心 AI 引擎。将用户需求转化为个性化机器学习训练方案。

要求：
1. 不要输出 <think>、思维链、Markdown 解释或多余前言。
2. 直接输出一个合法 JSON 对象。
3. JSON 必须包含 business_layer、technical_layer、confidence、needs_more_info。
4. technical_layer 的四个子字段都必须存在：data_strategy、model_strategy、loss_function、evaluation。
5. 内容要简洁、可执行，优先给出适合当前任务的数据与训练策略。"""


def _get_plan_user_prompt(user_goal: str, priority: str):
    return f"""用户需求："{user_goal}"
优化偏好：{priority}

请直接输出 JSON，不要输出任何解释性文字：

{{
    "business_layer": {{
        "understanding": "用你自己的话复述需求",
        "personalized_approach": "通俗描述方案",
        "core_promises": ["承诺1", "承诺2"],
        "vs_standard": "与通用方案的区别"
    }},
    "technical_layer": {{
        "data_strategy": {{"title": "...", "reasoning": "...", "details": "..."}},
        "model_strategy": {{"title": "...", "reasoning": "...", "details": "..."}},
        "loss_function": {{"title": "...", "reasoning": "...", "details": "..."}},
        "evaluation": {{"title": "...", "reasoning": "...", "details": "..."}}
    }},
    "confidence": 0.85,
    "needs_more_info": []
}}"""


def _truncate_plan_reasoning(response_text: str, limit: int = 1200) -> str:
    cleaned = response_text.strip()
    cleaned = cleaned.replace("<thinking>", "").replace("</thinking>", "")
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned[:limit]


@app.post("/api/intent/clarify-start")
def start_clarify(req: ClarifyRequest) -> dict[str, Any]:
    """启动方案生成（后台线程），立即返回 task_id"""
    import uuid as _uuid
    task_id = _uuid.uuid4().hex[:12]
    
    _plan_tasks[task_id] = {
        "status": "running",
        "log": [],
        "result": None,
        "error": None,
    }
    
    t = threading.Thread(
        target=_run_plan_background,
        args=(task_id, req.user_goal, req.dataset_id, req.priority),
        daemon=True,
    )
    t.start()
    
    return {"task_id": task_id, "status": "running"}


@app.get("/api/intent/clarify-stream/{task_id}")
async def clarify_stream(task_id: str):
    """SSE 流式输出方案生成的思考过程"""
    import asyncio
    
    if task_id not in _plan_tasks:
        raise HTTPException(404, "Task not found")
    
    async def generate():
        last_thinking_len = 0
        log_since = 0
        max_wait = 6000  # 100 分钟
        elapsed = 0
        
        while elapsed < max_wait:
            task = _plan_tasks.get(task_id)
            if not task:
                break
            
            had_new = False
            
            # 推送新增的 thinking tokens（增量）
            for evt in task["log"]:
                if evt["type"] == "thinking":
                    new_text = evt["content"][last_thinking_len:]
                    if new_text:
                        yield f"data: {json.dumps({'step': 'thinking', 'token': new_text}, ensure_ascii=False)}\n\n"
                        last_thinking_len = len(evt["content"])
                        had_new = True
            
            # 推送新增的其他事件（status/done/error）
            new_events = task["log"][log_since:]
            for evt in new_events:
                if evt["type"] in ("status", "done"):
                    yield f"data: {json.dumps({'step': 'status', 'message': evt['content']}, ensure_ascii=False)}\n\n"
                    had_new = True
                elif evt["type"] == "error":
                    yield f"data: {json.dumps({'step': 'error', 'message': evt['content']}, ensure_ascii=False)}\n\n"
                    had_new = True
            log_since = len(task["log"])
            
            # 完成/失败
            if task["status"] == "completed":
                yield f"data: {json.dumps({'step': 'result', 'done': True, 'plan': task['result']}, ensure_ascii=False)}\n\n"
                break
            elif task["status"] == "failed":
                err = task.get("error", "未知错误")
                yield f"data: {json.dumps({'step': 'error', 'done': True, 'message': err}, ensure_ascii=False)}\n\n"
                break
            
            # 心跳（只在没有新数据时发）
            if not had_new:
                yield f"data: {json.dumps({'step': 'heartbeat'}, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(0.5)
            elapsed += 0.5
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# 保留原有的同步版本作为兼容
@app.post("/api/intent/clarify", response_model=ClarifyResponse)
def clarify_intent(req: ClarifyRequest) -> ClarifyResponse:
    """
    意图澄清 API
    
    流程：
    1. LLM 判断是否为 ML 需求、是否足够清晰
    2. 如果清晰 → 生成个性化双层方案（业务层 + 技术层）
    3. 如果模糊 → 返回引导追问
    """
    # 空输入快速拒绝
    if not req.user_goal or not req.user_goal.strip():
        return ClarifyResponse(
            is_ambiguous=True,
            reasons=["请输入你的需求"],
            follow_up_questions=["请描述你的业务场景和数据，我来帮你设计训练方案。"],
        )
    
    # LLM 判断：是否为 ML 需求 → 是则直接进方案生成
    # 设计理念：LLM 判断 is_ml_request=true 就说明它理解了需求，
    # 不需要再追问"缺少必须保留项"之类的表单字段。
    # 方案生成阶段会在 needs_more_info 里标注真正需要补充的信息。
    _debug_info = {"deploy_tag": "20260408-v6"}
    try:
        is_ambiguous, reasons, follow_ups = detect_ambiguity(req)
        _debug_info["detect_raw"] = {"is_ambiguous": is_ambiguous}
    except Exception as e:
        _debug_info["detect_error"] = str(e)[:200]
        is_ambiguous, reasons, follow_ups = False, [], []
    
    personalized_plan = None
    
    if not is_ambiguous:
        # 需求清晰 → 生成个性化方案
        from backend.compiler import _use_llm_compiler
        if _use_llm_compiler():
            try:
                from backend.llm_client import get_llm_client
                client = get_llm_client()
                
                # 增加 httpx 超时（如果客户端超时太短）
                if hasattr(client, 'client') and hasattr(client.client, '_transport'):
                    try:
                        import httpx
                        client.client = httpx.Client(
                            base_url=client.config.base_url,
                            headers={"Authorization": f"Bearer {client.config.api_key}", "Content-Type": "application/json"},
                            timeout=600.0,
                        )
                    except Exception:
                        pass
                
                # 获取数据 schema 和 Agent 探索的 insights
                data_schema = None
                data_insights = None
                if req.dataset_id:
                    try:
                        ds = data_manager.get_dataset(req.dataset_id)
                        if ds.columns:
                            data_schema = {
                                "columns": [
                                    {"name": c.name, "type": c.column_type.value,
                                     "sample": c.sample_values[:3] if c.sample_values else []}
                                    for c in ds.columns
                                ],
                                "target_column": ds.target_column,
                                "n_rows": ds.n_rows,
                            }
                        # 注入 Agent 探索的 insights（如果有）
                        if ds.exploration:
                            data_insights = {
                                "data_type": ds.exploration.get("data_type", ds.data_type),
                                "summary": ds.exploration.get("business_summary", ""),
                                "training_implications": ds.exploration.get("training_implications", []),
                                "statistics": ds.exploration.get("statistics", {}),
                                "quality_issues": ds.exploration.get("quality_issues", []),
                            }
                    except Exception:
                        pass
                
                # Agent 探索的 training_implications 直接影响方案生成
                enriched_goal = req.user_goal
                if data_insights and data_insights.get("training_implications"):
                    implications = "\n".join(f"- {imp}" for imp in data_insights["training_implications"])
                    enriched_goal += f"\n\n[数据探索 Agent 的发现]\n{implications}"
                    if data_insights.get("summary"):
                        enriched_goal += f"\n\n[数据概况] {data_insights['summary']}"
                
                personalized_plan = client.compile_personalized_plan(
                    user_goal=enriched_goal,
                    must_keep=req.must_keep if req.must_keep else None,
                    worst_errors=req.worst_errors if req.worst_errors else None,
                    priority=req.priority,
                    data_schema=data_schema,
                )
            except Exception as e:
                _debug_info["plan_error"] = str(e)[:300]
                personalized_plan = {
                    "business_layer": {
                        "understanding": f"我理解了你的需求（{req.user_goal[:80]}...），正在生成个性化方案。",
                        "personalized_approach": "方案生成需要较长时间，请稍后刷新或重新提交。如果问题持续，可以直接上传数据开始训练。",
                        "core_promises": [],
                        "vs_standard": "",
                    },
                    "technical_layer": {},
                    "confidence": 0.3,
                    "needs_more_info": [f"方案生成超时，请重试"],
                }
    
    resp = ClarifyResponse(
        is_ambiguous=is_ambiguous,
        reasons=reasons,
        follow_up_questions=follow_ups,
        personalized_plan=personalized_plan,
    )
    # 临时 debug：在 compiled_intent 里塞诊断信息
    resp.compiled_intent = _debug_info
    return resp


@app.post("/api/intent/compile")
async def compile_intent(req: ClarifyRequest) -> dict[str, Any]:
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
async def start_training(req: StartTrainingRequest) -> dict[str, Any]:
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
async def get_training_status(job_id: str) -> JobStatusResponse:
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
            error_message=(
                job.training_result.error_message
                if job.training_result and job.status != "completed"
                else job.message if job.status == "failed" else None
            ),
            recent_logs=job.recent_logs,
            train_metrics=job.train_metrics,
            val_metrics=job.val_metrics,
            best_score=job.training_result.best_metric_score if job.training_result else None,
            elapsed_time=round(elapsed, 1),
        )


@app.post("/api/training/{job_id}/pause")
async def pause_training(job_id: str) -> dict[str, Any]:
    """暂停训练"""
    job = job_manager.pause_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "message": "训练已暂停",
    }


@app.post("/api/training/{job_id}/resume")
async def resume_training(job_id: str) -> dict[str, Any]:
    """恢复训练"""
    job = job_manager.resume_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "message": "训练已恢复",
    }


@app.post("/api/training/{job_id}/stop")
async def stop_training(job_id: str) -> dict[str, Any]:
    """停止训练"""
    job = job_manager.stop_job(job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "message": "训练已停止",
    }


# ---- 训练任务列表 + Checkpoint / 模型下载 API ----

@app.get("/api/training/jobs")
async def list_training_jobs() -> dict[str, Any]:
    """列出所有训练任务"""
    jobs = []
    for job_id, job in job_manager.jobs.items():
        jobs.append({
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "dataset_id": job.dataset_id,
            "progress": job.progress,
            "message": job.message,
            "has_model": _resolve_artifact(job, "model")[0] is not None,
            "train_metrics": job.train_metrics,
            "val_metrics": job.val_metrics,
        })
    return {"jobs": jobs}


@app.get("/api/training/{job_id}/checkpoints")
async def list_checkpoints(job_id: str) -> dict[str, Any]:
    """列出所有 checkpoint"""
    job = job_manager.get(job_id)
    
    checkpoints = []
    
    model_path, model_name = _resolve_artifact(job, "model")
    if model_path and model_name:
        checkpoints.append({
            "type": "model",
            "file": model_name,
            "size": model_path.stat().st_size,
            "created": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
        })
    
    preprocessor_path, preprocessor_name = _resolve_artifact(job, "preprocessor")
    if preprocessor_path and preprocessor_name:
        checkpoints.append({
            "type": "preprocessor",
            "file": preprocessor_name,
            "size": preprocessor_path.stat().st_size,
            "created": datetime.fromtimestamp(preprocessor_path.stat().st_mtime).isoformat(),
        })
    
    result_path, result_name = _resolve_artifact(job, "metadata")
    if result_path and result_name:
        checkpoints.append({
            "type": "metadata",
            "file": result_name,
            "size": result_path.stat().st_size,
            "created": datetime.fromtimestamp(result_path.stat().st_mtime).isoformat(),
        })
    
    return {
        "job_id": job_id,
        "status": job.status,
        "checkpoints": checkpoints,
    }


@app.get("/api/training/{job_id}/download/{file_type}")
async def download_artifact(job_id: str, file_type: str) -> FileResponse:
    """
    下载训练产物
    
    file_type: model, preprocessor, metadata
    """
    job = job_manager.get(job_id)

    path, filename = _resolve_artifact(job, file_type)
    if not path or not filename:
        raise HTTPException(404, f"文件不存在: {job_id}/{file_type}")

    media_type = (
        "application/json"
        if file_type == "metadata"
        else mimetypes.guess_type(filename)[0] or "application/octet-stream"
    )
    
    return FileResponse(
        path=path,
        filename=filename,
        media_type=media_type,
    )


@app.get("/api/training/{job_id}/result")
async def get_training_result(job_id: str) -> dict[str, Any]:
    """获取完整训练结果"""
    job = job_manager.get(job_id)
    return _build_training_result_payload(job)


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
