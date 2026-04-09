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
from backend.data_manager import data_manager, DataSpec, DataManager, DATA_DIR
from backend.trainer import AutoMLTrainer, TrainingConfig, TrainingResult, CHECKPOINT_DIR

# 导入 V2 API
from backend.v2_api import router as v2_router

# 目录设置
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
NEXT_FRONTEND_DIR = PROJECT_DIR / "web" / "dist"
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

# 挂载新前端（React 重构版，渐进迁移入口）
if NEXT_FRONTEND_DIR.exists():
    app.mount("/app-next", StaticFiles(directory=NEXT_FRONTEND_DIR, html=True), name="app-next")


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
                        timeout=180.0,
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


@app.post("/api/data/{dataset_id}/explore")
def explore_dataset(dataset_id: str, user_goal: str = "") -> dict[str, Any]:
    """
    LLM 驱动的数据探索 Agent
    
    让 LLM 像数据科学家一样浏览数据：
    - 理解每列的业务含义
    - 识别 feature vs target
    - 发现数据质量问题
    - 给出通俗的数据摘要
    """
    try:
        spec = data_manager.get_dataset(dataset_id)
    except ValueError:
        raise HTTPException(404, f"数据集不存在: {dataset_id}")
    
    # 加载 DataFrame 获取样本行
    try:
        df = data_manager.load_dataframe(dataset_id)
        # 取前几行作为样本给 LLM 看
        sample_rows = df.head(8).to_dict(orient='records')
        # 让 NaN 变成 null（JSON 兼容）
        for row in sample_rows:
            for k, v in row.items():
                if pd.isna(v):
                    row[k] = None
    except Exception:
        sample_rows = []
    
    # 构建列信息
    columns_info = [c.to_dict() for c in spec.columns] if spec.columns else []
    
    # 调用 LLM 数据探索
    from backend.compiler import _use_llm_compiler
    if _use_llm_compiler():
        try:
            from backend.llm_client import get_llm_client
            client = get_llm_client()
            
            exploration = client.explore_data(
                columns_info=columns_info,
                sample_rows=sample_rows,
                n_rows=spec.n_rows,
                n_cols=spec.n_cols,
                filename=spec.filename,
                user_goal=user_goal if user_goal else None,
            )
            
            # 如果 LLM 推荐了不同的 target，更新 DataSpec
            target_rec = exploration.get("target_recommendation", {})
            if target_rec.get("column") and target_rec.get("confidence", 0) > 0.6:
                recommended_target = target_rec["column"]
                if recommended_target in [c["name"] for c in columns_info]:
                    spec.target_column = recommended_target
                    spec.feature_columns = [
                        c["name"] for c in columns_info
                        if c["name"] != recommended_target and c.get("column_type") != "id"
                    ]
                    # 更新持久化的元数据
                    meta_path = DATA_DIR / f"{dataset_id}_meta.json"
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
                    data_manager.datasets[dataset_id] = spec
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "exploration": exploration,
                "updated_spec": {
                    "target_column": spec.target_column,
                    "feature_columns": spec.feature_columns,
                },
            }
        except Exception as e:
            # LLM 失败，返回基础信息
            return {
                "success": True,
                "dataset_id": dataset_id,
                "exploration": {
                    "business_summary": f"数据集包含 {spec.n_rows} 行 × {spec.n_cols} 列。LLM 分析暂时不可用。",
                    "data_understanding": {"summary": f"基本信息：{spec.n_rows} 行, {spec.n_cols} 列"},
                },
                "updated_spec": {
                    "target_column": spec.target_column,
                    "feature_columns": spec.feature_columns,
                },
                "llm_error": str(e)[:200],
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
        # 尝试通过 datasets API 获取下载链接
        # e.g. huggingface.co/datasets/scikit-learn/iris → 尝试直接获取 parquet/csv
        import re
        match = re.search(r'huggingface\.co/datasets/([^/?#]+/[^/?#]+)', url)
        if match:
            ds_id = match.group(1)
            # 尝试下载默认 split 的 CSV
            url = f"https://huggingface.co/datasets/{ds_id}/resolve/main/data/train.csv"
            filename = f"hf_{ds_id.replace('/', '_')}.csv"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
    except Exception as e:
        raise HTTPException(502, f"下载失败: {str(e)[:200]}。请检查链接是否可公开访问。")

    if len(content) < 10:
        raise HTTPException(400, "下载的文件为空。")

    # 确保有文件扩展名
    if not any(filename.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.xls', '.zip', '.7z', '.parquet']):
        content_type = response.headers.get('content-type', '')
        if 'parquet' in content_type:
            filename += '.parquet'
        elif 'excel' in content_type or 'spreadsheet' in content_type:
            filename += '.xlsx'
        else:
            filename += '.csv'

    # 处理 parquet
    if filename.lower().endswith('.parquet'):
        try:
            import io
            df = pd.read_parquet(io.BytesIO(content))
            csv_buf = df.to_csv(index=False)
            content = csv_buf.encode('utf-8')
            filename = filename.replace('.parquet', '.csv')
        except Exception as e:
            raise HTTPException(400, f"Parquet 解析失败: {str(e)[:200]}")

    try:
        spec = data_manager.upload_file(
            content=content,
            filename=filename,
            target_hint=req.target_hint if req.target_hint else None,
        )
        return {
            "success": True,
            "source": "public_download",
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
    _debug_info = {"deploy_tag": "20260408-v5"}
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
                            timeout=180.0,
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
