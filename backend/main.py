from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


Priority = Literal["quality", "latency", "cost"]
Mode = Literal["conservative", "standard", "aggressive"]


class ClarifyRequest(BaseModel):
    user_goal: str = Field(min_length=5)
    must_keep: list[str] = Field(default_factory=list)
    can_change: list[str] = Field(default_factory=list)
    worst_errors: list[str] = Field(default_factory=list)
    priority: Priority = "quality"


class ClarifyResponse(BaseModel):
    is_ambiguous: bool
    reasons: list[str]
    follow_up_questions: list[str]
    compiled_intent: dict[str, Any]


class StartTrainingRequest(BaseModel):
    objective: str = Field(min_length=5)
    must_keep: list[str] = Field(default_factory=list)
    can_change: list[str] = Field(default_factory=list)
    worst_errors: list[str] = Field(default_factory=list)
    priority: Priority = "quality"
    mode: Mode = "standard"
    sample_notes: str = ""


@dataclass
class JobState:
    job_id: str
    created_at: str
    status: str = "running"
    current_step: int = 0
    total_steps: int = 60
    checkpoint_every: int = 5
    train_loss: float = 1.0
    val_loss: float = 1.2
    best_checkpoint_step: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, JobState] = {}

    def create_job(self, cfg: dict[str, Any]) -> JobState:
        job_id = uuid.uuid4().hex[:10]
        job = JobState(job_id=job_id, created_at=_utc_now(), config=cfg)
        self.jobs[job_id] = job
        thread = threading.Thread(target=self._training_loop, args=(job,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> JobState:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def _training_loop(self, job: JobState) -> None:
        while True:
            time.sleep(0.6)
            with job.lock:
                if job.status in {"completed", "stopped"}:
                    return
                if job.status == "paused":
                    continue

                job.current_step += 1
                decay = 0.007 if job.config.get("mode") != "aggressive" else 0.011
                val_decay = 0.005 if job.config.get("mode") != "aggressive" else 0.008
                job.train_loss = max(0.03, job.train_loss - decay)
                job.val_loss = max(0.05, job.val_loss - val_decay)
                job.logs.append(
                    f"step={job.current_step}, train={job.train_loss:.4f}, val={job.val_loss:.4f}, status=running"
                )

                if job.current_step % job.checkpoint_every == 0:
                    ckpt = persist_checkpoint(job)
                    job.logs.append(f"checkpoint={ckpt['file']}")
                    if job.best_checkpoint_step == 0 or ckpt["metrics"]["val_loss"] <= min(
                        c["metrics"]["val_loss"] for c in job.checkpoints
                    ):
                        job.best_checkpoint_step = job.current_step

                if job.current_step >= job.total_steps:
                    job.status = "completed"
                    persist_checkpoint(job)
                    job.logs.append("training completed")
                    return


jobs = JobManager()

app = FastAPI(title="VibeML Agent", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="app")


@app.get("/")
def root() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_ambiguity(req: ClarifyRequest) -> ClarifyResponse:
    reasons: list[str] = []
    follow_ups: list[str] = []

    if not req.must_keep:
        reasons.append("缺少必须保留项")
        follow_ups.append("哪些元素绝对不能被改变？请至少给 1-3 条。")
    if not req.worst_errors:
        reasons.append("缺少不可接受错误定义")
        follow_ups.append("最不能接受的错误是什么（例如结构漂移、漏报、格式错误）？")
    if len(req.user_goal) < 12:
        reasons.append("目标描述过短")
        follow_ups.append("请补充一句：谁在什么场景使用、怎样才算成功。")

    compiled = {
        "objective": req.user_goal,
        "must_keep": req.must_keep,
        "can_change": req.can_change,
        "worst_errors": req.worst_errors,
        "priority": req.priority,
        "recommended_mode": "conservative" if len(reasons) > 0 else "standard",
    }
    return ClarifyResponse(
        is_ambiguous=len(reasons) > 0,
        reasons=reasons,
        follow_up_questions=follow_ups,
        compiled_intent=compiled,
    )


def persist_checkpoint(job: JobState) -> dict[str, Any]:
    filename = f"{job.job_id}_step_{job.current_step}.json"
    path = CHECKPOINT_DIR / filename
    payload = {
        "job_id": job.job_id,
        "timestamp": _utc_now(),
        "step": job.current_step,
        "status": job.status,
        "metrics": {"train_loss": round(job.train_loss, 4), "val_loss": round(job.val_loss, 4)},
        "config_snapshot": job.config,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "file": filename,
        "step": payload["step"],
        "timestamp": payload["timestamp"],
        "metrics": payload["metrics"],
    }
    job.checkpoints.append(meta)
    return meta


@app.post("/api/intent/clarify", response_model=ClarifyResponse)
def clarify_intent(req: ClarifyRequest) -> ClarifyResponse:
    return detect_ambiguity(req)


@app.post("/api/training/start")
def start_training(req: StartTrainingRequest) -> dict[str, Any]:
    job = jobs.create_job(
        {
            "objective": req.objective,
            "must_keep": req.must_keep,
            "can_change": req.can_change,
            "worst_errors": req.worst_errors,
            "priority": req.priority,
            "mode": req.mode,
            "sample_notes": req.sample_notes,
        }
    )
    return {"job_id": job.job_id, "status": job.status, "status_url": f"/api/training/{job.job_id}"}


@app.get("/api/training/{job_id}")
def get_training(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    with job.lock:
        return {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "status": job.status,
            "progress": round((job.current_step / job.total_steps) * 100, 1),
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "train_loss": round(job.train_loss, 4),
            "val_loss": round(job.val_loss, 4),
            "best_checkpoint_step": job.best_checkpoint_step,
            "latest_checkpoints": job.checkpoints[-10:],
            "latest_logs": job.logs[-12:],
            "config": job.config,
        }


@app.post("/api/training/{job_id}/pause")
def pause_training(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    with job.lock:
        if job.status in {"completed", "stopped"}:
            return {"job_id": job_id, "status": job.status, "message": "job is already finished"}
        job.status = "paused"
        checkpoint = persist_checkpoint(job)
    return {"job_id": job_id, "status": "paused", "checkpoint": checkpoint}


@app.post("/api/training/{job_id}/resume")
def resume_training(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    with job.lock:
        if job.status in {"completed", "stopped"}:
            return {"job_id": job_id, "status": job.status, "message": "cannot resume finished job"}
        job.status = "running"
    return {"job_id": job_id, "status": "running"}


@app.post("/api/training/{job_id}/stop")
def stop_training(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    with job.lock:
        if job.status in {"completed", "stopped"}:
            return {"job_id": job_id, "status": job.status, "message": "job already finished"}
        job.status = "stopped"
        checkpoint = persist_checkpoint(job)
    return {"job_id": job_id, "status": "stopped", "checkpoint": checkpoint}


@app.get("/api/training/{job_id}/checkpoints")
def list_checkpoints(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    with job.lock:
        return {
            "job_id": job_id,
            "count": len(job.checkpoints),
            "checkpoints": job.checkpoints,
            "best_checkpoint_step": job.best_checkpoint_step,
        }


@app.get("/api/training/{job_id}/download/latest")
def download_latest_checkpoint(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    with job.lock:
        if not job.checkpoints:
            raise HTTPException(status_code=404, detail="No checkpoint available yet")
        filename = job.checkpoints[-1]["file"]
    return download_checkpoint(job_id=job_id, filename=filename)


@app.get("/api/training/{job_id}/download/{filename}")
def download_checkpoint(job_id: str, filename: str) -> FileResponse:
    jobs.get(job_id)
    path = CHECKPOINT_DIR / filename
    if not path.exists() or not filename.startswith(f"{job_id}_"):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return FileResponse(path=path, filename=filename, media_type="application/json")
