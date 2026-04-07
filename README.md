# VibeML-Agent（可用版）

这是一个可直接运行的前后端应用，支持：

- 意图模糊检测与追问
- 训练任务启动/暂停/继续/停止
- 训练中随时暂停并下载 checkpoint
- 前端控制台直接操作完整流程

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

启动后访问：

- `http://127.0.0.1:8000/`（主页面）
- `http://127.0.0.1:8000/api/health`

## API 概览

### 意图澄清

- `POST /api/intent/clarify`

### 训练流程

- `POST /api/training/start`
- `GET /api/training/{job_id}`
- `POST /api/training/{job_id}/pause`
- `POST /api/training/{job_id}/resume`
- `POST /api/training/{job_id}/stop`

### Checkpoint

- `GET /api/training/{job_id}/checkpoints`
- `GET /api/training/{job_id}/download/latest`
- `GET /api/training/{job_id}/download/{filename}`

## 手动验证示例

```bash
curl -s http://127.0.0.1:8000/api/health

curl -s -X POST http://127.0.0.1:8000/api/intent/clarify \
  -H 'Content-Type: application/json' \
  -d '{"user_goal":"将客服工单转成稳定JSON","must_keep":["JSON结构"],"can_change":["措辞"],"worst_errors":["格式错误"],"priority":"quality"}'
```

## Vast.ai 接入建议

当前训练器是本地模拟器（便于先跑通产品流程）。
你后续接入 Vast.ai 时建议替换为：

1. `/api/training/start` 提交远端任务并记录任务 ID
2. 用后台轮询同步远端状态到本地 `JobState`
3. checkpoint 下载链接改为对象存储 URL（S3/MinIO）
