# VibeML Agent 部署指南

## 🚀 快速启动（本地）

```bash
# 1. 进入项目目录
cd /root/VibeML-Agent

# 2. 启动服务（本地模式）
./start.sh local

# 或者使用 Docker
./start.sh docker
```

服务启动后访问：
- 前端界面: http://localhost:8001
- V2版本: http://localhost:8001/app/v2.html
- API文档: http://localhost:8001/docs

## ⚙️ 配置 LLM（可选但推荐）

编辑 `.env` 文件：

```bash
# OpenAI 或兼容 API
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

# 或使用其他兼容 API（如 Azure、第三方代理）
# LLM_BASE_URL=https://your-api-endpoint.com/v1
```

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t vibeml-agent .

# 运行容器
docker run -d \
  --name vibeml-agent \
  -p 8000:8000 \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -v "$(pwd)/outputs:/app/outputs" \
  vibeml-agent

# 查看日志
docker logs -f vibeml-agent
```

## ☁️ 云服务器部署（生产环境）

### 使用 systemd 服务

创建 `/etc/systemd/system/vibeml.service`：

```ini
[Unit]
Description=VibeML Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/vibeml
Environment=LLM_API_KEY=your-api-key
Environment=PYTHONPATH=/opt/vibeml
ExecStart=/usr/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable vibeml
sudo systemctl start vibeml
sudo systemctl status vibeml
```

### 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## 🔧 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API 密钥 | 空（使用规则回退） |
| `LLM_BASE_URL` | API 基础 URL | https://api.openai.com/v1 |
| `LLM_MODEL_NAME` | 模型名称 | gpt-4o-mini |
| `PORT` | 服务端口 | 8000 |

### 数据目录

- `outputs/` - 训练输出和 checkpoint
- `backend/data/` - 上传的数据集

## 🧪 测试

```bash
# 运行测试
pytest tests/ -v

# 测试特定模块
pytest tests/test_clarification.py -v
pytest tests/test_checkpoint_download.py -v
```

## 📊 当前部署状态

```
服务地址: http://localhost:8001
状态: ✅ 运行中
LLM: 未配置（使用规则回退）
训练: CPU 模式可用
```

## 📝 注意事项

1. **LLM配置**: 如需意图解析功能，请配置 LLM_API_KEY
2. **CPU训练**: 当前部署使用CPU训练，适合小规模数据
3. **数据持久化**: 使用Docker时记得挂载 volumes
4. **端口冲突**: 如果8000被占用，会自动使用8001

## 🔍 故障排查

```bash
# 检查服务状态
curl http://localhost:8001/api/health

# 查看日志
tail -f /tmp/vibeml.log

# 重启服务
pkill -f "uvicorn backend.main:app"
./start.sh local
```
