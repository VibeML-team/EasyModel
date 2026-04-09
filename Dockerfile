FROM node:22-alpine AS web-builder

WORKDIR /app

# 使用 pnpm 构建前端产物
RUN corepack enable && corepack prepare pnpm@10.18.3 --activate

COPY pnpm-lock.yaml pnpm-workspace.yaml ./
COPY web/package.json ./web/package.json
COPY web/ ./web/

RUN pnpm install --frozen-lockfile
RUN pnpm --dir web build


FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（tree/file 用于 Agent 探索数据集）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    tree \
    file \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY backend/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
# .build_tag 文件每次部署前更新，确保 Docker 不跳过 COPY
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 复制前端重构版构建产物（挂载到 /app-next）
COPY --from=web-builder /app/web/dist ./web/dist

# 创建输出目录
RUN mkdir -p outputs backend/data

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# 启动命令（--limit-max-request-size 0 = 无限制，大文件上传必需）
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "300"]
