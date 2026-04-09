#!/bin/bash

# VibeML Agent 启动脚本
# 支持本地启动和Docker启动

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}   VibeML Agent 启动脚本${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""

# 检查环境变量
if [ -z "$LLM_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  警告: LLM_API_KEY 未设置${NC}"
    echo -e "${YELLOW}   系统将使用规则回退模式（无LLM意图解析）${NC}"
    echo ""
fi

# 创建必要目录
mkdir -p outputs backend/data

# 检查启动模式
MODE=${1:-local}

if [ "$MODE" = "docker" ]; then
    echo -e "${GREEN}🐳 使用 Docker 启动...${NC}"
    
    # 检查docker是否安装
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装${NC}"
        exit 1
    fi
    
    # 构建镜像
    echo "📦 构建 Docker 镜像..."
    docker build -t vibeml-agent .
    
    # 启动容器
    echo "🚀 启动容器..."
    docker run -d \
        --name vibeml-agent \
        -p 8000:8080 \
        -e LLM_API_KEY="$LLM_API_KEY" \
        -e LLM_BASE_URL="${LLM_BASE_URL:-https://api.openai.com/v1}" \
        -e LLM_MODEL_NAME="${LLM_MODEL_NAME:-gpt-4o-mini}" \
        -v "$(pwd)/outputs:/app/outputs" \
        -v "$(pwd)/backend/data:/app/backend/data" \
        vibeml-agent
    
    echo ""
    echo -e "${GREEN}✅ 服务已启动${NC}"
    echo -e "   访问: http://localhost:8000"
    echo -e "   API文档: http://localhost:8000/docs"
    echo ""
    echo -e "查看日志: ${YELLOW}docker logs -f vibeml-agent${NC}"
    echo -e "停止服务: ${YELLOW}docker stop vibeml-agent && docker rm vibeml-agent${NC}"

else
    echo -e "${GREEN}🐍 使用本地 Python 启动...${NC}"
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 未安装${NC}"
        exit 1
    fi
    
    # 检查依赖
    echo "📦 检查依赖..."
    if ! python3 -c "import fastapi" 2>/dev/null; then
        echo "📥 安装依赖..."
        pip install -r backend/requirements.txt
    fi
    
    # 启动服务
    echo "🚀 启动服务..."
    echo ""
    echo -e "${GREEN}====================================${NC}"
    echo -e "${GREEN}   服务启动成功！${NC}"
    echo -e "${GREEN}====================================${NC}"
    echo ""
    echo -e "   🌐 前端界面: ${YELLOW}http://localhost:8000${NC}"
    echo -e "   📚 API文档:  ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "   🔍 健康检查: ${YELLOW}http://localhost:8000/api/health${NC}"
    echo ""
    echo -e "   ${GREEN}V2版本界面:${NC} ${YELLOW}http://localhost:8000/app/v2.html${NC}"
    echo ""
    echo -e "${GREEN}====================================${NC}"
    echo ""
    
    # 启动uvicorn
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
fi
