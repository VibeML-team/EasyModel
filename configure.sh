#!/bin/bash

# VibeML Agent LLM 配置脚本

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}   VibeML Agent LLM 配置${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""

# 选择提供商
echo -e "${BLUE}请选择 LLM 提供商:${NC}"
echo "1) OpenAI (官方)"
echo "2) DeepSeek"
echo "3) Azure OpenAI"
echo "4) 其他 OpenAI 兼容 API"
echo ""
read -p "请输入数字 (1-4): " provider

case $provider in
    1)
        DEFAULT_URL="https://api.openai.com/v1"
        DEFAULT_MODEL="gpt-4o-mini"
        PROVIDER_NAME="OpenAI"
        ;;
    2)
        DEFAULT_URL="https://api.deepseek.com/v1"
        DEFAULT_MODEL="deepseek-chat"
        PROVIDER_NAME="DeepSeek"
        ;;
    3)
        DEFAULT_URL=""
        DEFAULT_MODEL="gpt-4"
        PROVIDER_NAME="Azure OpenAI"
        echo -e "${YELLOW}提示: Azure URL 格式: https://your-resource.openai.azure.com/openai/deployments/your-deployment${NC}"
        ;;
    4)
        DEFAULT_URL=""
        DEFAULT_MODEL=""
        PROVIDER_NAME="自定义"
        ;;
    *)
        echo "无效选择，使用默认 OpenAI 配置"
        DEFAULT_URL="https://api.openai.com/v1"
        DEFAULT_MODEL="gpt-4o-mini"
        PROVIDER_NAME="OpenAI"
        ;;
esac

echo ""
echo -e "${BLUE}配置 $PROVIDER_NAME:${NC}"
echo ""

# 读取 API Key
read -p "请输入 API Key: " api_key

if [ -z "$api_key" ]; then
    echo -e "${YELLOW}⚠️  API Key 为空，将使用规则回退模式${NC}"
    exit 0
fi

# 读取 Base URL
if [ -z "$DEFAULT_URL" ]; then
    read -p "请输入 Base URL: " base_url
else
    read -p "请输入 Base URL (默认: $DEFAULT_URL): " base_url
    base_url=${base_url:-$DEFAULT_URL}
fi

# 读取模型名称
if [ -z "$DEFAULT_MODEL" ]; then
    read -p "请输入模型名称: " model_name
else
    read -p "请输入模型名称 (默认: $DEFAULT_MODEL): " model_name
    model_name=${model_name:-$DEFAULT_MODEL}
fi

# 创建 .env 文件
cat > .env << EOF
# VibeML Agent 环境变量配置
# 生成时间: $(date)

# LLM 配置
LLM_API_KEY=$api_key
LLM_BASE_URL=$base_url
LLM_MODEL_NAME=$model_name

# 备用环境变量名
OPENAI_API_KEY=$api_key
EOF

echo ""
echo -e "${GREEN}✅ 配置已保存到 .env 文件${NC}"
echo ""
echo -e "配置信息:"
echo "  提供商: $PROVIDER_NAME"
echo "  模型: $model_name"
echo "  API URL: $base_url"
echo ""

# 测试连接
echo -e "${BLUE}🧪 测试 LLM 连接...${NC}"

# 设置临时环境变量并测试
export LLM_API_KEY=$api_key
export LLM_BASE_URL=$base_url
export LLM_MODEL_NAME=$model_name

python3 << 'PYEOF'
import os
import sys

sys.path.insert(0, '/root/VibeML-Agent')

try:
    from backend.llm_client import LLMClient, LLMConfig
    
    config = LLMConfig(
        api_key=os.getenv('LLM_API_KEY'),
        base_url=os.getenv('LLM_BASE_URL'),
        model_name=os.getenv('LLM_MODEL_NAME')
    )
    
    client = LLMClient(config)
    
    # 简单测试
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "你是一个帮助用户配置ML系统的助手。请简短回复。"},
            {"role": "user", "content": "配置测试，请回复：VibeML配置成功"}
        ],
        temperature=0.1,
        max_tokens=50
    )
    
    if "成功" in response or "VibeML" in response:
        print("✅ LLM 连接测试通过！")
        print(f"模型响应: {response[:100]}")
    else:
        print(f"⚠️  测试响应: {response[:100]}")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    print("请检查 API Key 和网络连接")
PYEOF

echo ""
echo -e "${GREEN}====================================${NC}"
echo -e "现在可以启动服务: ${YELLOW}./start.sh local${NC}"
echo -e "${GREEN}====================================${NC}"
