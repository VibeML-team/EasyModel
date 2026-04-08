# VibeML Zeabur 部署信息

## 部署状态
- **项目 ID**: `69d5f8d342bf3b50dde1fd1d`
- **服务 ID**: `69d5f9039da252559b38d314`
- **访问域名**: https://vibeml-agent.zeabur.app
- **服务器 IP**: 49.12.239.159 (Hetzner Nuremberg)

## 环境变量
- `LLM_API_KEY`: 已配置
- `LLM_BASE_URL`: https://api.siliconflow.cn/v1
- `LLM_MODEL_NAME`: deepseek-ai/DeepSeek-V3.2

## 常用命令

```bash
# 查看部署日志
npx zeabur@latest deployment log --service-id 69d5f9039da252559b38d314 -t runtime -i=false

# 查看构建日志
npx zeabur@latest deployment log --service-id 69d5f9039da252559b38d314 -t build -i=false

# 重启服务
npx zeabur@latest service restart --id 69d5f9039da252559b38d314 -y -i=false

# 更新代码后重新部署
npx zeabur@latest deploy --project-id 69d5f8d342bf3b50dde1fd1d --service-id 69d5f9039da252559b38d314 --json
```

## Zeabur 控制台
https://zeabur.com/projects/69d5f8d342bf3b50dde1fd1d
