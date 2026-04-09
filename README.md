# VibeML Agent - 全自动炼丹平台

从自然语言需求到可部署模型权重的端到端 AutoML 平台。

> **chat2objective + chat2model = 全自动 ML 交付**

## ✨ 核心特性

### 1. 自然语言需求编译 (chat2objective)
- 自动将模糊业务需求编译为结构化 `ObjectiveSpec`
- 智能识别任务类型（分类/回归/时序/排序）
- 自动选择评估指标和损失函数
- 提取业务约束（预算、延迟、精度要求）

### 2. 真实 AutoML 训练 (chat2model)
- 支持 XGBoost、LightGBM、RandomForest、LogisticRegression 等模型
- 自动超参数搜索（贝叶斯优化 + 随机搜索）
- 特征工程管道（自动编码、缩放、缺失值处理）
- 交叉验证和早停机制

### 3. 模型产物交付
- 可加载的模型权重文件（`.pkl`）
- 预处理管道（特征工程状态）
- 元数据（超参数、评估指标、特征重要性）
- 批量预测 API

## 🚀 快速开始

### 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 配置 LLM（推荐）

VibeML 使用 LLM 进行意图解析和任务规划。配置以下环境变量：

```bash
# 必需：API Key
export LLM_API_KEY="your-api-key"

# 可选：模型名称（默认 gpt-4o-mini）
export LLM_MODEL_NAME="gpt-4o"

# 可选：API 基础 URL（默认 OpenAI）
export LLM_BASE_URL="https://api.openai.com/v1"
```

支持任意 OpenAI 兼容的 API（如 Azure、第三方代理等）。

### 启动服务

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/` 打开控制台界面。

### 前端重构开发（React + pnpm）

前端重构工程位于 `web/`，采用 React + Vite + TypeScript + Zustand。

```bash
# 1) 安装依赖（仓库根目录）
pnpm install

# 2) 启动前端开发服务
pnpm --dir web dev
```

开发模式下：
- 新前端地址: `http://localhost:5173/`
- API 通过 Vite 代理到 `http://localhost:8000/api/*`

生产构建：

```bash
pnpm --dir web build
```

构建后产物在 `web/dist`，后端会自动挂载到 `http://localhost:8000/app-next/`（不影响现有 `http://localhost:8000/app/*` 旧页面）。

### API 文档

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📖 使用流程

### 1. 上传数据
```bash
curl -X POST "http://localhost:8000/api/data/upload" \
  -F "file=@your_data.csv" \
  -F "target_hint=churn"
```

### 2. 编译意图
```bash
curl -X POST "http://localhost:8000/api/intent/compile" \
  -H "Content-Type: application/json" \
  -d '{
    "user_goal": "预测用户是否会流失，宁可误报也别漏报",
    "must_keep": ["高价值客户优先"],
    "worst_errors": ["漏报比误报更糟"],
    "priority": "quality"
  }'
```

响应示例：
```json
{
  "objective_spec": {
    "task_family": "binary_classification",
    "primary_metric": "recall",
    "recommended_models": ["xgboost", "lightgbm", "random_forest"],
    "validation_strategy": "train_test_split"
  }
}
```

### 3. 启动训练
```bash
curl -X POST "http://localhost:8000/api/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "预测用户是否会流失",
    "dataset_id": "your_dataset_id",
    "target_column": "churn",
    "max_training_time": 300,
    "max_trials": 30
  }'
```

### 4. 查询状态
```bash
curl "http://localhost:8000/api/training/{job_id}"
```

### 5. 下载模型
```bash
# 下载模型权重
curl "http://localhost:8000/api/training/{job_id}/download/model" \
  -o model.pkl

# 下载预处理器
curl "http://localhost:8000/api/training/{job_id}/download/preprocessor" \
  -o preprocessor.pkl
```

### 6. 批量预测
```bash
curl -X POST "http://localhost:8000/api/predict/{job_id}" \
  -F "file=@new_data.csv"
```

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        前端界面                              │
│              (数据上传 / 意图输入 / 训练监控)                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      FastAPI 后端                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  DataManager │  │   Compiler   │  │ JobManager   │      │
│  │  - 上传/解析  │  │ - 意图编译   │  │ - 任务调度   │      │
│  │  - 类型推断   │  │ - Objective  │  │ - 状态管理   │      │
│  │  - 特征工程   │  │   Spec       │  │ - Checkpoint │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    AutoMLTrainer                             │
│  ├─ ModelRegistry: 模型注册表                                │
│  ├─ Hyperparameter Search: 超参数搜索                        │
│  ├─ Feature Engineering: 预处理管道                         │
│  └─ Model Persistence: 模型持久化                           │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
VibeML-Agent/
├── backend/
│   ├── main.py              # FastAPI 主入口
│   ├── compiler.py          # Objective Compiler (chat2objective)
│   ├── data_manager.py      # 数据管理
│   ├── trainer.py           # AutoML 训练引擎
│   └── requirements.txt
├── frontend/
│   └── index.html           # Web 控制台
├── tests/
│   └── test_api.py          # API 测试
└── README.md
```

## 🎯 支持的模型

| 模型 | 分类 | 回归 | 特点 |
|------|------|------|------|
| XGBoost | ✅ | ✅ | 高性能，支持缺失值 |
| LightGBM | ✅ | ✅ | 快速，大数据集友好 |
| RandomForest | ✅ | ✅ | 鲁棒，可解释性好 |
| LogisticRegression | ✅ | ❌ | 快速，低延迟 |
| ElasticNet | ❌ | ✅ | 正则化，稀疏数据 |

## 📊 支持的评估指标

**分类任务:**
- Accuracy, Precision, Recall, F1
- AUC-ROC
- Precision@K

**回归任务:**
- MAE, RMSE, MAPE
- R²

## 🔧 配置选项

训练请求支持以下配置：

```json
{
  "objective": "业务目标描述",
  "dataset_id": "数据集ID",
  "target_column": "目标列名",
  "max_training_time": 300,    // 最大训练时间（秒）
  "max_trials": 30,            // 超参搜索次数
  "priority": "quality",       // quality/latency/cost
  "worst_errors": ["漏报"]     // 错误偏好
}
```

## 🧪 运行测试

```bash
pytest tests/ -v
```

## 📝 技术亮点

1. **ObjectiveSpec**: 统一的任务规范中间表示，解耦需求层与执行层
2. **自动类型推断**: 智能识别数值、类别、文本、日期时间列
3. **特征工程管道**: sklearn ColumnTransformer + Pipeline，支持序列化
4. **进度实时推送**: 训练状态实时更新到前端
5. **模型可复用**: 下载的模型可直接用于生产环境推理

## 🔮 路线图

- [ ] 支持深度学习（PyTorch/TensorFlow）
- [ ] 时序预测专用管道
- [ ] 图神经网络（GNN）支持
- [ ] 在线学习/增量训练
- [ ] A/B 测试与模型版本管理
- [ ] 模型解释性（SHAP）

## 📄 License

MIT
