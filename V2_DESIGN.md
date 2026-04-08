# VibeML V2 - 意图驱动训练设计

## 核心理念

**"用户介入意图层，系统执行机制层"**

不是把训练细节（lr、batch size、optimizer）暴露给用户，而是把会影响业务语义和最终效果的关键决策做成可观测、可介入。

## 六个用户可介入点

### 0. 意图澄清 (新增)

**触发条件：** 当系统检测到用户意图过于模糊时

**检测维度：**
- 描述长度是否足够（< 20字符视为模糊）
- 目标变量是否明确（缺少"预测/判断/识别"等关键词）
- 错误偏好是否说明
- 质量/速度优先级是否明确
- 数据信息是否提供

**系统行为：**
```
POST /v2/compile
→ 返回 intent_card.clarification:
  - is_ambiguous: true/false
  - ambiguity_score: 0-1
  - suggested_questions: [...]
  - can_proceed: false（模糊度过高时阻止继续）
```

**引导式追问示例：**
- "请详细描述您的业务场景：这个模型要解决什么具体问题？"
- "如果模型可能犯错，您更不能接受哪种情况？"
  - 宁可误报
  - 宁可漏报
  - 两者都要避免
- "您更看重以下哪个方面？"
  - 预测精度越高越好
  - 响应速度越快越好
  - 平衡

**用户回答后：**
```
POST /v2/clarify
→ 系统更新 intent_card
→ 重新评估模糊性
→ 可以继续或继续追问
```

**代码：** `backend/v2_intent_compiler.py::AmbiguityDetection, ClarificationQuestion`

### 1. 需求编译卡 (IntentCard)

**用户可观测：**
- 系统理解的任务类型
- 系统识别出的必须保留/可以变化的元素
- 系统的推理过程和置信度
- 不确定的地方（需要用户澄清）

**用户可介入：**
- 修改目标描述
- 增删必须保留的元素
- 调整质量/速度/成本优先级
- 选择错误偏好（宁可误报/宁可漏报）

**代码：** `backend/v2_intent_compiler.py::IntentCard`

### 2. 数据与样本构造卡 (DataConstructionCard)

**用户可观测：**
- 数据概览（总量、训练/验证分布）
- 数据质量问题
- 自动配对结果

**用户可介入：**
- 勾选排除不该进训练的样本
- 标记高优先级样本
- 修正配对关系
- 定义"允许的变化"和"禁止的变化"
- 选择增强风格（保守/标准/激进）

**代码：** `backend/v2_intent_compiler.py::DataConstructionCard`

### 3. 训练方案卡 (TrainingPlanCard)

**用户可观测：**
- 系统选择的训练路线（微调/蒸馏/配对编辑等）
- 选择理由
- 主要和次要优化目标

**用户可介入：**
- 质量/速度/成本偏好
- 部署目标（云端/边缘/移动端）

**代码：** `backend/v2_intent_compiler.py::TrainingPlanCard`

### 4. 训练中的观测 (TrainingObservation)

**用户可观测：**
- 当前阶段和进度
- 核心指标曲线
- 当前最佳 checkpoint
- 风险提示（过拟合、模式塌缩等）
- 中间结果预览

**用户可介入：**
- 暂停/恢复/停止
- 选择某个 checkpoint 作为候选
- 对中间结果打"更像/不像"

**新增：中间 Checkpoint 下载**
```
GET  /v2/train/checkpoints/{job_id}                    → 获取 checkpoint 列表
GET  /v2/train/checkpoints/{job_id}/{id}/download      → 下载指定 checkpoint
POST /v2/train/pause-and-download/{job_id}           → 暂停并获取最新 checkpoint
```

**使用场景：**
1. 训练进行中，用户点击"暂停"
2. 系统显示当前可用的 checkpoint 列表
3. 用户可以下载任意 checkpoint 进行本地测试
4. 测试满意可提前结束训练，不满意可恢复继续训练

**代码：** `backend/v2_training_controller.py::TrainingObservation, CheckpointInfo`

### 5. 验收与权重定版 (AcceptanceCard)

**用户可观测：**
- 候选 checkpoint 列表（带指标对比）
- 各版本在不同维度的表现

**用户可介入：**
- 选择最终 checkpoint
- 对各版本提供反馈
- 基于不满意点发起二次训练

**代码：** `backend/v2_intent_compiler.py::AcceptanceCard`

## 机制层隐藏

用户**不直接操作**以下内容（系统自动从意图层映射）：

```python
# 这些都在 MechanismConfig 中，由系统自动生成
- model_family, model_size
- use_lora, lora_rank
- optimizer, learning_rate, batch_size
- epochs, warmup_steps
- augmentation_policy (技术细节)
- loss_composition (加权细节)
```

映射示例：
- `quality_speed_cost=QUALITY_FIRST` → `model_size=large`, `epochs=20`
- `quality_speed_cost=EDGE_DEPLOY` → `model_size=small`, `use_lora=True`, `lora_rank=4`
- `augmentation_style=CONSERVATIVE` → 保守的数据增强策略
- `error_preference=PREFER_FN` → 增加漏报惩罚的 loss 权重

## API 设计

### 完整流程（含澄清和中间下载）

```
POST /v2/compile                                    → 编译意图，返回三张卡 + 模糊性检测
POST /v2/clarify                                    → 回答追问，更新意图卡（可多次调用）
POST /v2/refine                                     → 用户主动修改卡片
POST /v2/train/start                                → 开始训练
GET  /v2/train/status                               → 查询进度
GET  /v2/train/checkpoints/{job_id}                 → 获取中间 checkpoint 列表
GET  /v2/train/checkpoints/{job_id}/{id}/download   → 下载指定 checkpoint
POST /v2/train/pause-and-download/{job_id}          → 暂停并获取最新 checkpoint
POST /v2/train/finalize                             → 生成候选
POST /v2/train/select                               → 选择最终版本
```

### 模糊性检测响应示例

```json
{
  "intent_card": {
    "task": { "type": "unknown", "confidence": 0.3 },
    "clarification": {
      "is_ambiguous": true,
      "ambiguity_score": 0.8,
      "missing_elements": ["目标变量", "错误偏好"],
      "unclear_aspects": ["描述过短，难以准确理解任务"],
      "suggested_questions": [
        {
          "id": "q_target",
          "text": "您希望模型预测或判断什么？",
          "type": "open",
          "context": "需要明确模型输出的目标变量",
          "priority": 0
        },
        {
          "id": "q_error_pref",
          "text": "如果模型可能犯错，您更不能接受哪种情况？",
          "type": "multiple_choice",
          "options": ["宁可误报", "宁可漏报", "两者都要避免"],
          "context": "不同的错误偏好会影响模型优化目标",
          "priority": 1
        }
      ],
      "can_proceed": false
    }
  },
  "can_proceed": false
}
```

### 快速训练（最小介入）

```
POST /v2/quick-train  → 只提供意图和几个高层选择
```

## 前端设计

访问 `/app/v2.html` 体验卡片式交互：

### 流程步骤

1. **需求编译页** - 输入意图
   - 系统检测模糊性
   - 如模糊度过高，显示警告并生成追问
   - 用户回答追问后，系统重新评估
   - 澄清完成后显示"确认并继续"按钮

2. **数据构造页** - 配置样本和变化规则
3. **训练监控页** - 进度条、风险提示、中间预览
4. **验收定版页** - Checkpoint 对比选择

### 澄清交互设计

```
┌─────────────────────────────────────────┐
│ ⚠️ 意图需要澄清                          │
│                                         │
│ 系统检测到您的需求描述有些模糊            │
│ 模糊度: 80%                             │
│                                         │
│ 缺失的关键信息：                         │
│ • 目标变量                              │
│ • 错误偏好                              │
│                                         │
│ 澄清进度: ●●○○ 2/4                      │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ 问题 1: 您希望模型预测什么？         │ │
│ │ 💡 需要明确模型输出的目标变量        │ │
│ │ [请输入...]                         │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ 问题 2: 如果模型犯错，您更不能接受？ │ │
│ │ ⚪ 宁可误报                         │ │
│ │ ⚪ 宁可漏报                         │ │
│ │ ⚪ 两者都要避免                     │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ [提交回答]                              │
└─────────────────────────────────────────┘
```

## 代码结构

```
backend/
├── v2_intent_compiler.py      # 意图编译器（生成卡片）
├── v2_training_controller.py  # 训练控制器（卡片→机制→训练）
├── v2_api.py                  # V2 API 路由
└── main.py                    # 主应用（挂载 V2 路由）

frontend/
└── v2.html                    # V2 前端界面
```

## 与 V1 的区别

| 维度 | V1 | V2 |
|------|-----|-----|
| 抽象层级 | 参数级别 | 意图级别 |
| 用户输入 | max_trials, max_training_time | "宁可误报也别漏报" |
| 系统输出 | ObjectiveSpec | IntentCard + DataConstructionCard + TrainingPlanCard |
| 介入方式 | 改数字 | 改语义约束 |
| 目标用户 | 懂 ML 的工程师 | 业务人员 |

## 后续扩展

### 高级模式

可以开放机制层给专家用户：

```python
if user_level == "expert":
    plan_card.to_user_friendly_dict(expose_mechanism=True)
    # 暴露 _mechanism_config 中的具体参数
```

### 持续学习

- 用户对中间结果的反馈 → 触发 ReAct 调整
- 标记"这类偏差不能接受" → 增加对应约束

### 多模态支持

当前设计已经预留了扩展空间：
- `DataConstructionCard.allowed_variations` - 适合图像/文本/音频的不同语义
- `IntentCard.must_keep` - 可以是"结构"、"格式"、"风格"等
