# VibeML Research - 架构设计文档

## 核心目标

将**模糊的自然语言需求**转化为**符合具体业务的精细深度学习训练流**。

不同于传统 AutoML 的"配置选择"，我们让大模型扮演稀缺的人类 ML 研究员角色，直接**生成定制代码**（数据增强、损失函数、模型架构等），并通过**多层 QA** 和 **BO + ReAct 优化**确保正确性和性能。

## 架构概览

```
自然语言需求
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  MLEngineerAgent (核心入口)                                      │
│  ├─ 理解意图，协调各组件                                          │
│  └─ 跟踪任务状态，输出最终模型                                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ├──► codegen/ProgramGenerator (代码生成)
    │      ├─ Domain-specific prompt templates
    │      ├─ 生成: model.py, loss.py, data_pipeline.py
    │      └─ 生成: search_space.yaml (BO搜索空间)
    │
    ├──► codegen/QAPipeline (质量保证)
    │      ├─ Static Analyzer (语法、导入、安全、PyTorch模式)
    │      ├─ Unit Test Generator (形状、梯度、设备、序列化)
    │      ├─ Sandbox Executor (隔离环境冒烟测试)
    │      └─ Auto-Fix Loop (LLM修复，最多3次)
    │
    ├──► optimizer/HybridOptimizer (训练优化)
    │      ├─ Bayesian Optimizer (在search_space中搜索超参)
    │      ├─ ReAct Analyzer (启发式+LLM分析训练结果)
    │      └─ Program Evolution (根据分析结果进化代码)
    │
    └► 输出: 可部署模型 + 完整代码 + 实验报告
```

## 核心创新

### 1. 代码生成而非配置选择

传统 AutoML:
```python
# 从预定义选项中选择
model = select_from(["resnet18", "resnet50", "efficientnet"])
loss = select_from(["cross_entropy", "focal_loss"])
```

VibeML Research:
```python
# LLM根据需求直接生成代码
# "低质量样本多，需要不确定性建模"
↓
class EvidentialLoss(nn.Module):
    def forward(self, pred, target, sample_quality):
        # 根据样本质量调整权重
        uncertainty = self.evidence_to_uncertainty(pred)
        weighted_loss = base_loss * sample_quality * (1 + uncertainty)
        return weighted_loss
```

### 2. 多层 QA 确保正确性

| 层级 | 检查内容 | 失败处理 |
|------|---------|---------|
| **Static Analysis** | Python语法、AST解析、导入白名单、危险模式检测 | LLM修复代码 |
| **Unit Tests** | 输入输出形状、梯度传播、CPU/GPU兼容性、序列化 | LLM修复代码 |
| **Sandbox Smoke** | 模块导入、前向传播、损失计算、反向传播 | LLM修复代码 |

**关键设计**: 所有 QA 都在**沙箱**中执行，失败不会污染主环境。

### 3. BO + ReAct 混合优化

```
Iteration 1:
  ├─ BO搜索20个超参配置 → 最佳val_f1=0.82
  └─ ReAct分析: "验证损失>训练损失1.5倍，可能过拟合"
      └─ 建议: "增加dropout，减少模型容量"

Iteration 2 (代码进化):
  ├─ LLM生成新代码 (加入更多正则化)
  ├─ QA验证通过
  └─ BO搜索20个配置 → 最佳val_f1=0.88 ✓
```

### 4. 领域特定模板

不同领域的"精细训练流"完全不同：

- **AI4Science**: 物理约束、不确定性量化、多尺度建模
- **GNN**: 图采样、时序演化、冷启动处理
- **时序预测**: 季节性分解、节假日效应、概率区间
- **RL**: 安全约束、策略正则、离线学习稳定性

## 模块详解

### 1. codegen/generator.py - ProgramGenerator

职责: 将自然语言意图转换为可执行代码。

输入:
```python
{
    "intent": "预测蛋白质结构，低质量样本多，需要不确定性建模",
    "domain": "ai4science",
    "data_schema": {...},
    "constraints": {"max_params": 100_000_000}
}
```

输出:
```python
GeneratedProgram(
    model_code="class EvidentialTransformer(...)",
    loss_code="class UncertaintyAwareLoss(...)",
    data_pipeline_code="class ProteinDataModule(...)",
    train_loop_code="class Trainer(...)",
    search_space={
        "model": {"num_layers": {"type": "int", "low": 4, "high": 12}},
        "loss": {"uncertainty_weight": {"type": "float", "low": 0.01, "high": 1.0}},
    },
    dependencies=["torch", "torch_geometric"],
)
```

### 2. codegen/static_analyzer.py - StaticAnalyzer

职责: 不执行代码，静态检查正确性。

检查项:
- **Syntax**: AST能否解析
- **Imports**: 是否在白名单中（禁止`eval`, `exec`, `subprocess`, `socket`等）
- **Security**: 危险模式检测
- **PyTorch**: 是否继承`nn.Module`，是否有`forward`方法

### 3. codegen/test_generator.py - UnitTestGenerator

职责: 自动生成单元测试。

测试类型:
- **Shape Tests**: `input [B, C, H, W]` → `output [B, num_classes]`
- **Gradient Tests**: `loss.backward()` 后参数有梯度且无NaN
- **Device Tests**: 模型能在CPU和CUDA间移动
- **Serialization Tests**: `state_dict`保存加载后输出一致

### 4. codegen/sandbox.py - SandboxExecutor

职责: 隔离执行代码，验证能正确运行。

使用 `subprocess` + `resource` 限制:
- 内存限制 (默认2GB)
- CPU时间限制
- 网络禁用
- 文件系统隔离（临时目录）

冒烟测试阶段:
1. Import test: 所有模块能导入
2. Forward pass: 模型能前向传播
3. Loss computation: 损失能计算
4. Backward pass: 梯度能反向传播
5. Data loading: 数据管道能加载一个batch

### 5. optimizer.py - HybridOptimizer

职责: BO搜索超参 + ReAct分析改进。

算法:
```python
for iteration in range(max_iterations):
    # Phase 1: BO Search
    best_config, best_score = bo.optimize(program, budget=20)
    
    # Phase 2: ReAct Analysis
    analysis = analyzer.analyze(training_history)
    
    if analysis.confidence > 0.7 and "structural_change" in analysis.actions:
        # 需要代码层面的修改
        program = generator.evolve(program, analysis.suggestions)
        qa.validate(program)  # 重新验证
    else:
        # 超参优化已足够
        break
```

### 6. ml_engineer.py - MLEngineerAgent

职责: 协调全流程，提供简洁API。

```python
agent = MLEngineerAgent(llm_client=client)

result = agent.train(
    intent="预测用户流失，宁可误报不要漏报",
    domain="general",
    budget=60,  # BO trial数
    target_metric="val_recall",
)

# result包含:
# - 最佳模型权重
# - 完整可复现代码
# - 实验记录和超参配置
# - 训练曲线和评估报告
```

## 数据流

```
用户输入:
  intent="蛋白质结构预测，低质量样本多"
  domain="ai4science"

        ↓

Step 1: 代码生成
  LLM(prompt_template[ai4science] + intent)
  → GeneratedProgram
    - model.py (EvidentialTransformer)
    - loss.py (UncertaintyAwareLoss)
    - search_space.yaml

        ↓

Step 2: QA验证
  ├─ StaticAnalyzer: 语法OK，导入OK
  ├─ UnitTestGenerator: 生成20个测试
  │   └─ pytest运行，全部通过
  └─ SandboxExecutor: 5个冒烟测试全部通过

        ↓

Step 3: BO优化
  SearchSpaceBuilder将yaml转换为skopt空间
  
  for trial in range(budget):
      config = bo.suggest()  # GP推荐
      metrics = sandbox.execute_training(config)  # 实际训练
      bo.observe(config, metrics.val_rmsd)
  
  best_config, best_score = bo.get_best()

        ↓

Step 4: ReAct分析 (如果性能不佳)
  analyzer.analyze(training_history)
  → "验证损失远高于训练损失，建议增加正则化"
  
  generator.evolve(program, "增加dropout")
  → 新的GeneratedProgram
  
  回到Step 2重新QA，然后继续BO

        ↓

输出:
  - outputs/{job_id}/final/model.py
  - outputs/{job_id}/final/loss.py
  - outputs/{job_id}/final/checkpoint.pt
  - outputs/{job_id}/results.json
```

## 错误处理

| 错误类型 | 处理策略 |
|---------|---------|
| 代码生成失败 | 重试3次，增加更详细的prompt说明 |
| 静态分析失败 | LLM修复代码，保留已通过的部分 |
| 单元测试失败 | 定位失败测试，针对性修复 |
| 沙箱执行失败 | 检查依赖、内存、超时，修复后重试 |
| BO训练失败 | 标记该配置为失败，继续下一个trial |
| ReAct建议无效 | 降低置信度阈值，或人工介入 |

## 扩展性

### 添加新领域

1. 在 `DOMAIN_TEMPLATES` 中添加领域提示词
2. 提供2-3个高质量的 few-shot 示例
3. 定义领域特定的评估指标

### 添加新的QA检查

1. 在 `StaticAnalyzer` 中添加新的检查方法
2. 在 `UnitTestGenerator` 中添加新的测试类型
3. 更新 `QAPipeline.validate()` 的调用顺序

### 替换BO库

当前使用 scikit-optimize，可以轻松替换为:
- Optuna (更现代，支持多目标)
- Ax (Facebook出品，支持并行)
- Ray Tune (分布式)

只需重新实现 `BayesianOptimizer` 类接口。

## 安全考虑

1. **代码生成安全**
   - 导入白名单机制
   - 禁止`eval`, `exec`, `__import__`
   - 禁止网络操作

2. **沙箱隔离**
   - 独立的Python进程
   - 内存/CPU限制
   - 临时文件系统

3. **资源保护**
   - GPU显存监控
   - 训练超时强制终止
   - 自动checkpoint恢复

## 性能优化

1. **缓存机制**
   - 相同intent的生成结果缓存
   - QA通过的程序缓存
   - BO的surrogate model缓存

2. **并行化**
   - QA阶段的静态分析和单元测试可并行
   - BO的多个trial可并行（多GPU）

3. **早停**
   - 训练过程中的早停
   - BO的自动收敛检测
   - ReAct的不可能任务检测

## 部署建议

### 开发环境
```bash
pip install -r requirements.txt
python examples/protein_folding_example.py
```

### 生产环境
- LLM API: 建议使用 Claude 3.5 Sonnet 或 GPT-4o
- GPU: 至少1张A100用于BO trial
- 存储: 保存所有中间产物用于追溯
- 监控: 记录每个stage的耗时和成功率

## 未来方向

1. **多模态**: 支持图像、文本、表格混合输入
2. **自动蒸馏**: 大模型训好后自动生成小模型
3. **A/B测试**: 自动部署和线上对比
4. **持续学习**: 新数据到来时增量更新