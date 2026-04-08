"""
示例: 蛋白质结构预测 - 展示如何使用 MLEngineerAgent

场景: 低质量实验数据，需要不确定性建模
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml_engineer import MLEngineerAgent, quick_train


def mock_llm_client():
    """
    模拟LLM客户端，用于演示
    实际使用时替换为真实的OpenAI/Claude客户端
    """
    class MockLLM:
        def generate(self, prompt: str) -> str:
            # 这里返回预定义的代码模板
            # 实际项目中调用真实LLM API
            pass
    
    return MockLLM()


def example_full_pipeline():
    """完整流程示例"""
    
    # 初始化Agent
    agent = MLEngineerAgent(
        llm_client=None,  # 实际使用时传入真实LLM客户端
        output_dir="./outputs",
        max_qa_attempts=3,
        max_opt_iterations=2,
        bo_budget_per_iter=10,
    )
    
    # 定义意图
    intent = """
    我想预测蛋白质的三维结构。
    我的训练数据来自实验测定，其中有很多低质量的样本。
    我希望模型能够：
    1. 对低质量样本降低权重
    2. 输出结构预测的同时给出不确定性估计
    3. 对不确定性高的区域能够识别出来
    我的计算资源有限，模型不要超过100M参数。
    """
    
    # 数据schema（可选，帮助LLM理解数据）
    data_schema = {
        "sequence": {"type": "string", "description": "氨基酸序列"},
        "coordinates": {"type": "array", "shape": [None, 3], "description": "3D坐标"},
        "experimental_resolution": {"type": "float", "description": "实验分辨率，越低越好"},
        "pdb_id": {"type": "string", "description": "PDB标识符"},
    }
    
    # 约束条件
    constraints = {
        "max_params": 100_000_000,  # 100M参数
        "max_training_hours": 24,
        "target_device": "single_gpu",  # 或 "multi_gpu", "cpu"
    }
    
    # 执行训练
    result = agent.train(
        intent=intent,
        domain="ai4science",
        data_schema=data_schema,
        constraints=constraints,
        budget=20,  # 20个BO trial
        time_budget_sec=7200,  # 2小时
        target_metric="val_rmsd",  # 蛋白质结构预测常用指标
        target_threshold=2.0,  # RMSD < 2Å 算合格
    )
    
    print("\n" + "="*60)
    print("TRAINING RESULT")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Best Score: {result.get('optimization', {}).get('best_score')}")
    print(f"Output Path: {result.get('output', {}).get('model_path')}")
    
    return result


def example_gnn_timeseries():
    """GNN + 时序示例：社交网络传播预测"""
    
    intent = """
    预测社交网络中信息的传播路径。
    考虑因素：
    1. 用户之间的关系强度会随时间衰减
    2. 不同话题的传播模式不同
    3. 大V用户的影响力比普通用户高很多
    需要处理冷启动问题（新用户没有历史）
    """
    
    result = quick_train(
        intent=intent,
        domain="gnn",
        budget=30,
        target_metric="val_auc",
    )
    
    return result


def example_revenue_forecasting():
    """时序预测示例：收入预测"""
    
    intent = """
    预测SaaS公司未来30天的收入。
    特殊需求：
    1. 要考虑节假日效应（春节、双11等）
    2. 新功能上线会有脉冲式影响
    3. 宁可高估也不要低估（为了现金流安全）
    4. 需要给出预测区间，不只是点估计
    """
    
    result = quick_train(
        intent=intent,
        domain="timeseries",
        budget=25,
        target_metric="val_crps",  # Continuous Ranked Probability Score
    )
    
    return result


def example_safe_rl():
    """强化学习示例：安全约束的推荐系统"""
    
    intent = """
    基于历史用户行为训练一个推荐策略。
    约束：
    1. 不能推荐与用户兴趣差异太大的商品（避免惊吓）
    2. 新用户的前10次推荐必须包含热门商品（保证质量）
    3. 同一类目连续推荐不超过3次
    4. 策略不能太复杂（线上推理延迟<50ms）
    """
    
    result = quick_train(
        intent=intent,
        domain="rl",
        constraints={"max_latency_ms": 50},
        budget=40,
        target_metric="val_cumulative_reward",
    )
    
    return result


if __name__ == "__main__":
    # 运行示例
    print("MLEngineerAgent Examples")
    print("="*60)
    
    # 注意：这些示例需要真实的LLM客户端才能运行
    # 这里是展示API设计和使用方式
    
    # result = example_full_pipeline()
    # result = example_gnn_timeseries()
    # result = example_revenue_forecasting()
    # result = example_safe_rl()
    
    print("\n这些示例展示了不同领域的使用方式。")
    print("实际运行时需要配置LLM客户端（OpenAI/Claude等）。")