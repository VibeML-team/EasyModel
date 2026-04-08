"""
Optimizer - BO + ReAct 混合优化器

整合贝叶斯优化和ReAct推理-行动循环，实现高效训练。

架构:
1. LLM生成初始程序 + Search Space
2. QA Pipeline验证程序正确性
3. BO在Search Space中搜索最优超参数
4. 如果性能不佳，触发ReAct进行结构优化
5. 迭代直到预算耗尽
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

# 尝试导入BO库
try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real, Space
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class OptimizationResult:
    """优化结果"""
    status: str  # success, failed, timeout
    best_program: "GeneratedProgram" | None
    best_config: dict[str, Any]
    best_score: float
    all_trials: list[dict]
    total_duration_sec: float
    iterations: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "best_config": self.best_config,
            "best_score": self.best_score,
            "total_trials": len(self.all_trials),
            "total_duration_sec": self.total_duration_sec,
            "iterations": self.iterations,
        }


@dataclass
class ReActState:
    """ReAct状态"""
    iteration: int
    observations: list[dict]  # 每次训练观察到的现象
    actions_taken: list[dict]  # 已采取的行动
    current_program: "GeneratedProgram"
    current_search_space: dict[str, Any]


class SearchSpaceBuilder:
    """从YAML/字典构建BO搜索空间"""
    
    @staticmethod
    def from_dict(space_dict: dict[str, Any]) -> list:
        """
        将字典转换为skopt空间定义
        
        输入格式:
        {
            "model": {
                "num_layers": {"type": "int", "low": 2, "high": 8},
                "hidden_dim": {"type": "choice", "values": [256, 512, 1024]},
            },
            "optimizer": {
                "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            }
        }
        """
        dimensions = []
        
        for category, params in space_dict.items():
            for param_name, config in params.items():
                full_name = f"{category}.{param_name}"
                param_type = config.get("type", "float")
                
                if param_type == "int":
                    dim = Integer(
                        low=config["low"],
                        high=config["high"],
                        name=full_name,
                    )
                elif param_type == "float":
                    dim = Real(
                        low=config["low"],
                        high=config["high"],
                        prior="log-uniform" if config.get("log") else "uniform",
                        name=full_name,
                    )
                elif param_type == "choice":
                    dim = Categorical(
                        categories=config["values"],
                        name=full_name,
                    )
                else:
                    continue
                
                dimensions.append(dim)
        
        return dimensions
    
    @staticmethod
    def config_to_dict(config_list: list, dimensions: list) -> dict:
        """将BO的列表配置转换为嵌套字典"""
        result = {}
        
        for value, dim in zip(config_list, dimensions):
            name = dim.name  # e.g., "model.num_layers"
            parts = name.split('.')
            
            # 构建嵌套结构
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return result


class BayesianOptimizer:
    """贝叶斯优化器封装"""
    
    def __init__(
        self,
        space: list,
        n_initial_points: int = 5,
        acquisition_function: str = "EI",  # Expected Improvement
        random_state: int = 42,
    ):
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        self.space = space
        self.dimensions = space
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition_function
        self.random_state = random_state
        
        # 历史记录
        self.X = []  # 配置
        self.y = []  # 分数
        self.costs = []  # 实际花费（用于多保真优化）
        
    def suggest(self) -> dict:
        """建议下一个配置"""
        if len(self.X) < self.n_initial_points:
            # 随机采样初始点
            config = [dim.rvs(1)[0] for dim in self.dimensions]
        else:
            # 使用GP建议
            from skopt import gp_minimize
            
            # 构建目标函数（基于历史）
            def dummy_objective(x):
                return 0
            
            # 使用历史数据拟合GP
            res = gp_minimize(
                dummy_objective,
                self.dimensions,
                x0=self.X,
                y0=self.y,
                n_calls=1,
                n_initial_points=0,
                acq_func=self.acquisition,
                random_state=self.random_state,
            )
            
            config = res.x_iters[-1] if res.x_iters else [
                dim.rvs(1)[0] for dim in self.dimensions
            ]
        
        return SearchSpaceBuilder.config_to_dict(config, self.dimensions)
    
    def observe(self, config: dict, score: float, cost: float = 0):
        """记录观察结果"""
        # 将字典转换为列表
        config_list = []
        for dim in self.dimensions:
            parts = dim.name.split('.')
            value = config
            for p in parts:
                value = value[p]
            config_list.append(value)
        
        self.X.append(config_list)
        self.y.append(score)
        self.costs.append(cost)
    
    def get_best(self) -> tuple[dict, float]:
        """获取当前最佳配置和分数"""
        if not self.y:
            return {}, float('-inf')
        
        best_idx = np.argmax(self.y)
        best_config = SearchSpaceBuilder.config_to_dict(
            self.X[best_idx], self.dimensions
        )
        return best_config, self.y[best_idx]


class ReActAnalyzer:
    """
    ReAct分析器 - 分析训练结果并建议改进
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    def analyze(
        self,
        state: ReActState,
        training_history: dict,
    ) -> dict:
        """
        分析训练历史，建议下一步行动
        
        Returns:
            {
                "observation": "观察到的现象",
                "diagnosis": "根因分析",
                "suggested_actions": ["行动1", "行动2"],
                "confidence": 0.8,
                "should_continue": True,
            }
        """
        # 启发式规则（快速路径）
        heuristics = self._apply_heuristics(training_history)
        
        # 如果启发式规则有强建议，直接返回
        if heuristics.get("strong_signal"):
            return heuristics
        
        # 否则使用LLM分析
        if self.llm_client:
            return self._llm_analysis(state, training_history)
        
        return heuristics
    
    def _apply_heuristics(self, history: dict) -> dict:
        """应用启发式规则"""
        actions = []
        confidence = 0.5
        
        # 检查过拟合
        if history.get("train_loss") and history.get("val_loss"):
            train_loss = history["train_loss"][-1] if isinstance(history["train_loss"], list) else history["train_loss"]
            val_loss = history["val_loss"][-1] if isinstance(history["val_loss"], list) else history["val_loss"]
            
            if val_loss > train_loss * 1.5:
                actions.append("增加dropout_rate")
                actions.append("增加weight_decay")
                actions.append("减少模型容量")
                confidence = 0.8
                return {
                    "observation": f"验证损失({val_loss:.4f})显著高于训练损失({train_loss:.4f})，可能存在过拟合",
                    "diagnosis": "模型在训练集上表现过好，泛化能力不足",
                    "suggested_actions": actions,
                    "confidence": confidence,
                    "should_continue": True,
                    "strong_signal": True,
                }
        
        # 检查梯度爆炸
        if history.get("grad_norm"):
            grad_norm = history["grad_norm"]
            if isinstance(grad_norm, list):
                grad_norm = max(grad_norm)
            if grad_norm > 10:
                actions.append("启用gradient_clipping")
                actions.append("降低learning_rate")
                confidence = 0.9
                return {
                    "observation": f"梯度范数({grad_norm:.2f})过大，存在梯度爆炸风险",
                    "diagnosis": "学习率可能过高或数据存在异常值",
                    "suggested_actions": actions,
                    "confidence": confidence,
                    "should_continue": True,
                    "strong_signal": True,
                }
        
        # 检查训练停滞
        if history.get("train_loss"):
            losses = history["train_loss"] if isinstance(history["train_loss"], list) else [history["train_loss"]]
            if len(losses) > 5:
                recent_change = abs(losses[-1] - losses[-5]) / (abs(losses[-5]) + 1e-8)
                if recent_change < 0.01:
                    actions.append("提高learning_rate")
                    actions.append("更换优化器(Adam -> SGD with momentum)")
                    confidence = 0.7
                    return {
                        "observation": "训练损失近期变化很小，可能陷入局部最优或学习率过小",
                        "diagnosis": "优化过程停滞",
                        "suggested_actions": actions,
                        "confidence": confidence,
                        "should_continue": True,
                        "strong_signal": True,
                    }
        
        return {
            "observation": "训练过程正常",
            "diagnosis": "无明显异常",
            "suggested_actions": [],
            "confidence": confidence,
            "should_continue": False,
        }
    
    def _llm_analysis(self, state: ReActState, history: dict) -> dict:
        """使用LLM进行分析"""
        prompt = f"""你是一个深度学习训练专家。请分析以下训练结果并建议改进措施。

当前迭代: {state.iteration}

训练历史指标:
```json
{json.dumps(history, indent=2)}
```

已采取的行动:
{json.dumps(state.actions_taken, indent=2)}

请分析:
1. 观察到了什么现象？
2. 可能的原因是什么？
3. 建议采取什么行动？（最多3个）
4. 置信度多高？（0-1）
5. 是否应该继续优化？

按JSON格式输出:
{{
    "observation": "...",
    "diagnosis": "...",
    "suggested_actions": ["..."],
    "confidence": 0.8,
    "should_continue": true
}}
"""
        
        try:
            response = self.llm_client.generate(prompt)
            # 提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # 失败时返回保守建议
        return {
            "observation": "无法分析训练结果",
            "diagnosis": "未知",
            "suggested_actions": [],
            "confidence": 0.0,
            "should_continue": False,
        }


class HybridOptimizer:
    """
    BO + ReAct 混合优化器
    
    算法流程:
    1. 初始化：LLM生成程序P0和搜索空间S0
    2. 阶段A：BO在Si中搜索最优配置（花费60%预算）
    3. 检查：如果性能满意，返回最优配置
    4. 阶段B：ReAct分析，LLM生成新程序Pi+1和搜索空间Si+1
    5. 重复直到预算耗尽
    """
    
    def __init__(
        self,
        qa_pipeline: "QAPipeline",
        sandbox: "SandboxExecutor",
        llm_client=None,
        max_iterations: int = 3,
        bo_budget_per_iter: int = 20,
        target_metric: str = "val_f1",
        target_threshold: float | None = None,
    ):
        self.qa_pipeline = qa_pipeline
        self.sandbox = sandbox
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.bo_budget_per_iter = bo_budget_per_iter
        self.target_metric = target_metric
        self.target_threshold = target_threshold
        
        self.analyzer = ReActAnalyzer(llm_client)
        
    def optimize(
        self,
        initial_program: "GeneratedProgram",
        total_budget: int = 60,  # 总BO trial数
        time_budget_sec: int = 3600,
    ) -> OptimizationResult:
        """
        执行完整优化流程
        """
        start_time = time.time()
        
        current_program = initial_program
        all_trials = []
        best_score = float('-inf')
        best_config = {}
        
        budget_per_iteration = total_budget // self.max_iterations
        
        for iteration in range(self.max_iterations):
            # 检查时间预算
            if time.time() - start_time > time_budget_sec:
                break
            
            print(f"\n{'='*60}")
            print(f"Optimization Iteration {iteration + 1}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # Phase 1: BO Search
            bo_result = self._bo_search(
                current_program,
                budget=min(budget_per_iteration, total_budget - len(all_trials)),
                time_remaining=time_budget_sec - (time.time() - start_time),
            )
            
            all_trials.extend(bo_result["trials"])
            
            # 更新最佳结果
            if bo_result["best_score"] > best_score:
                best_score = bo_result["best_score"]
                best_config = bo_result["best_config"]
            
            # 检查是否达到目标
            if self.target_threshold and best_score >= self.target_threshold:
                print(f"\n✓ 达到目标阈值 {self.target_threshold}")
                break
            
            # Phase 2: ReAct Analysis (如果不是最后一轮)
            if iteration < self.max_iterations - 1:
                should_evolve = self._react_analysis(
                    current_program,
                    bo_result["trials"],
                    iteration,
                )
                
                if should_evolve:
                    # 生成新程序
                    new_program = self._evolve_program(
                        current_program,
                        bo_result["trials"],
                    )
                    
                    # QA验证
                    qa_result = self.qa_pipeline.validate(new_program, auto_fix=True)
                    
                    if qa_result.passed:
                        current_program = qa_result.program
                        print(f"\n✓ 程序已进化，进入下一轮优化")
                    else:
                        print(f"\n✗ 新程序验证失败，保持当前程序")
        
        total_duration = time.time() - start_time
        
        return OptimizationResult(
            status="success" if best_score > float('-inf') else "failed",
            best_program=current_program,
            best_config=best_config,
            best_score=best_score,
            all_trials=all_trials,
            total_duration_sec=total_duration,
            iterations=iteration + 1,
        )
    
    def _bo_search(
        self,
        program: "GeneratedProgram",
        budget: int,
        time_remaining: float,
    ) -> dict:
        """
        执行BO搜索
        """
        # 构建搜索空间
        space = SearchSpaceBuilder.from_dict(program.search_space)
        
        if not space:
            return {"trials": [], "best_score": float('-inf'), "best_config": {}}
        
        bo = BayesianOptimizer(
            space=space,
            n_initial_points=min(5, budget // 3),
        )
        
        trials = []
        
        for trial_idx in range(budget):
            # 建议配置
            config = bo.suggest()
            
            print(f"  Trial {trial_idx + 1}/{budget}: {config}")
            
            # 执行训练
            start = time.time()
            result = self.sandbox.execute_training(
                program,
                config,
                timeout=min(600, int(time_remaining / (budget - trial_idx))),
            )
            duration = time.time() - start
            
            # 提取分数
            score = self._extract_score(result)
            
            print(f"    -> Score: {score:.4f} (took {duration:.1f}s)")
            
            # 记录
            trial = {
                "iteration": trial_idx,
                "config": config,
                "score": score,
                "duration_sec": duration,
                "status": result.get("status", "unknown"),
            }
            trials.append(trial)
            
            # 告知BO
            bo.observe(config, score, duration)
        
        best_config, best_score = bo.get_best()
        
        return {
            "trials": trials,
            "best_score": best_score,
            "best_config": best_config,
        }
    
    def _react_analysis(
        self,
        program: "GeneratedProgram",
        trials: list[dict],
        iteration: int,
    ) -> bool:
        """
        ReAct分析，决定是否需要结构修改
        
        Returns:
            是否需要进行程序进化
        """
        # 构建状态
        state = ReActState(
            iteration=iteration,
            observations=[],
            actions_taken=[],
            current_program=program,
            current_search_space=program.search_space,
        )
        
        # 构建训练历史
        history = {
            "trials": len(trials),
            "best_score": max((t["score"] for t in trials), default=0),
            "mean_score": sum(t["score"] for t in trials) / len(trials) if trials else 0,
            "configurations": [t["config"] for t in trials],
        }
        
        # 分析
        analysis = self.analyzer.analyze(state, history)
        
        print(f"\nReAct Analysis:")
        print(f"  Observation: {analysis['observation']}")
        print(f"  Diagnosis: {analysis['diagnosis']}")
        print(f"  Suggested Actions: {analysis.get('suggested_actions', [])}")
        print(f"  Confidence: {analysis.get('confidence', 0):.2f}")
        
        # 如果有高置信度的结构修改建议，触发进化
        actions = analysis.get("suggested_actions", [])
        high_confidence = analysis.get("confidence", 0) > 0.7
        
        structural_keywords = ["dropout", "层", "layer", "capacity", "架构", "architecture"]
        needs_structural_change = any(
            keyword in action for action in actions for keyword in structural_keywords
        )
        
        return high_confidence and needs_structural_change
    
    def _evolve_program(
        self,
        program: "GeneratedProgram",
        trials: list[dict],
    ) -> "GeneratedProgram":
        """
        基于BO结果进化程序
        """
        # 找出最佳和最差配置的模式
        sorted_trials = sorted(trials, key=lambda t: t["score"], reverse=True)
        best_configs = sorted_trials[:3]
        worst_configs = sorted_trials[-3:] if len(sorted_trials) > 3 else []
        
        prompt = f"""基于以下训练结果，请优化深度学习代码。

原始需求: {program.intent}

当前最佳配置及分数:
{json.dumps([{"config": t["config"], "score": t["score"]} for t in best_configs], indent=2)}

表现较差的配置:
{json.dumps([{"config": t["config"], "score": t["score"]} for t in worst_configs], indent=2)}

当前模型代码:
```python
{program.model_code}
```

当前损失函数:
```python
{program.loss_code}
```

请根据训练结果优化代码:
1. 如果深层模型表现更好，考虑增加深度
2. 如果正则化参数大的配置更好，考虑增强正则化
3. 如果某些超参数区域表现差，调整搜索空间避开这些区域

输出完整的优化后代码，使用相同的格式（```python filename.py）
"""
        
        # 使用generator修复代码
        generator = ProgramGenerator(self.llm_client, domain=program.domain)
        
        # 模拟LLM调用
        if self.llm_client:
            response = self.llm_client.generate(prompt)
            new_program = generator._parse_response(response, program.intent)
            return new_program
        
        # 如果没有LLM，简单调整搜索空间
        return program
    
    def _extract_score(self, result: dict) -> float:
        """从训练结果中提取分数"""
        if result.get("status") != "success":
            return float('-inf')
        
        metrics = result.get("metrics", {})
        
        # 尝试获取目标metric
        if self.target_metric in metrics:
            return float(metrics[self.target_metric])
        
        # 回退到常见metric
        for key in ["val_f1", "val_accuracy", "val_auc", "val_r2"]:
            if key in metrics:
                return float(metrics[key])
        
        # 默认返回0
        return 0.0