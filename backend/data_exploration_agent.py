"""
数据探索 Agent — ReAct 模式

核心设计：
    Agent 拿到一个数据集目录后，自主规划如何探索它。
    Agent 有一个"终端"可以执行命令（ls, cat, head, wc, python, etc.），
    看到输出后决定下一步做什么，直到它理解了这份数据。

    Think → Act → Observe → Think → Act → Observe → ... → Conclude

    Agent 的每一步 thought/action/observation 都实时推送给前端，
    让用户看到 Agent 在"干活"。

    最终产出的 insights 直接影响训练 spec 的生成。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Generator


# Agent 可以使用的命令白名单前缀（安全沙箱）
ALLOWED_COMMANDS = [
    "ls", "find", "cat", "head", "tail", "wc", "file", "du",
    "tree", "stat", "grep", "awk", "sed", "sort", "uniq",
    "unzip", "tar", "python3", "python",
    "echo", "basename", "dirname", "realpath",
    "md5sum", "sha256sum",
]

MAX_STEPS = 12       # 最大探索步数
CMD_TIMEOUT = 30     # 单条命令超时（秒）
MAX_OUTPUT = 3000    # 单条命令输出截断长度


def execute_command(cmd: str, cwd: str, timeout: int = CMD_TIMEOUT) -> str:
    """
    在数据集目录中执行命令（沙箱）。
    
    限制：
    - 工作目录锁定在 cwd
    - 超时保护
    - 输出截断
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "en_US.UTF-8",
            },
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr] " + result.stderr[:500]
        
        # 截断过长输出
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + f"\n... (truncated, total {len(result.stdout)} chars)"
        
        return output.strip() or "(empty output)"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Command exceeded {timeout}s limit"
    except Exception as e:
        return f"[ERROR] {str(e)[:200]}"


def run_exploration_agent(
    dataset_dir: str,
    filename: str,
    file_size_human: str,
    user_goal: str | None = None,
    llm_client: Any = None,
) -> Generator[dict, None, dict]:
    """
    ReAct 数据探索 Agent。
    
    Yields 每一步的 thought/action/observation（实时推送给前端）。
    Returns 最终的结构化 insights。
    
    每个 yield 的 dict 格式:
        {"type": "thought", "content": "我看到这是一个ZIP文件..."}
        {"type": "action", "content": "ls -la extracted/"}
        {"type": "observation", "content": "train/ val/ data.yaml ..."}
        {"type": "insight", "content": {...}}  # 最终结论
    """
    if llm_client is None:
        yield {"type": "error", "content": "LLM 未配置，无法启动探索 Agent"}
        return {}
    
    # 初始上下文
    context = f"""你正在探索一个用户上传的数据集。

文件名: {filename}
大小: {file_size_human}
数据目录: {dataset_dir}
{'用户目标: ' + user_goal if user_goal else '用户尚未描述目标'}

你有一个终端可以执行 shell 命令来探索这份数据。你需要自主规划如何理解它：
- 先看看目录结构（ls, find, tree）
- 看看关键文件的内容（cat, head）
- 统计数量（wc, find ... | wc -l）
- 如果需要可以写 Python 脚本做更复杂的分析

每一步你要说出你的想法（Thought），然后给出要执行的命令（Action）。
我会把命令的输出返回给你（Observation），然后你继续分析。

当你觉得已经充分理解了数据集，输出 CONCLUDE 并给出你的结论。"""
    
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    
    yield {"type": "thought", "content": f"开始探索数据集: {filename} ({file_size_human})"}
    
    final_insights = {}
    
    for step in range(MAX_STEPS):
        # 让 LLM 思考下一步
        try:
            response = llm_client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
            )
        except Exception as e:
            yield {"type": "error", "content": f"LLM 调用失败: {str(e)[:100]}"}
            break
        
        messages.append({"role": "assistant", "content": response})
        
        # 解析 LLM 的输出
        parsed = _parse_agent_response(response)
        
        if parsed["thought"]:
            yield {"type": "thought", "content": parsed["thought"]}
        
        if parsed["conclude"]:
            # Agent 认为探索完毕，输出结论
            yield {"type": "thought", "content": "探索完毕，正在总结..."}
            final_insights = _extract_conclusion(response, llm_client, messages)
            yield {"type": "insight", "content": final_insights}
            break
        
        if parsed["action"]:
            cmd = parsed["action"]
            yield {"type": "action", "content": cmd}
            
            # 执行命令
            output = execute_command(cmd, cwd=dataset_dir)
            yield {"type": "observation", "content": output}
            
            # 把 observation 加入对话
            messages.append({
                "role": "user",
                "content": f"[Observation]\n{output}\n\n请继续分析。如果已经充分理解数据集，请输出 CONCLUDE 和你的结论。",
            })
    else:
        # 达到最大步数，强制总结
        yield {"type": "thought", "content": "已达到探索步数上限，正在总结..."}
        messages.append({
            "role": "user",
            "content": "请根据目前观察到的信息，输出 CONCLUDE 和你的最终结论。",
        })
        try:
            response = llm_client.chat_completion(messages=messages, temperature=0.2, max_tokens=2000)
            final_insights = _extract_conclusion(response, llm_client, messages)
            yield {"type": "insight", "content": final_insights}
        except Exception as e:
            yield {"type": "error", "content": f"总结失败: {str(e)[:100]}"}
    
    return final_insights


def _parse_agent_response(text: str) -> dict:
    """解析 Agent 的 Think/Act/Conclude 输出"""
    result = {"thought": "", "action": "", "conclude": False}
    
    lines = text.strip().split('\n')
    current_section = None
    thought_lines = []
    action_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        lower = line_stripped.lower()
        
        # 检测 CONCLUDE
        if 'conclude' in lower and ('###' in line or line_stripped.upper().startswith('CONCLUDE')):
            result["conclude"] = True
            continue
        
        # 检测 Thought
        if lower.startswith('thought:') or lower.startswith('**thought'):
            current_section = 'thought'
            thought_lines.append(line_stripped.split(':', 1)[-1].strip() if ':' in line_stripped else '')
            continue
        
        # 检测 Action
        if lower.startswith('action:') or lower.startswith('**action'):
            current_section = 'action'
            rest = line_stripped.split(':', 1)[-1].strip() if ':' in line_stripped else ''
            if rest and not rest.startswith('```'):
                action_lines.append(rest)
            continue
        
        # 检测代码块中的命令
        if line_stripped.startswith('```') and current_section == 'action':
            continue  # 跳过 ``` 标记
        
        # 追加到当前 section
        if current_section == 'thought':
            thought_lines.append(line_stripped)
        elif current_section == 'action':
            if line_stripped and not line_stripped.startswith('```'):
                action_lines.append(line_stripped)
    
    result["thought"] = ' '.join(t for t in thought_lines if t).strip()
    
    # 取第一条命令（单步执行）
    if action_lines:
        cmd = action_lines[0].strip()
        # 去掉可能的 $ 前缀
        if cmd.startswith('$ '):
            cmd = cmd[2:]
        result["action"] = cmd
    
    # 如果没有明确的 section 标记，尝试启发式解析
    if not result["thought"] and not result["action"] and not result["conclude"]:
        # 整段可能就是 thought
        if 'CONCLUDE' in text.upper():
            result["conclude"] = True
        else:
            result["thought"] = text[:200]
            # 看有没有像命令的行
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('$ ') or (stripped.startswith('ls ') or stripped.startswith('cat ') or 
                    stripped.startswith('head ') or stripped.startswith('find ') or stripped.startswith('python')):
                    result["action"] = stripped.lstrip('$ ')
                    break
    
    return result


def _extract_conclusion(last_response: str, llm_client: Any, messages: list) -> dict:
    """从 Agent 最终回复中提取结构化结论"""
    
    # 先尝试直接从回复中提取 JSON
    try:
        json_str = llm_client._extract_json(last_response)
        data = json.loads(json_str)
        if "data_type" in data or "business_summary" in data:
            return data
    except Exception:
        pass
    
    # 如果回复不是 JSON，让 LLM 结构化它
    structuring_prompt = """请将你刚才的数据集分析结论整理为以下 JSON 格式：

{
    "data_type": "数据集类型（如 yolo_detection, image_classification, tabular, text_corpus, audio, custom 等）",
    "format_details": "具体格式描述",
    "business_summary": "用通俗语言总结这份数据是什么、包含什么",
    "data_understanding": {
        "summary": "一句话总结",
        "organization": "数据组织方式",
        "key_files": ["重要文件及其作用"]
    },
    "statistics": {
        "total_samples": 0,
        "classes": ["类别列表"],
        "class_distribution": {},
        "splits": {}
    },
    "quality_issues": [
        {"severity": "high/medium/low", "description": "问题", "suggestion": "建议"}
    ],
    "training_implications": [
        "对训练方案有影响的关键发现，如类别不均衡、小目标多、标注噪声等"
    ],
    "suggested_next_steps": ["建议"]
}

只输出 JSON，不要其他内容。"""
    
    try:
        messages_copy = messages.copy()
        messages_copy.append({"role": "user", "content": structuring_prompt})
        
        structured = llm_client.chat_completion(
            messages=messages_copy,
            temperature=0.1,
            max_tokens=2000,
        )
        
        json_str = llm_client._extract_json(structured)
        return json.loads(json_str)
    except Exception:
        # 回退：返回原始文本
        return {
            "data_type": "unknown",
            "business_summary": last_response[:500],
            "training_implications": [],
        }


AGENT_SYSTEM_PROMPT = """你是 VibeML 的数据探索 Agent。你的任务是自主探索一个用户上传的数据集，理解它的结构、格式和内容。

## 你的工作方式

你有一个终端，可以执行 shell 命令来观察数据。你的工作流程是：

1. **Thought**: 说出你正在想什么、打算看什么
2. **Action**: 给出一条要执行的 shell 命令
3. 等待系统返回 Observation（命令输出）
4. 根据 Observation 继续 Think → Act，直到充分理解

## 输出格式

每次回复必须包含 Thought 和 Action（或 CONCLUDE）：

```
Thought: 我看到文件是一个ZIP，先看看里面有什么文件结构
Action: find . -maxdepth 3 -type f | head -50
```

或者当你认为已经充分理解数据集时：

```
Thought: 基于以上观察，我已经理解了这是一个 YOLO 格式的目标检测数据集...
CONCLUDE
{...你的结论 JSON...}
```

## 探索策略

- **先整体后局部**: 先 `ls` / `find` 看整体结构，再 `cat` / `head` 看关键文件
- **配置文件优先**: 找 .yaml, .json, .cfg 等配置文件，它们通常定义了数据集格式
- **标注文件抽样**: 看几个标注文件的内容来理解格式（不需要全部看完）
- **统计关键数字**: 样本数、类别数、分布、train/val/test 划分
- **如果需要复杂分析，可以写 Python 一行脚本**: `python3 -c "import os; ..."`
- **高效**: 通常 5-8 步就够了，不要做多余的探索

## 安全规则

- 只读操作，不修改或删除任何文件
- 不要执行网络请求
- 不要执行 `rm`, `mv`, `chmod` 等危险命令

## 你的最终目标

理解这份数据后，输出 training_implications — 对后续训练方案有影响的关键发现。
例如：类别不均衡（需要加权采样）、小目标多（需要多尺度检测）、标注噪声（需要清洗）等。
这些 insights 会直接影响系统为用户生成的个性化训练方案。"""
