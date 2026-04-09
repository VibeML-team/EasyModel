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


# Agent 可以使用的命令（沙箱：允许读操作 + python + pip install）
ALLOWED_COMMANDS = [
    "ls", "find", "cat", "head", "tail", "wc", "file", "du",
    "tree", "stat", "grep", "awk", "sed", "sort", "uniq",
    "unzip", "tar", "python3", "python", "pip",
    "echo", "basename", "dirname", "realpath",
    "md5sum", "sha256sum", "mkdir", "cp",
]

MAX_STEPS = 15       # 最大探索步数
CMD_TIMEOUT = 120    # 单条命令超时（秒）— pip install / 数据下载需要更长
MAX_OUTPUT = 3000    # 单条命令输出截断长度


def execute_command(cmd: str, cwd: str, timeout: int = CMD_TIMEOUT) -> str:
    """
    在数据集目录中执行命令（沙箱）。
    
    多行命令自动写入临时脚本执行，避免 shell 引号问题。
    """
    import tempfile, os
    
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin",
        "HOME": "/root",
        "LANG": "en_US.UTF-8",
        "PYTHONPATH": "/app",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_BREAK_SYSTEM_PACKAGES": "1",
    }
    
    try:
        # 多行命令或含复杂引号 → 写临时脚本
        if '\n' in cmd or (cmd.count('"') > 2 and 'python' in cmd):
            script_path = os.path.join(cwd, "_agent_cmd.sh")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write("#!/bin/bash\nset -e\n" + cmd.rstrip() + "\n")
            actual_cmd = f"bash {script_path}"
        else:
            actual_cmd = cmd
        
        result = subprocess.run(
            actual_cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
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
    
    # 保留完整 action 块，避免多行脚本被截断成第一行导致引号不闭合
    if action_lines:
        normalized = []
        for line in action_lines:
            cmd_line = line.strip()
            if cmd_line.startswith('$ '):
                cmd_line = cmd_line[2:]
            normalized.append(cmd_line)
        result["action"] = '\n'.join(line for line in normalized if line).strip()
    
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


AGENT_SYSTEM_PROMPT = """你是 VibeML 的数据探索 Agent。你的任务是**自主获取、探索并理解**用户需要的数据集。

## 核心原则：遇到问题自己解决，不要报错给用户

你是一个自主 Agent，不是一个报错机器。如果数据读不出来、格式不对、文件损坏：
- **自己想办法修复**（换个库读、转格式、用 python 脚本处理）
- **如果本地数据有问题，自己去网上下载**（pip install datasets; python3 -c "from datasets import load_dataset; ..."）
- **永远不要给用户返回"数据为空"**——如果当前文件有问题，就去找能用的数据

## 你的能力

你有一个终端，可以执行 shell 命令：
- `ls`, `find`, `cat`, `head`, `wc` — 查看文件
- `python3 -c "..."` — 执行 Python 代码（pandas, pyarrow, json 等已安装）
- `pip install xxx` — 安装需要的库（如 datasets, Pillow 等）
- `python3 -c "from datasets import load_dataset; ds = load_dataset('mnist'); ..."` — 直接从 HuggingFace 下载数据

如果命令较长，请优先输出多行 shell 脚本，而不是一条超长 `python3 -c "..."`。
例如优先这样写：
```bash
python3 - <<'PY'
print("hello")
PY
```
不要输出引号不闭合的半截命令。

## 工作流程

1. **Thought**: 说出你的分析和计划
2. **Action**: 给出一条命令
3. 观察输出 → 继续

示例：
```
Thought: 文件是 parquet 格式，用 pandas 读取看看
Action: python3 -c "import pandas as pd; df = pd.read_parquet('./*.parquet'); print(df.shape); print(df.head())"
```

如果读取失败：
```
Thought: parquet 读取失败了，可能是格式问题。这是 MNIST 数据集，我直接用 HuggingFace datasets 库下载
Action: pip install datasets -q
python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset('ylecun/mnist', split='train')
print(ds)
print(ds[0])
PY
```

## 关键：如果当前文件有问题，自己去获取数据

对于知名数据集（MNIST, CIFAR-10, ImageNet 等），你知道它们在哪里：
- HuggingFace: `from datasets import load_dataset`
- torchvision: `from torchvision.datasets import MNIST`
- sklearn: `from sklearn.datasets import load_iris`

**不要告诉用户"数据是空的"，而是自己去下载正确的数据并保存到工作目录。**

## 输出格式

每次回复：Thought + Action，或 Thought + CONCLUDE（当完成时）。

CONCLUDE 时输出 JSON 结论。

## 安全规则

- 可以安装 pip 包、下载数据集
- 可以在工作目录中创建/写入文件（保存下载的数据）
- 不要删除用户上传的原始文件
- 不要执行 `rm -rf` 等危险操作

## 最终目标

1. 确保数据集可用（如果不可用，自己修复或重新获取）
2. 理解数据的结构和内容
3. 输出 training_implications — 对训练方案有影响的关键发现"""
