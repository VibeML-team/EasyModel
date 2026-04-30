#!/usr/bin/env python3
"""
VibeML Modal 部署入口

参考 Synapse-agent backend/deploy_modal.py 的做法。

使用：
    # 一次性部署所有 GPU 函数到 Modal 云端：
    python -m modal deploy backend/codegen/deploy_modal.py

部署成功后会输出：
    ✓ Created execute_training_t4
    ✓ Created execute_training_a10g
    ✓ Created execute_training_a100
    ✓ Created execute_training_h100
    ✓ Created push_dataset_files
    ✓ App deployed at https://modal.com/apps/vibeml-training

之后客户端 ModalSandboxExecutor 会用 modal.Function.from_name(...)
直接复用已部署的函数（容器复用，秒级提交，不再触发 3-10 分钟的镜像重建）。
"""

# 必须在模块级别 import，让 modal CLI 能扫到 app 对象
from modal_functions import (  # type: ignore  # noqa: F401
    app,
    execute_training_t4,
    execute_training_a10g,
    execute_training_a100,
    execute_training_h100,
    push_dataset_files,
)


if __name__ == "__main__":
    print("=" * 60)
    print("VibeML - Modal 部署入口")
    print("=" * 60)
    print()
    print("请使用 modal CLI 部署：")
    print()
    print("    python -m modal deploy backend/codegen/deploy_modal.py")
    print()
    print("部署函数清单：")
    print("  - execute_training_t4   (T4 GPU)")
    print("  - execute_training_a10g (A10G GPU)")
    print("  - execute_training_a100 (A100-40GB GPU)")
    print("  - execute_training_h100 (H100 GPU)")
    print("  - push_dataset_files    (CPU 辅助函数，向 Volume 推送数据集)")
    print()
    print("App 名称: vibeml-training")
