#!/usr/bin/env python3
"""
部署当前项目的 Modal 训练函数。

用法:
    python3 -m modal deploy backend/deploy_modal.py
"""

from backend.modal_functions import (
    app,
    execute_training_a10g,
    execute_training_a100,
    execute_training_h100,
    execute_training_t4,
)


__all__ = [
    "app",
    "execute_training_t4",
    "execute_training_a10g",
    "execute_training_a100",
    "execute_training_h100",
]


if __name__ == "__main__":
    print("Modal app: vibeml-training")
    print("Deploy with: python3 -m modal deploy backend/deploy_modal.py")
