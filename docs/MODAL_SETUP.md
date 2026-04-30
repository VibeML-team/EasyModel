# Modal Labs 远端 GPU 训练 —— 部署与运行

VibeML 的 BO 训练可以跑在 Modal 远端 GPU 节点上（参考 Synapse-agent 的后端做法）。
这样：

- 后端 Docker 镜像里**不需要**装 `torch` / `cv2` / `transformers`，构建快、Zeabur 部署稳。
- 训练时按需拉起 GPU（T4 / A10G / A100 / H100），按秒计费。
- BO 的每个 trial 都通过 `modal.Function.spawn()` 跑在 GPU 容器里，结果通过 `call.get()` 同步取回。

> 架构总览
>
> | 文件 | 角色 |
> |---|---|
> | `backend/codegen/modal_functions.py` | **完全独立**的 Modal app，定义 4 个 GPU 函数 + 数据集 push 函数。**不**依赖项目其他模块，只用于 `modal deploy`。 |
> | `backend/codegen/deploy_modal.py` | `modal deploy` 入口。 |
> | `backend/codegen/modal_executor.py` | 客户端 `ModalSandboxExecutor`，对外接口与本地 `SandboxExecutor.execute_training` 完全一致。 |
> | `backend/ml_engineer.py` | 根据 `USE_MODAL` 环境变量自动在本地/Modal 之间切换。 |

---

## 一次性准备

### 1. 注册 Modal 账户

[https://modal.com](https://modal.com) → Sign Up（GitHub / 邮箱皆可），免费额度 $30/月（写文档时）。

### 2. 安装 Modal CLI 并登录（**仅本地开发机**需要做）

```bash
pip install "modal>=0.64.0"
python -m modal setup
```

`modal setup` 会打开浏览器完成 OAuth，token 写入 `~/.modal.toml`。

> Zeabur 容器里**不**需要这一步，token 走环境变量（见下一节）。

### 3. 部署 GPU 函数到 Modal 云端 ⭐关键

**首次部署需要 3-5 分钟**（构建镜像，里面装了 torch / cv2 / transformers / einops / timm / librosa / albumentations / xgboost / lightgbm 等全套 ML 库）。

```bash
cd /root/VibeML-Agent
python -m modal deploy backend/codegen/deploy_modal.py
```

预期输出：

```
✓ Initialized. View app at https://modal.com/apps/vibeml-training
✓ Created execute_training_t4
✓ Created execute_training_a10g
✓ Created execute_training_a100
✓ Created execute_training_h100
✓ Created push_dataset_files
✓ App deployed!
```

验证：

```bash
modal app list                                # 应该看到 vibeml-training
modal app functions vibeml-training           # 应该看到 5 个函数
```

> 一旦部署，后续提交 trial 是**秒级**的（Modal 容器复用），不会再触发镜像重建。
> 如果以后修改了 `modal_functions.py`，重新跑一次 `modal deploy` 即可。

---

## 在 Zeabur 上启用

只需要配两个环境变量然后开关一打开。

### 1. 在 Modal Dashboard 里创建 token

[https://modal.com/settings/tokens](https://modal.com/settings/tokens) → New Token，复制：

- `MODAL_TOKEN_ID` （形如 `ak-...`）
- `MODAL_TOKEN_SECRET` （形如 `as-...`）

### 2. 写入 Zeabur 服务的环境变量

```bash
# 本仓库 zeabur 项目 ID 见 CLAUDE.md
SERVICE_ID=69d5f9039da252559b38d314

npx zeabur@latest variable set --service-id $SERVICE_ID -i=false \
  MODAL_TOKEN_ID=ak-xxxxxxxxxxxxxxxxxxxx \
  MODAL_TOKEN_SECRET=as-xxxxxxxxxxxxxxxxxxxx \
  USE_MODAL=1 \
  MODAL_GPU=A100
```

可选：

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `USE_MODAL` | `1`/`true` 启用 Modal 后端 | `0`（本地沙箱） |
| `MODAL_GPU` | `T4` / `A10G` / `A100` / `H100` | `A100` |
| `MODAL_INLINE_PACK_LIMIT_MB` | 数据集小于该大小时直接 zip+base64 inline；超过自动走 Volume | `200` |

### 3. 重启服务

```bash
npx zeabur@latest service restart --id $SERVICE_ID -y -i=false
```

启动日志里会出现：

```
INFO ml_engineer Using ModalSandboxExecutor (gpu=A100, dataset_id=..., dataset_root=...)
```

---

## 数据集传输策略（自动判断）

`ModalSandboxExecutor` 会按数据集大小自动选模式：

| 大小 | 模式 | 实现 |
|---|---|---|
| ≤ `MODAL_INLINE_PACK_LIMIT_MB`（默认 200MB） | inline | 客户端 zip + base64，随 `spawn()` 一起传过去；Modal 端解压到 `/tmp/dataset/<id>/` |
| > 阈值 | volume | 客户端必须先调用 `push_dataset_to_volume()` 把数据 sync 到 Modal Volume `vibeml-datasets`；训练函数挂载到 `/datasets/<id>/` |

> 同一个 dataset 在 BO 60 个 trial 里只会被打包一次（cache 在 executor 里）。

手动推送大数据集到 Volume 的脚本示例：

```python
from backend.codegen.modal_executor import ModalSandboxExecutor

ex = ModalSandboxExecutor(
    gpu_type="A100",
    dataset_id="my_big_dataset",
    dataset_root="/app/backend/data/my_big_dataset",
)
print(ex.push_dataset_to_volume(batch_size_mb=64))
```

---

## 常见报错

### `modal.exception.AuthError: Token missing`

→ 没设 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`，或值错了。

### `Function 'execute_training_a100' not found in app 'vibeml-training'`

→ 还没跑 `modal deploy`。

### Trial 一直卡住，几分钟没动静

→ 多半是镜像第一次冷构建。等头一次跑完后续就快了。
   也可以提前在本地跑一次 `modal run backend/codegen/modal_functions.py::execute_training_a100 ...` 触发构建。

### `Driver did not emit FINAL_RESULT`

→ 生成的 `train_loop.Trainer.fit()` 抛异常或没正常返回。看日志里
   driver 段最后一行的 traceback。

---

## 成本提示

| GPU | $/小时 | 10 分钟 trial 成本 |
|---|---|---|
| T4 | $0.35 | ~$0.06 |
| A10G | $0.50 | ~$0.08 |
| A100-40GB | $0.60 | ~$0.10 |
| H100 | $1.20 | ~$0.20 |

BO 一次 60 trial × 10 分钟 ≈ A100 上 $6，H100 上 $12。免费额度 $30 够测试很久。
