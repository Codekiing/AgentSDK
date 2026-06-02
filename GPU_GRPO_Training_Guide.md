# GPU 环境下跑通 GRPO 数学训练的完整指南

本文档记录了在 AgentSDKMetricSkill 仓库中，使用 GPU (NVIDIA A100) 跑通 Qwen2.5-0.5B + GSM8K + GRPO 训练所需的所有修改。

## 1. 环境准备

### 1.1 硬件要求
- 2x NVIDIA A100-SXM4-80GB (或其他 Ampere/Ada 架构 GPU)
- CUDA Driver >= 12.8
- 系统内存 >= 64GB

### 1.2 创建虚拟环境并安装依赖

```bash
python3 -m venv test
source test/bin/activate

# 1. 安装 PyTorch (必须先装, flash-attn 编译依赖它)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2. 安装 verl (会自动拉取 ray, megatron-core 等)
pip install verl==0.7.1

# 3. 安装 vllm (必须 ==0.19.1, verl 0.7.1 不兼容更低版本)
pip install vllm==0.19.1

# 4. 从源码编译 flash-attn (预编译 wheel 与 torch 2.10 ABI 不兼容)
pip install ninja packaging wheel
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir

# 5. 安装其他依赖
pip install hydra-core>=1.3.0 omegaconf>=2.3.0 einops transformers
```

### 1.3 验证安装

```bash
python -c "
import torch; print('torch', torch.__version__)  # 应为 2.10.0+cu128
import vllm; print('vllm', vllm.__version__)      # 应为 0.19.1
import verl; print('verl', verl.__version__)       # 应为 0.7.1
from flash_attn.bert_padding import unpad_input; print('flash_attn OK')
"
```

---

## 2. 代码修改

### 2.1 新增文件: `run_grpo_train.py`

这是主训练脚本, 绕过 AgentSDK 的 adapter 层, 直接调用 verl 原生 API。关键设计:

- **Worker 类**: 必须使用 `AsyncActorRolloutRefWorker` (不能用 `ActorRolloutRefWorker`), 因为 verl 0.7.1 的 async rollout 模式需要 `update_weights` 方法
- **load_format**: 设为 `"dummy"` (vllm 0.19.1 不支持 `"dummy_dtensor"`)
- **log_prob_micro_batch_size_per_gpu**: 必须显式设置, verl 默认为 null 会导致 TypeError
- **Critic**: GRPO 不需要 Critic, 但 verl 仍要求配置 `critic.model.path`
- **attn_implementation**: 设为 `"eager"` 避免 flash-attn 在训练侧的额外依赖问题

```python
#!/usr/bin/env python3
"""
Qwen2.5-0.5B + GSM8K + GRPO via VERL native API
"""
import os

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

VERL_CONFIG_DIR = os.path.join(
    os.path.dirname(__import__("verl").__file__), "trainer", "config"
)

with initialize_config_dir(config_dir=VERL_CONFIG_DIR, version_base=None):
    cfg = compose(config_name="ppo_trainer")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Qwen2.5-0.5B-Instruct")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data/gsm8k")

cfg_dict = OmegaConf.to_container(cfg, resolve=True)

# Data
cfg_dict["data"]["train_files"] = f"{DATA_DIR}/train.parquet"
cfg_dict["data"]["val_files"] = f"{DATA_DIR}/test.parquet"
cfg_dict["data"]["train_batch_size"] = 8
cfg_dict["data"]["max_prompt_length"] = 512
cfg_dict["data"]["max_response_length"] = 512

# Model
cfg_dict["actor_rollout_ref"]["model"]["path"] = MODEL_PATH
cfg_dict["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"] = "eager"

# Actor
cfg_dict["actor_rollout_ref"]["actor"]["strategy"] = "fsdp"
cfg_dict["actor_rollout_ref"]["actor"]["optim"]["lr"] = 1e-5
cfg_dict["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 2
cfg_dict["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 2
cfg_dict["actor_rollout_ref"]["actor"]["use_kl_loss"] = False
cfg_dict["actor_rollout_ref"]["actor"]["use_torch_compile"] = False

# Rollout (vllm)
cfg_dict["actor_rollout_ref"]["rollout"]["name"] = "vllm"
cfg_dict["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 1
cfg_dict["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.5
cfg_dict["actor_rollout_ref"]["rollout"]["n"] = 2
cfg_dict["actor_rollout_ref"]["rollout"]["max_model_len"] = 1024
cfg_dict["actor_rollout_ref"]["rollout"]["max_num_seqs"] = 16
cfg_dict["actor_rollout_ref"]["rollout"]["enforce_eager"] = True
cfg_dict["actor_rollout_ref"]["rollout"]["load_format"] = "dummy"
cfg_dict["actor_rollout_ref"]["rollout"]["enable_chunked_prefill"] = False
cfg_dict["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 2

# Critic (disabled by GRPO, but path needed)
cfg_dict["critic"]["model"]["path"] = MODEL_PATH
cfg_dict["critic"]["optim"]["lr"] = 1e-5
cfg_dict["critic"]["ppo_mini_batch_size"] = 2
cfg_dict["critic"]["ppo_micro_batch_size_per_gpu"] = 2

# Algorithm
cfg_dict["algorithm"]["adv_estimator"] = "grpo"
cfg_dict["algorithm"]["use_kl_in_reward"] = False

# Trainer
cfg_dict["trainer"]["project_name"] = "verl_grpo_gsm8k_gpu"
cfg_dict["trainer"]["experiment_name"] = "qwen2_0.5b_math"
cfg_dict["trainer"]["total_training_steps"] = 3
cfg_dict["trainer"]["total_epochs"] = 1
cfg_dict["trainer"]["n_gpus_per_node"] = 2
cfg_dict["trainer"]["save_freq"] = 999999
cfg_dict["trainer"]["test_freq"] = 999999
cfg_dict["trainer"]["val_before_train"] = False
cfg_dict["trainer"]["logger"] = ["console"]
cfg_dict["trainer"]["device"] = "cuda"

# Reward
cfg_dict["reward_model"]["enable"] = False

config = OmegaConf.create(cfg_dict)

print("=" * 60)
print("VERL GRPO Training: Qwen2.5-0.5B on GSM8K")
print(f"  Model:   {MODEL_PATH}")
print(f"  Data:    {DATA_DIR}")
print(f"  Steps:   3")
print(f"  GPUs:    2")
print(f"  Strategy: {config.actor_rollout_ref.actor.strategy}")
print("=" * 60)

import ray
ray.init(include_dashboard=False)

from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
from verl.single_controller.ray import RayWorkerGroup

tokenizer = hf_tokenizer(MODEL_PATH)

role_worker_mapping = {
    Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    Role.Critic: ray.remote(CriticWorker),
}

resource_pool_manager = ResourcePoolManager(
    resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node]},
    mapping={
        Role.ActorRollout: "global_pool",
        Role.Critic: "global_pool",
    },
)

trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    ray_worker_group_cls=RayWorkerGroup,
)

print("Initializing workers...")
trainer.init_workers()
print("Workers initialized. Starting training (3 steps)...")
trainer.fit()
print("Training completed!")

ray.shutdown()
```

### 2.2 修改: `data/gsm8k/train.parquet` 和 `test.parquet`

verl 的 `NaiveRewardManager` 要求 parquet 文件包含以下列:

| 列名 | 类型 | 说明 |
|------|------|------|
| `prompt` | `list[dict]` | 聊天格式, 例 `[{"role": "user", "content": "..."}]` |
| `answer` | `str` | 完整答案 (含推理过程) |
| `data_source` | `str` | **必须为 `"openai/gsm8k"`, 用于路由到 verl 内置的 GSM8K 奖励函数** |
| `reward_model` | `dict` | **必须包含 `{"ground_truth": "<数字>"}`, 从 `#### number` 提取** |

原始 GSM8K 数据只有 `prompt` 和 `answer` 列, 需要添加 `data_source` 和 `reward_model`。

处理脚本 (一次性运行):

```python
import pandas as pd
import re

for split in ["train", "test"]:
    df = pd.read_parquet(f"data/gsm8k/{split}.parquet")

    # 添加 data_source 列
    df["data_source"] = "openai/gsm8k"

    # 从 answer 中提取 ground_truth (格式: "... #### 数字")
    def extract_gt(answer):
        if isinstance(answer, str):
            match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer)
            if match:
                return match.group(1).replace(",", "")
        return ""

    df["reward_model"] = df["answer"].apply(lambda x: {"ground_truth": extract_gt(x)})

    df.to_parquet(f"data/gsm8k/{split}.parquet")
    print(f"{split}: {len(df)} rows, ground_truth coverage: {(df['reward_model'].apply(lambda r: r['ground_truth'] != '')).sum()}/{len(df)}")
```

### 2.3 修改: `Qwen2.5-0.5B-Instruct/config.json`

添加一行 `_attn_implementation` 字段, 避免 flash-attn 在模型加载时的兼容问题:

```json
{
  ...
  "_attn_implementation": "eager",
  "vocab_size": 151936
}
```

仅此一行改动, 其余字段保持原始模型配置不变。

---

## 3. 踩坑记录与关键决策

### 3.1 vllm 版本必须 >= 0.19.1

verl 0.7.1 的 async rollout 模式依赖 vllm 的以下 API:
- `--enable-sleep-mode`
- `--logprobs_mode processed_logprobs`
- `run_headless` / 异步 HTTP server 模式

vllm 0.8.5 缺少这些 API, 会导致启动失败。但 vllm >= 0.20.2 需要 CUDA 13, 当前驱动不支持。
**结论: vllm == 0.19.1 是唯一可行的版本。**

### 3.2 Worker 类必须用 AsyncActorRolloutRefWorker

verl 0.7.1 默认使用 async rollout 模式 (`rollout.mode: async`)。权重同步 (`update_weights`) 方法只定义在 `AsyncActorRolloutRefWorker` 上, `ActorRolloutRefWorker` 没有此方法。用错会报:

```
AttributeError: 'RayWorkerGroup' object has no attribute 'update_weights'
```

### 3.3 load_format 必须是 "dummy"

vllm 0.19.1 支持的 load_format 列表: auto, hf, bitsandbytes, **dummy**, fastsafetensors, gguf, ...
不支持 `"dummy_dtensor"` (这是更早版本的格式)。用错会报:

```
ValueError: Load format `dummy_dtensor` is not supported
```

### 3.4 log_prob_micro_batch_size_per_gpu 必须显式设置

verl 默认值为 `null`, 会导致 `compute_log_prob` 中 `data.split(None)` 报 TypeError:

```
TypeError: 'NoneType' object cannot be interpreted as an integer
```

### 3.5 flash-attn 必须从源码编译

预编译的 flash-attn wheel (如 `flash_attn-2.7.4+cu12torch2.6`) 与 torch 2.10.0 的 C++ ABI 不兼容, 会报:

```
ImportError: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationE...
```

必须用 `--no-build-isolation --no-cache-dir` 从源码编译。编译耗时约 15-30 分钟 (MAX_JOBS=4)。

### 3.6 flash-attn 安装后需重新安装 torch

flash-attn 的 pip 依赖会把 torch 升级到不兼容的版本。安装完 flash-attn 后, 必须:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

然后验证 `nvidia-nccl-cu12` (而非 `nvidia-nccl-cu13`) 存在。

---

## 4. 运行

```bash
source test/bin/activate
python run_grpo_train.py
```

预期输出:
```
============================================================
VERL GRPO Training: Qwen2.5-0.5B on GSM8K
  Model:   .../Qwen2.5-0.5B-Instruct
  Data:    .../data/gsm8k
  Steps:   3
  GPUs:    2
  Strategy: fsdp
============================================================
...
Workers initialized. Starting training (3 steps)...
...
Training Progress: 100%|██████████| 3/3
Training completed!
```

### 正式训练建议

上述配置仅用于验证 pipeline (3步, dummy 权重)。正式训练需修改:

| 配置项 | 当前值 (验证) | 建议值 (正式) |
|--------|-------------|-------------|
| `total_training_steps` | 3 | 100+ |
| `load_format` | `"dummy"` | `"auto"` (从磁盘加载真实权重) |
| `train_batch_size` | 8 | 128-1024 |
| `save_freq` | 999999 | 每 N 步保存 |
| `test_freq` | 999999 | 每 N 步验证 |
| `n_gpus_per_node` | 2 | 按实际 GPU 数量 |

---

## 5. 完整安装顺序 (一键脚本)

```bash
#!/bin/bash
set -e

python3 -m venv test
source test/bin/activate

# Step 1: PyTorch
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 2: VERL + VLLM
pip install verl==0.7.1
pip install vllm==0.19.1

# Step 3: flash-attn (from source, ~20min)
pip install ninja packaging wheel
MAX_JOBS=4 pip install flash-attn --no-build-isolation --no-cache-dir

# Step 4: Fix torch version (flash-attn may have upgraded it)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# Step 5: Verify
python -c "
import torch; assert '2.10.0' in torch.__version__, f'torch {torch.__version__}'
import vllm; assert vllm.__version__ == '0.19.1', f'vllm {vllm.__version__}'
import verl; assert verl.__version__ == '0.7.1', f'verl {verl.__version__}'
from flash_attn.bert_padding import unpad_input
print('All dependencies verified!')
"
```
