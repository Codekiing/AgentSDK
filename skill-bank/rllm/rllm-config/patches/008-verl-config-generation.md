---
id: "008-verl-config-generation"
target_section: "output"
action: append
description: "When backend=verl is passed, generate run_verl.sh shell script alongside config.json using rllm_train/verl_config.py, with model-size-specific defaults and dataset path resolution"
source: "2026-05-28 VERL backend integration"
created: "2026-05-28"

depends_on:
  - "007-dataset-scope-isolation"
conflicts_with: []

status: active
superseded_by: ""
---

### VERL 配置生成 (backend=verl)

当编排者传入 `backend=verl` 标记时，除标准 config.json 外，还需生成 `run_verl.sh`：

#### 1. 生成标准 config.json (不变)

使用 TrainingConfig 生成标准 JSON，附加 `"backend": "verl"` 字段：
- config.json 作为元数据记录，供后续 Phase 参考
- 必须包含 `"backend": "verl"` 字段，rllm-run 和 rllm-monitor 通过此字段判断后端

#### 2. 生成 run_verl.sh 启动脚本

使用 `rllm_train/verl_config.py` 工具：

```bash
python -c "
from rllm_train.verl_config import generate_verl_script, get_verl_config_summary
from rllm_train.config import TrainingConfig
import json

cfg = TrainingConfig.from_json('rllm_train/output/runs/<run_id>/config.json')
script_path = generate_verl_script(cfg)
summary = get_verl_config_summary(cfg)
print(f'VERL launch script: {script_path}')
print(json.dumps(summary, indent=2))
"
```

#### 3. 参数映射表 (TrainingConfig → VERL Hydra args)

| TrainingConfig 字段 | VERL Hydra CLI arg | 7B 默认值 |
|---|---|---|
| model_name | actor_rollout_ref.model.path | Qwen/Qwen2.5-7B-Instruct |
| learning_rate | actor_rollout_ref.actor.optim.lr | 5e-6 |
| num_generations | actor_rollout_ref.rollout.n | 4 |
| num_epochs | trainer.total_epochs | 1 |
| batch_size | data.train_batch_size | 16 (自动计算) |
| max_response_length | data.max_response_length | 2048 |
| max_prompt_length | data.max_prompt_length | 1024 |
| temperature | actor_rollout_ref.rollout.temperature | 0.7 |
| gradient_checkpointing | actor_rollout_ref.model.enable_gradient_checkpointing | True |

#### 4. VERL 模型级默认值

| 模型规模 | 默认配置 |
|---|---|
| 7B (如 Qwen2.5-7B) | FSDP param_offload=True, optimizer_offload=True, vLLM TP=1, gpu_mem=0.6, 4 GPU, train_batch=16, mini_batch=32 |
| 0.5B-3B | FSDP param_offload=False, vLLM TP=1, gpu_mem=0.4, 1-2 GPU, train_batch=8, mini_batch=16 |
| 14B+ | FSDP param_offload=True, optimizer_offload=True, vLLM TP=2, gpu_mem=0.5, 8 GPU, train_batch=32, mini_batch=64 |

#### 5. 数据集解析

- `dataset="deepscaler"` → `data/deepscaler_verl/train.parquet`, `data/deepscaler_verl/test.parquet`
- `dataset_path` 非空 → 需为 Parquet 格式目录（含 train.parquet 和 test.parquet）
- Parquet 文件必须包含字段: `data_source`, `prompt`, `ability`, `reward_model`, `extra_info` [, `uid`]
- 自定义奖励函数自动指向 `custom_rewards/deepscaler_reward.py`

#### 6. 配置摘要展示

生成完成后，向用户展示 VERL 配置摘要：

```
VERL 配置已生成 [Run: <run_id>]
  模型规模:      7B (4 GPU)
  训练数据:      data/deepscaler_verl/train.parquet (256 条)
  验证数据:      data/deepscaler_verl/test.parquet (32 条)
  推理引擎:      vLLM (TP=1, n=4, mem=0.6)
  训练策略:      FSDP + CPU offload
  学习率:        5e-6
  训练轮次:      1 epoch
  TRL 整除:      N/A (VERL 不受此约束)
```
