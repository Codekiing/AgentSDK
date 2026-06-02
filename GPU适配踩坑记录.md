# AgentSDK GPU 适配踩坑记录

> 环境：4x A100-SXM4-80GB, CUDA 13.0, Python 3.10, Ubuntu 22.04
> 时间：2026-05-11

---

## 一、项目背景

AgentSDK (`agentic_rl`) 是华为为昇腾 NPU 开发的 Agentic RL 训练框架，支持 MindSpeed-RL (NPU) 和 VERL (GPU) 两种后端。
原项目仅在 NPU 上验证过，需要适配到 GPU 环境跑通 GRPO 训练。

---

## 二、核心依赖版本选择

| 组件 | NPU 版本 | GPU 版本（最终可用） | 说明 |
|------|---------|---------------------|------|
| PyTorch | 昇腾定制 | 2.6.0+cu124 | 必须与 flash-attn、vllm 版本对齐 |
| vllm | vllm_ascend（定制分支） | 0.8.5 | 0.20.2 API 不兼容，降级解决 |
| verl | 特定版本 | 0.7.1 | AgentSDK 适配层与其存在 API 差异 |
| flash-attn | 不需要 | 2.8.3 | 必须对齐 torch 版本重新编译 |
| ray | 2.55.1 | 2.55.1 | 基本无兼容问题 |
| megatron-core | mindspeed 定制 | >=0.12.0 | NPU 的 mindspeed 系列包全部移除 |

**关键教训**：vllm 版本决定了 torch 版本，flash-attn 必须在最终 torch 版本下编译。版本选择顺序：
```
确定 vllm 版本 → 确定 torch 版本 → 编译 flash-attn
```

---

## 三、逐个踩过的坑

### 坑 1：文件权限校验（AgentSDK FileCheck）

**现象**：`mode is not right, it must be 640`

**原因**：AgentSDK 强制要求配置文件 640、目录 750 的权限。

**解决**：
```bash
find . -type f -name "*.yaml" -o -name "*.py" -o -name "*.json" -o -name "*.parquet" | xargs chmod 640
find . -type d | xargs chmod 750
```

### 坑 2：GlobalConfig 校验失败

**现象**：`Extra inputs are not permitted`（如 `adv_estimator` 放在顶层）；`save_freq: -1` 必须为正数。

**原因**：AgentSDK 用 Pydantic 定义了严格的 `GlobalConfig` schema，字段位置和类型有严格约束。

**解决**：
- `adv_estimator` 等训练参数放到 `verl:` 子配置段
- `save_freq` 设为 999999（大于训练步数即可）

### 坑 3：AgentSDK 适配层与 verl 0.7.1 API 不兼容（最关键）

**现象**：多种报错：
- `No module named 'verl.workers.sharding_manager.fsdp_vllm'`
- `RayPPOTrainer.__init__() got an unexpected keyword argument 'reward_fn'`
- `NaiveRewardManager.__init__() got an unexpected keyword argument 'num_examine'`

**原因**：AgentSDK 的 VERL 适配层（`train_agent_grpo.py`、`agent_grpo_trainer.py`）是为更新版本 verl 编写的，与 verl 0.7.1 的接口存在大量差异：
- `FSDPVLLMShardingManager` 模块不存在（训推共卡模式实际不需要）
- `RayPPOTrainer` 构造函数签名不同，不接受 `reward_fn` 等参数
- `load_reward_manager()` 的参数签名不同

**解决**：彻底绕过 AgentSDK 适配层，直接用 verl 原生 API 编写 `run_grpo_train.py`：
```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# 用 hydra compose 加载 verl 完整默认配置
with initialize_config_dir(config_dir=VERL_CONFIG_DIR, version_base=None):
    cfg = compose(config_name="ppo_trainer")

# 转成 plain dict，直接修改，再转回 OmegaConf
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
cfg_dict["algorithm"]["adv_estimator"] = "grpo"
# ... 其他覆盖
config = OmegaConf.create(cfg_dict)
```

### 坑 4：OmegaConf struct mode 阻止配置合并

**现象**：向 OmegaConf 配置对象添加新字段时报 `struct mode` 错误。

**原因**：verl 默认配置以 struct 模式加载，不允许添加新 key。

**解决**：先 `OmegaConf.to_container(cfg, resolve=True)` 转为 plain dict，修改后再 `OmegaConf.create()` 转回。

### 坑 5：vllm 0.20.2 模块路径变更

**现象**：`No module named 'vllm.entrypoints.openai.protocol'`

**原因**：vllm 0.20.2（太新）重构了模块结构，与 AgentSDK 和 verl 0.7.1 都不兼容。

**解决**：降级到 vllm 0.8.5。

### 坑 6：flash-attn ABI 不兼容

**现象**：`undefined symbol: _ZN3c105ErrorC2...` in `flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`

**原因**：之前安装过 vllm 0.20.2（附带 torch 2.11），flash-attn 在该环境下编译。后来降级 torch 到 2.6.0，但 flash-attn 的 `.so` 是针对 torch 2.11 的 C++ ABI 编译的，无法加载。

**解决方案**（二选一）：
- **方案 A（推荐）**：确保 torch 版本正确后重新编译 flash-attn
  ```bash
  pip install torch==2.6.0+cu124  # 确保版本正确
  pip install flash-attn --no-build-isolation  # 编译时链接当前 torch
  ```
- **方案 B**：不装 flash-attn，用 eager attention（需要在 verl 的 override_config 中正确设置 `attn_implementation: "eager"`）

### 坑 7：attn_implementation 覆盖不生效

**现象**：卸载 flash-attn 后，设置 `_attn_implementation: "eager"` 但 transformers 仍尝试用 FlashAttention2。

**原因**：
- `_attn_implementation` 不是 `AutoConfig.from_pretrained` 的标准参数，会被忽略
- 正确的参数名是 `attn_implementation`（无前缀下划线）
- verl 的 `override_config` 字段中的 key 会传递给 `AutoConfig.from_pretrained` 作为 kwargs

**解决**：
```python
cfg_dict["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"] = "eager"
```
或在模型 `config.json` 中添加（但可能被 verl 的 override 流程覆盖）：
```json
{"_attn_implementation": "eager"}
```

### 坑 8：Ray OTel 依赖冲突

**现象**：Ray 启动时 OpenTelemetry 相关 import 报错。

**解决**：
```bash
pip uninstall opentelemetry-exporter-prometheus opentelemetry-api opentelemetry-sdk -y
```

### 坑 9：torch_c_dlpack_ext undefined symbol

**现象**：`undefined symbol` from `torch_c_dlpack_ext`

**原因**：残留的不兼容包。

**解决**：`pip uninstall torch_c_dlpack_ext`

---

## 四、数据准备要点

GSM8K 数据需要转为 verl 的 RLHFDataset 格式（parquet）：
- `prompt` 列：`[{"role": "user", "content": "问题内容"}]` 格式的 chat message list
- `answer` 列（可选）：原始答案文本

示例：
```python
import pandas as pd

data = [{"prompt": [{"role": "user", "content": "Janet's eggs..."}], "answer": "42"}]
pd.DataFrame(data).to_parquet("train.parquet")
```

---

## 五、verl 原生 GRPO 训练脚本关键配置

直接使用 verl 原生 API 时的核心配置项：

```python
# 模型路径
cfg_dict["actor_rollout_ref"]["model"]["path"] = MODEL_PATH

# 数据
cfg_dict["data"]["train_files"] = "data/gsm8k/train.parquet"
cfg_dict["data"]["val_files"] = "data/gsm8k/test.parquet"

# GRPO 算法（自动禁用 Critic）
cfg_dict["algorithm"]["adv_estimator"] = "grpo"

# 不需要 reward model
cfg_dict["reward_model"]["enable"] = False

# 训练步数
cfg_dict["trainer"]["total_training_steps"] = 10

# rollout（vllm 推理）配置
cfg_dict["actor_rollout_ref"]["rollout"]["enforce_eager"] = True       # 不用 CUDA graph
cfg_dict["actor_rollout_ref"]["rollout"]["load_format"] = "dummy_dtensor"  # FSDP 权重格式

# 显存控制
cfg_dict["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.5
cfg_dict["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"] = 1

# attention 实现（无 flash-attn 时）
cfg_dict["actor_rollout_ref"]["model"]["override_config"]["attn_implementation"] = "eager"
```

---

## 六、总结：GPU 适配的正确步骤

1. **确定版本链**：vllm → torch → flash-attn，版本必须对齐
2. **创建隔离环境**：`python -m venv test`，避免系统包污染
3. **安装顺序**：torch → vllm → verl → flash-attn（必须最后装，确保编译时 torch 正确）
4. **绕过 AgentSDK 适配层**：直接用 verl 原生 API，避免 API 不兼容问题
5. **用 hydra compose 加载 verl 配置**：转成 dict 后修改，避免 OmegaConf struct 问题
6. **GRPO 不需要 Critic**：verl 自动检测 `adv_estimator != "gae"` 并禁用
7. **训推共卡（hybrid engine）不需要 FSDPVLLMShardingManager**
8. **文件权限**：AgentSDK 要求 640/750
