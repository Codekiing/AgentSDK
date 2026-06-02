---
description: Generate and tune training configurations for rllm_train. Handles parameter
  safety ranges, constraint validation, and model-specific defaults.
metadata:
  categories:
  - training
  - configuration
  version: 1.0.0
name: rllm-config
---


# rllm-config — 训练配置生成与调参

你是 rllm_train 训练配置专家。你的职责是根据用户需求生成合理的训练配置，并在多轮调参中根据训练反馈调整参数。

配置通过 `TrainingConfig` dataclass 管理，支持自然语言解析：

```python
from rllm_train.config import TrainingConfig, parse_natural_language
cfg = parse_natural_language("用 qwen-0.5b 训练数学 agent，64 个问题，2 个 epoch")
```

### 参数安全范围

#### 模型级别安全配置（硬约束）

生成配置时，必须根据模型大小查表，参数不得超出对应上限。

| 参数 | 0.5B 上限 | 1.5B 上限 | 3B 上限 | 依据 |
|------|----------|----------|--------|------|
| learning_rate | 1e-5 | 2e-5 | 5e-5 | 0.5B 在 2e-5 时策略崩溃 |
| num_epochs | 2 | 4 | 6 | 0.5B 在 4ep 时 catastrophic forgetting |
| max_completion_length | 256 | 512 | 512 | MPS 显存限制 |
| num_problems (MPS) | 32 | 32 | 16 | 配合 num_generations=4 的显存上限 |

调参建议超出上限时，必须警告并拒绝。例如：
  建议 num_epochs: 2 → 4
  → 检查: 0.5B 模型 epochs 上限为 2
  → 拒绝，改为建议: 增加 num_problems 或换 1.5B 模型

#### 通用参数范围

| 参数 | 最小值 | 最大值 | 说明 |
|---|---|---|---|
| learning_rate | 1e-7 | 1e-3 | 超出范围大概率不收敛 |
| temperature | 0.3 | 1.5 | 太低无探索，太高太随机 |
| num_generations | 2 | 8 | GRPO 至少需要 2 |
| batch_size | 1 | 4 | Mac 内存限制 |
| num_problems | 8 | 512 | 太少不够学，太多太慢 |
| num_epochs | 1 | 20 | 过多可能过拟合 |
| max_agent_steps | 1 | 8 | 影响生成长度和速度 |
| gradient_accumulation_steps | 1 | 16 | 等效增大 batch |

当 num_problems >= 64 时，0.5B 模型的安全范围收紧:

| 参数 | 原上限 | 新上限 | 条件 | 依据 |
|------|--------|--------|------|------|
| learning_rate | 1e-5 | 5e-6 | num_problems >= 64 | lr=1e-5 在 64 problems 时导致 catastrophic forgetting |
| num_epochs | 2 | 1 | num_problems >= 64 | 2 epochs 在 Step 8/128 时 reward 已开始崩溃 |

推荐初始配置 (0.5B + 64 problems):
- lr=5e-6, epochs=1, batch=2, generations=4
- 预期: reward 稳定在 0.5-0.8 范围，不会崩溃

当使用内置合成数据且 difficulty=mixed 时，0.5B 模型的 num_problems 安全上限进一步收紧:

| 参数 | 原上限 | 新上限 | 条件 | 依据 |
|------|--------|--------|------|------|
| num_problems | 64 | 32 | dataset_path 为空、difficulty=mixed 且 model=0.5B | 合成 mixed 数据历史 run 在 64 problems 下出现 forgetting |

推荐配置（仅 0.5B + 合成 mixed）:
- num_problems=32, lr=5e-6, epochs=1, batch=2, generations=4
- 预期: 32 步训练，reward 稳定不崩溃

替代方案（仅合成数据）: 保持 64 problems 但切换 difficulty=simple
- 适用于需要更多合成训练数据但不需要 hard 题目的场景

外部数据集规则:
- 当 dataset_path 非空时，禁止应用本节 num_problems 上限和 simple/mixed/hard 切换建议。

### Seed 随机化策略

多轮训练时（traj-loop 或手动多轮）:
- 每轮使用不同 seed: `seed = base_seed + round_number`
- 或启用 dataset shuffle: `shuffle=True`
- 目的: 避免相同问题固定在相同 step，导致零 reward 步骤的周期性模式

轨迹证据:
- R3 和 R5 使用相同 seed=42，零 reward 步骤完全一致 [5,6,12,25,31]
- 训练未改善模型在这些特定问题上的表现
- 变更 seed 可以让模型接触不同的问题排列，获得更多样的学习信号

### num_problems 精细化范围（仅限 0.5B + 合成数据）

基于历史合成数据训练更新推荐范围；只在 `dataset_path` 为空时使用:

| difficulty | 推荐范围 | 依据 |
|-----------|---------|------|
| mixed (20% hard) | 40-48 | 32 太简单 (loss=0), 64 forgetting |
| mixed-hard (50% hard) | 24-32 | hard 比例增加后需减少总量 |
| hard | 16-24 | 64 完全超出能力 |

默认推荐配置（仅 0.5B + 合成正式训练）:
- num_problems=48, difficulty=mixed, lr=5e-6, epochs=1
- 预期: 比 32 problems 更有挑战性，但不会 forgetting

外部数据集规则:
- DeepScaler/自定义 Dataset 不应用这些范围。
- 外部数据集的 num_problems 只依据当前 run 的耗时、OOM、reward variance、plateau、calculator_error 等实际分析结果调整。

### num_problems 最优范围精细化（仅限 0.5B + 合成 mixed 数据）

以下经验只适用于 `dataset_path` 为空、使用内置合成 mixed 数据的 0.5B 训练；不得用于 DeepScaler 或其他外部数据集。

| num_problems | 结果 | 证据 |
|-------------|------|------|
| 32 | 无 forgetting 但 loss=0 (无学习) | 历史合成数据 run：avg≈0.77, loss=0 全程 |
| 48 | avg≈0.85 但后期格式退化 | 历史合成数据 run：后 20% reward 明显下降 |
| 40 (推断) | 平衡点 | 32 太简单, 48 后期崩溃 |

更新推荐（仅合成 mixed）:
- 首轮训练: num_problems=40 (安全起点)
- avg_reward >= 0.85 且无后期 forgetting: 可尝试 44
- 出现后期 forgetting: 减少到 36
- 禁止 0.5B+mixed 合成数据使用 num_problems > 48

**Loss=0 诊断**:
| 症状 | 调整 | 原因 |
|------|------|------|
| loss=0 全程 + reward >= 0.8 | difficulty 提升一级 (simple→mixed, mixed→hard) | 题目太简单，模型预训练能力已覆盖，GRPO 无学习信号 |
| loss=0 全程 + reward < 0.5 | 检查 num_generations 和 temperature | reward variance 不足，GRPO baseline 估计有问题 |

### 题目难度自动升级

当调参输入满足以下条件时，优先建议提高难度而非调整超参数:
- avg_reward >= 0.8 且 loss 接近 0
- difficulty 当前为 simple 或 mixed

诊断逻辑:
  if reward >= 0.8 and loss ≈ 0:
      if difficulty == "simple": → 建议切换到 mixed
      if difficulty == "mixed": → 建议增加 num_problems 或切换到 hard
      不要调 lr/epochs/batch 等超参数，问题不在训练动态而在数据难度

### 后半段 Reward 下降检测

当 analysis.json 显示后半段 avg_reward < 前半段 * 0.85 时:
- 首先排除数据分布因素（不同 seed 的随机波动）
- 如果连续 2 轮出现此模式: 建议减少 num_problems（缩短训练长度避免退化）
- 如果仅单轮出现: 标记为观察项，不调参
- 不要因单轮的后半段下降就降低 lr 或 epochs，这可能是正常波动

### 7B 模型 GRPO 关键约束（A100 80GB 单卡）

| 参数 | 7B 推荐值 | 原因 |
|------|----------|------|
| num_generations | >= 4（最低 3） | 2 个 completion 时 GRPO advantage 常为 0, 无有效梯度, loss=0 |
| difficulty | 不跨数据集自动修改 | `difficulty` 只对合成数据生效；外部数据集必须依据当前 run 的日志/轨迹/analysis 调参，不得套用历史 mixed/hard reward 经验 |
| batch_size | 1 | 单卡显存限制, batch=2 + gen=4 会 OOM |
| learning_rate | 2e-6 ~ 5e-6 | 7B 模型需要较低的 lr 避免 grad norm 爆炸 |

#### 显存预估

| num_generations | batch_size | 预计显存 | 是否可行 |
|----------------|-----------|---------|---------|
| 2 | 1 | ~30GB | 可行但无学习效果 |
| 4 | 1 | ~50GB | 可行（推荐） |
| 4 | 2 | ~75GB | 临界, 可能 OOM |
| 2 | 2 | ~55GB | 可行但无学习效果 |

### 多 GPU 参数缩放

当硬件为 N 张 GPU 时（从用户输入或硬件信息中提取），参数按以下规则调整：

1. **batch_size 是 per_device 值**: TRL/Transformers 的 `per_device_train_batch_size` 指单卡 batch。N 卡时有效 batch = batch_size × N
2. **TRL 整除约束不变**: 仍按 per_device 值校验 `(batch_size * gradient_accumulation_steps) % num_generations == 0`
3. **显存可分片**: 多卡时可用 FSDP/DeepSpeed 分片模型，每卡显存负载约为总显存 / N
4. **不决定后端**: rllm-config 只负责生成 TrainingConfig JSON 和参数安全校验。多卡训练的后端选择（rllm_train HF / VERL / DeepSpeed）由 rllm-run 根据实际环境决定

GPU 显存估算（单卡，bf16 + LoRA，不含 FSDP 分片）:

| 模型 | 基础占用 | + batch=1,gen=4,len=512 | + batch=2,gen=4,len=512 | 单卡推荐 |
|------|---------|------------------------|------------------------|---------|
| 0.5B | ~2GB | ~4GB | ~6GB | batch=2, gen=4 |
| 1.5B | ~5GB | ~10GB | ~15GB | batch=2, gen=4 |
| 3B | ~10GB | ~18GB | ~28GB | batch=1, gen=4 |
| 7B | ~14GB | ~30GB | ~50GB | batch=1, gen=4 (80GB卡) |

7B 多卡推荐:

| GPU 配置 | batch_size | num_generations | 有效 batch | 显存/卡 |
|---------|-----------|----------------|-----------|--------|
| 1×A100-80G | 1 | 4 | 1 | ~50GB |
| 2×A100-80G | 2 | 4 | 4 | ~30GB |
| 4×A100-80G | 2 | 4 | 8 | ~20GB |
| 4×A100-40G | 1 | 4 | 4 | ~35GB (临界) |

### 初始配置生成

根据用户输入的模型大小和任务类型，生成初始配置。需考虑：
1. 模型大小 → 影响 batch_size 和 max_completion_length
2. GPU 显存 → 单卡 vs 多卡
3. 任务难度 → num_problems 和 difficulty

### 难度配置

`difficulty` 只控制 rllm_train 的内置合成数据生成器 `generate_math_problems()`，适用条件是 `dataset_path` 为空且未使用外部数据集。

合成数据下的含义:
- `"simple"`: 100% 简单两数运算 (适合流程验证)
- `"hard"`: 100% 多步骤应用题
- `"mixed"`: 80% simple + 20% hard

外部数据集下的规则:
- 当 `dataset_path` 非空时，`difficulty` 不会改变样本分布；训练代码只执行 `load_from_disk(dataset_path)`、shuffle、select。
- 禁止因为旧 run 的 `difficulty` / reward 经验修改外部数据集配置，除非该数据集已有明确难度字段且当前代码实现了过滤逻辑。
- 对 DeepScaler 或自定义 HuggingFace Dataset，调参只能依据当前 run 的 `analysis.json`、日志和轨迹指标；不要套用合成数据的 `simple/mixed/hard` 经验。

初始配置推荐（仅合成数据）:

| 场景 | difficulty | 原因 |
|------|-----------|------|
| 流程验证 | simple | 确认 pipeline 正常 |
| 正式训练 | mixed | 80/20 比例经验证有效 |
| 能力评估 | hard | 评估模型上限，不用于训练 |

调参时的难度调整（仅合成数据）:
- reward=1.0 + loss=0 → 题目太简单，切换到 mixed 或 hard
- reward<0.1 + difficulty=hard → 太难，切换到 mixed
- mixed 下 reward 在 0.3-0.7 → 比例合适，保持不变

### 数据源作用域隔离（必须优先执行）

生成或调参配置前，先判定数据源作用域:

| 条件 | 数据源类型 | 可使用的经验规则 |
|---|---|---|
| `dataset_path` 为空 | 内置合成数据 | 可使用 simple/mixed/hard、0.5B mixed problem-count、difficulty scaling 等合成数据经验 |
| `dataset_path` 非空 | 外部 HuggingFace Dataset | 不使用合成数据 difficulty 经验；只根据当前 run 的 config/log/trajectory/analysis 调参 |
| `dataset="deepscaler"` 或路径含 deepscaler | DeepScaler 外部数据集 | 不使用 simple/mixed/hard 调参；只调整通用训练参数和当前样本选择参数 |

硬性规则:
1. 外部数据集下，`difficulty` 只是配置字段，不会改变 `load_external_dataset()` 的样本分布。除非训练代码已实现基于数据集字段的过滤，否则不得声称 mixed/hard 会改变 DeepScaler 难度。
2. 历史 run 的 reward、loss、forgetting、plateau 结论只能用于相同数据源、相同模型量级、相同 reward 函数和相同训练代码路径。缺少任一条件时，只能作为背景风险提示，不能直接驱动参数选择。
3. 禁止在外部数据集配置中引用固定历史指标（例如 “mixed reward 已达 0.773”）作为切换 difficulty 或判断任务太简单的依据。
4. 调参循环必须优先读取当前 run 的 `analysis.json`。若 `analysis.json` 与 skill 中的历史经验冲突，以当前 run 为准。
5. `training_state.json` 只能作为当前训练会话的历史摘要，不得把其它目标、其它数据集或已标记 completed 的旧会话当作本次训练轮次依据。

DeepScaler/外部数据集可调参数优先级:
1. 进程异常/OOM/静默退出 → 降低 `num_problems` 或生成长度，保留可复现实验规模。
2. reward_variance 长期接近 0 → 增大 `temperature` 或 `num_generations`，但保持 TRL 整除约束。
3. calculator_error 高 → 不通过 difficulty 调参；应降低对工具不可执行题型的依赖，或在数据预处理/环境工具能力上处理。
4. completion 接近上限或 clipped_ratio 高 → 增大 `max_completion_length` / `max_response_length`。
5. reward 震荡且 grad_norm/loss 异常 → 降低 `learning_rate` 或增加稳定性约束。

### 调参策略

收到训练分析报告后，根据问题模式调整参数：
- reward 下降 → 降低 lr，增大 batch
- grad norm 过高 → 降低 lr，添加 gradient clipping
- OOM → 减小 batch_size / num_generations / max_completion_length

### 参数联动约束（生成配置前必须验证）

1. **TRL 整除约束**:
   `(batch_size * gradient_accumulation_steps) % num_generations == 0`
   违反时: 自动调整 num_generations 为最近的合法值

2. **generation_batch_size 副作用**:
   `generation_batch_size = batch_size * gradient_accumulation_steps`
   当 grad_accum 增大时，每步生成的 trajectory 数量也增大
   影响: GRPO baseline 估计变化，训练动态改变
   建议: 调整 grad_accum 时同步说明对 generation_batch_size 的影响

3. **显存估算 (MPS)**:
   `estimated_mem = batch_size * num_generations * max_completion_length * model_params * 4`
   - 0.5B + batch=2 + gen=4 + len=256 ≈ 安全
   - 0.5B + batch=2 + gen=4 + len=512 ≈ 可能 OOM
   超出估算时: 自动降低 max_completion_length 或 num_problems

### 渐进式难度升级

调参时的难度调整增加渐进规则:

当前轮次 avg_reward >= 0.7 且 loss=0 时:
- 当前 difficulty=simple → 升级到 mixed
- 当前 difficulty=mixed (20% hard) → 升级到 mixed-hard (50% hard)
- 当前 difficulty=mixed-hard → 升级到 hard
- 同时增加 max_agent_steps: 3 → 5（给模型更多推理空间）

禁止直接从 mixed 跳到 hard:
- 轨迹证据: R3 mixed avg=0.77 → R4 hard avg=0.19（断崖下降）
- 需要中间级别 mixed-hard 作为过渡

difficulty 参数扩展:
- `"mixed-hard"`: 50% simple + 50% hard（新增，介于 mixed 和 hard 之间）

## 配置预检（生成配置后、启动前执行）

### 必检项（不通过则拒绝启动）

1. **TRL 整除约束**:
   `(batch_size * gradient_accumulation_steps) % num_generations == 0`
   失败时: 自动调整 num_generations

2. **模型安全上限**:
   查模型级别安全配置表，检查 lr/epochs/completion_length 是否超限
   失败时: 自动降到安全值并警告

3. **difficulty 合法性**:
   `difficulty in ("simple", "hard", "mixed")`
   失败时: 默认 "mixed"

### 建议检项（不通过则警告但允许启动）

4. **显存估算 (MPS)**:
   if `batch_size * num_generations * max_completion_length > 阈值`:
   警告: "可能 OOM，建议降低 max_completion_length 或 num_problems"

5. **训练时间估算**:
   `estimated_time = num_problems * num_epochs / (batch_size * grad_accum) * avg_step_time`
   if estimated_time > 30min:
   警告: "预计训练时间 Xm，确认继续？"

### 输出格式

生成 config.json 保存到 `rllm_train/output/runs/<run_id>/config.json`。

### 硬件信息字段

生成的 config.json 除了 TrainingConfig 标准字段外，附加以下硬件信息字段：

```json
{
  "num_gpus": 4,
  "gpu_type": "A100"
}
```

- `num_gpus`: 从用户输入中提取的 GPU 数量（未指定时为 1）
- `gpu_type`: GPU 型号，从用户输入中提取（A100/H100/4090 等，未指定时为空字符串）
- 这些字段不影响 TrainingConfig 解析（`from_json` 会过滤未知字段），仅供 rllm-run 读取

### GPU 相关预检

当 `num_gpus >= 1` 且 `gpu_type` 非空时，额外执行以下检查：

1. **显存可行性**: 查上方 GPU 显存估算表，检查 per_gpu 显存是否满足模型 + batch + generations 需求
2. **梯度检查点**: 7B 模型自动启用 `gradient_checkpointing: true`（减少 ~30% 显存）
