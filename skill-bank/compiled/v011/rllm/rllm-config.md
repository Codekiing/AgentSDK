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

当 difficulty=mixed 时，0.5B 模型的 num_problems 安全上限进一步收紧:

| 参数 | 原上限 | 新上限 | 条件 | 依据 |
|------|--------|--------|------|------|
| num_problems | 64 | 32 | difficulty=mixed 且 model=0.5B | lr=5e-6 + 1 epoch + 64 problems 仍在 step 9 开始 forgetting |

推荐配置 (0.5B + mixed):
- num_problems=32, lr=5e-6, epochs=1, batch=2, generations=4
- 预期: 32 步训练，reward 稳定不崩溃

替代方案: 保持 64 problems 但切换 difficulty=simple
- 适用于需要更多训练数据但不需要 hard 题目的场景

### Seed 随机化策略

多轮训练时（traj-loop 或手动多轮）:
- 每轮使用不同 seed: `seed = base_seed + round_number`
- 或启用 dataset shuffle: `shuffle=True`
- 目的: 避免相同问题固定在相同 step，导致零 reward 步骤的周期性模式

轨迹证据:
- R3 和 R5 使用相同 seed=42，零 reward 步骤完全一致 [5,6,12,25,31]
- 训练未改善模型在这些特定问题上的表现
- 变更 seed 可以让模型接触不同的问题排列，获得更多样的学习信号

### num_problems 精细化范围 (0.5B 模型)

基于 5 轮训练数据更新推荐范围:

| difficulty | 推荐范围 | 依据 |
|-----------|---------|------|
| mixed (20% hard) | 40-48 | 32 太简单 (loss=0), 64 forgetting |
| mixed-hard (50% hard) | 24-32 | hard 比例增加后需减少总量 |
| hard | 16-24 | 64 完全超出能力 (avg=0.19) |

默认推荐配置 (0.5B + 正式训练):
- num_problems=48, difficulty=mixed, lr=5e-6, epochs=1
- 预期: 比 32 problems 更有挑战性，但不会 forgetting

轨迹证据:
- R1/R2 (64p, mixed): catastrophic forgetting at step 14-16
- R3/R5 (32p, mixed): loss=0, 无学习效果
- 推断: 最优点在 32-64 之间，推荐 40-48

### num_problems 最优范围精细化 (0.5B + mixed, 基于 2 轮数据)

2 轮训练数据收敛出更精确的推荐范围:

| num_problems | 结果 | 证据 |
|-------------|------|------|
| 32 | 无 forgetting 但 loss=0 (无学习) | run_1777723566: avg=0.773, loss=0 全程 |
| 48 | avg=0.849 但 step 41-47 格式退化 | run_1777726900: 后 20% avg=0.475 |
| 40 (推断) | 平衡点 | 32 太简单, 48 后期崩溃 |

更新推荐:
- 首轮训练: num_problems=40 (安全起点)
- avg_reward >= 0.85 且无后期 forgetting: 可尝试 44
- 出现后期 forgetting: 减少到 36
- 禁止 0.5B+mixed 使用 num_problems > 48

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
| difficulty | hard | 7B base model 在 mixed 上 reward 已达 0.773, 需提高难度才能产生学习信号 |
| batch_size | 1 | 单卡显存限制, batch=2 + gen=4 会 OOM |
| learning_rate | 2e-6 ~ 5e-6 | 7B 模型需要较低的 lr 避免 grad norm 爆炸 |

#### 显存预估

| num_generations | batch_size | 预计显存 | 是否可行 |
|----------------|-----------|---------|---------|
| 2 | 1 | ~30GB | 可行但无学习效果 |
| 4 | 1 | ~50GB | 可行（推荐） |
| 4 | 2 | ~75GB | 临界, 可能 OOM |
| 2 | 2 | ~55GB | 可行但无学习效果 |

### 初始配置生成

根据用户输入的模型大小和任务类型，生成初始配置。需考虑：
1. 模型大小 → 影响 batch_size 和 max_completion_length
2. GPU 显存 → 单卡 vs 多卡
3. 任务难度 → num_problems 和 difficulty

### 难度配置

difficulty 参数控制训练数据的难度分布:
- `"simple"`: 100% 简单两数运算 (适合流程验证)
- `"hard"`: 100% 多步骤应用题 (0.5B 模型几乎无法学会)
- `"mixed"`: 80% simple + 20% hard (推荐，提供学习信号的同时引入挑战)

初始配置推荐:

| 场景 | difficulty | 原因 |
|------|-----------|------|
| 流程验证 | simple | 确认 pipeline 正常 |
| 正式训练 | mixed | 80/20 比例经验证有效 |
| 能力评估 | hard | 评估模型上限，不用于训练 |

调参时的难度调整:
- reward=1.0 + loss=0 → 题目太简单，切换到 mixed 或 hard
- reward<0.1 + difficulty=hard → 太难，切换到 mixed
- mixed 下 reward 在 0.3-0.7 → 比例合适，保持不变

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
