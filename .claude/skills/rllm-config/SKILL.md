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

### 0.5B 模型 GSM8K 参数安全范围（4轮实证）

基于 Qwen2.5-0.5B-Instruct 在 GSM8K 上 4 轮 GRPO 训练的实证参数范围:

| 参数 | 通用范围 | **0.5B GSM8K 推荐** | 实证依据 |
|---|---|---|---|
| learning_rate | 1e-7 ~ 1e-3 | **5e-7 ~ 2e-6** | R1 lr=5e-6 -> Step 17后崩溃; R2-4 lr=1e-6 -> 稳定训练 |
| max_completion_length | 128 ~ 2048 | **1024 ~ 2048** | R1 len=256 -> 88.3%截断; R2 len=1024 -> 截断降至45%; R3-4 len=1536 -> 截断<14% |
| num_generations | 2 ~ 8 | **6 ~ 8** | R1-2 gen=4 -> 有时advantage=0; R3-4 gen=8 -> advantage稳定(+/-0.935) |
| temperature | 0.3 ~ 1.5 | **0.8 ~ 1.1** | R1 temp=0.7 -> entropy快速下降; R3 temp=1.0 -> 探索充分; R4 temp=0.85 -> epoch2退化,需配合低lr |
| num_epochs | 1 ~ 20 | **1 ~ 3** | R4 epochs=2 -> epoch1上升, epoch2退化; 多epoch需降低lr或early stopping |
| num_problems | 8 ~ 512 | **32 ~ 128** | R1 64题 ok; R2 32题 49.2%; R3-4 48题 37-52%; 0.5B不需要海量数据 |

### 截断-崩溃预防规则

```python
# 0.5B GSM8K 配置安全检查
if task == 'gsm8k':
    assert config.max_completion_length >= 1024, "GSM8K需要长推理链, len<1024将导致截断崩溃(R1证据:88%)"
    assert config.num_generations >= 4, "gen<4可能导致GRPO advantage=0(R1-2证据)"
    if config.num_epochs > 1:
        assert config.learning_rate <= 1e-6, "多epoch需极低lr防止退化(R4证据: lr=1e-6+epochs=2仍有退化)"
```

### 0.5B 模型容量上限

15层诊断 L15 结论: 0.5B 在 GSM8K 单轮(无tool)场景下验证精度上限约 **49-52%**。
若高于此精度需求，必须: 换大模型(1.5B/7B) 或 启用多轮 tool-calling。

### 0.5B 模型 lr 安全范围更新 (R4 验证)

Round 1 第4轮实验验证: lr=5e-6 在稳定配置下安全且高效。
- R1-R3 (lr=1e-6): 最佳 reward=0.1964
- R4 (lr=5e-6): reward=0.5200, val-acc=0.488

在 ppo_mini_batch_size 正确计算和 max_response_length 安全的前提下，lr 安全上限可扩展至 1e-5。

### 0.5B num_problems 安全上限收紧 (Round 2 验证)

Round 2 num_problems=512 导致严重退化:
- PPO_KL=0 (policy未更新)
- val-acc=0.3343 (vs Round 1 0.488)
- Reward趋势下降

**0.5B 模型 num_problems 安全范围: 32 ~ 64**
推荐: 32-64问题 + 2-3 epochs深度训练

## 7B 模型参数安全范围

基于Round 2中3次Qwen2.5-7B-Instruct实验 (每次16-96步, 4×A800):

### 学习率
- **安全范围**: 1e-6 ~ 3e-6
- **下限**: 1e-6 (1e-7导致KL=0.001, 模型几乎不更新)
- **上限**: 3e-6 (5e-6导致entropy崩塌到0.038, reward hacking)
- **推荐**: 2e-6 (平衡学习速度和稳定性)

### Temperature
- **下限**: 0.5 (0.3导致7B熵快速塌缩)
- **推荐**: 0.6-0.8 (数学推理场景)
- 注意: 7B模型对低温比0.5B更敏感

### Entropy Coeff
- **必须启用**: entropy_coeff ≥ 0.001
- 7B模型参数空间大, 无约束时更容易坍缩到确定性模式

### Num Problems
- **推荐**: 128-256 (7B容量大于0.5B, 可处理更多问题)
- **上限**: 512 (未验证, 但基于0.5B教训应保守)

### 实验数据
| Run | LR | Temp | Entropy Coeff | Entropy Range | Score | 诊断 |
|-----|-----|------|---------------|---------------|-------|------|
| R2-R1 | 1e-7 | 0.3 | 0.0 | 0.275→0.438 | 0.446 | lr过低 |
| R2-R2 | 1e-6 | 0.3 | 0.001 | 0.289→0.150 | 0.485 | 可工作但慢 |
| R2-R3 | 5e-6 | 0.3 | 0.001 | 0.038→0.072 | 0.961⚠️ | 模式坍缩+reward hacking |

## 0.5B 模型参数安全范围

### num_problems
- **安全上限**: 64 (超过此值学习信号被噪声淹没)
- **推荐范围**: 32-48 (Round 1实证验证)
- **最大允许**: 128 (需配合entropy_coeff≥0.001 + epochs≥2)
- **禁止**: >256 (证实有害, val-acc单调下降)

### 设计理由
基于两轮训练实证对比:
- Round 1: 32 problems × 2 epochs × lr=5e-6 → val-acc=0.488 (stable, improving)
- Round 2: 512 problems × 1 epoch × lr=5e-6 → val-acc=0.334 (declining monotonically)

0.5B模型容量有限, 每个problem需要multi-epoch深度接触才能形成稳定学习信号。海量shallow contact导致:
1. 每step梯度主要由problem difficulty差异驱动, 非policy improvement
2. 模型无法建立跨step的持续学习轨迹
3. Entropy快速崩塌 (0.289→0.092, 低于0.1临界值)

## 0.5B num_problems 迭代建议

### 当前状态
R3实证: 48 problems + batch=2 + epochs=2 → **仅13步训练**
- 配置质量很高 (entropy稳定, val不再退化)
- 但训练步数不足, 限制了模型充分学习

### 迭代建议
```yaml
# Round 4 推荐
num_problems: 96     # 2x R3 (仍在安全上限128内)
batch_size: 2        # 保持 (R3验证batch=2稳定)
num_epochs: 2        # 保持
# 预期: ~24步/epoch × 2 = ~48步 (匹配R1的48步)
```

### 备选方案
```yaml
# 如果problems增加不可行, 增大batch
num_problems: 48     # 保持R3
batch_size: 4        # 2x R3
# 预期: ~24步 (但per-step梯度可能不够精细)
```

### 安全边界
- num_problems上限: 128 (基于R2教训: 512有害)
- batch_size上限: 4 (基于R2教训: 8过大)
- 优先增加problems而非batch (更多数据多样性 > 更大梯度batch)

## 0.5B num_problems 迭代: 48 → 96

### R3 验证结果
- 48 problems + entropy_coeff=0.001: entropy 稳定 (rate=0.0004)
- 但仅 13 步 (DataLoader OOM 中断) — 训练不充分

### 下一轮推荐
```yaml
num_problems: 96     # 2x R3, 仍在上限128内
batch_size: 2        # 保持 (R3验证稳定)
num_epochs: 2        # 保持
entropy_coeff: 0.001 # 保持 (强制, 已验证)
lr: 3e-6             # 保持 (已验证)
# 预期: 96/2×2 = ~96 batches total, ~48步/epoch
```

### 安全边界 (基于R2教训)
- 上限: 128 (R2: 512 → val崩塌)
- 推荐: 96 (2x R3, 提供充足步数)
- 下限: 48 (R3验证, 但步数不足)

## entropy_coeff 与 num_problems 联动规则 [VERL源码验证]

### 机制 (losses.py:128)
```python
policy_loss -= entropy_coeff * entropy_loss  # 每次step的探索压力
```

### 关键发现: 探索压力密度
entropy_coeff的绝对值不是关键 — **有效探索压力密度**才是:
```
effective_pressure = entropy_coeff × contact_frequency
contact_frequency = batch_size / num_problems
```

### 实证证据 (R3 vs R4, 0.5B模型)
| 轮次 | problems | coeff | contact_freq | effective_pressure | entropy | val-acc |
|------|----------|-------|-------------|-------------------|---------|---------|
| R3 | 48 | 0.001 | 2/48=0.042 | 4.2e-5 | ✅稳定 | 0.475 |
| R4 | 96 | 0.001 | 2/96=0.021 | 2.1e-5 | ⚠️振荡50%<0.1 | 0.439↓ |

### 联动规则
```yaml
# 0.5B 模型 entropy_coeff 推荐值
num_problems <= 48:  entropy_coeff = 0.001   # R3验证有效
num_problems 49-96:  entropy_coeff = 0.002   # 2x R4当前值, 恢复R3的有效密度
num_problems 97-128: entropy_coeff = 0.003   # 3x, 需验证
```

### 设计理由
R4的paradox (train score↑0.553→0.597但val↓0.475→0.439)证实:
- 相同的coeff在更多problems上产生的探索压力不足
- entropy在低值振荡→低entropy step中模型学到了"窄模式"
- 窄模式在train data上有效(train score↑)但在val data上不泛化(val↓)
- 这不是coeff太低的问题 — 是coeff与problems未联动

### 实现
config生成时:
```python
if num_problems <= 48:
    entropy_coeff = 0.001
elif num_problems <= 96:
    entropy_coeff = 0.002
else:
    entropy_coeff = 0.003
# 警告: entropy_coeff > 0.005 可能过度鼓励随机探索
```

## 大Batch模式: 平滑 Entropy 坍缩 [R8实证]

### 核心发现
R8首次证明: **entropy下降不一定有害**。关键在于下降是"平滑"还是"震荡"。

### 两种Entropy坍缩模式

| 模式 | Batch | Entropy行为 | Val结果 | 代表轮次 |
|------|-------|------------|---------|---------|
| **震荡型** | ≤8 | 上下振荡, >30% steps <0.1 | ↓ 退化 | R4 (batch=2, val↓) |
| **平滑型** | ≥16 | 单调下降, window-over-window稳定 | ↑ 改善 | R8 (batch=32, val↑) |

### 机制 (VERL源码)
```
core_algos.py:1138-1199 (agg_loss):
  small batch (2-8): gradient = avg of 2-8 problems → high noise
    → step N pulls toward mode A, step N+1 pulls toward mode B
    → entropy oscillates → model learns conflicting narrow modes
    → val degrades

  large batch (16-32): gradient = avg of 16-32 problems → low noise
    → each step represents consensus direction across many problems
    → entropy smoothly converges along consensus
    → model converges to a single effective mode
    → val improves
```

### R8 实证 (0.5B, batch=32, 10 epochs)
```
10-step window analysis:
  Window 1 (steps 1-10):   score=0.566  entropy=0.129  gn=0.96
  Window 2 (steps 11-20):  score=0.636  entropy=0.117  gn=1.00
  Window 3 (steps 21-30):  score=0.685  entropy=0.102  gn=0.92
  Window 4 (steps 31-40):  score=0.719  entropy=0.094  gn=0.93
  Window 5 (steps 41-50):  score=0.757  entropy=0.084  gn=0.89
  Window 6 (steps 51-60):  score=0.774  entropy=0.077  gn=0.86
  Window 7 (steps 61-70):  score=0.799  entropy=0.073  gn=0.88
  Window 8 (steps 71-80):  score=0.809  entropy=0.070  gn=0.92
  
  Score: monotonically increasing across all windows
  Entropy: monotonically decreasing, smooth (no oscillation)
  Grad norm: stable 0.86-1.00 (consistent updates)
  Val: 0.470 → 0.522 (improving throughout!)
```

### 配置建议
```yaml
# 大batch稳定训练配置 (R8验证)
batch_size: 32               # 大batch → 平滑梯度
entropy_coeff: 0.003         # 配合大batch (联动规则)
use_kl_loss: true            # KL约束防止策略偏离
kl_loss_coef: 0.01
kl_loss_type: low_var_kl     # 低方差KL估计
epochs: 10                   # 充分训练
lr: 2e-6                     # 配合大batch略降lr
GPU: 4                       # FSDP分布式
```

### 初始配置生成

根据用户输入的模型大小和任务类型，生成初始配置。需考虑：
1. 模型大小 → 影响 batch_size 和 max_completion_length
2. GPU 显存 → 单卡 vs 多卡
3. 数据来源 → 外部数据集不展示 difficulty，题目难度由数据集本身决定

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

### 0.5B GSM8K 推荐初始配置

基于 4 轮 GSM8K GRPO 训练的最优配置基线:

```python
# 0.5B GSM8K 推荐起手配置
gsm8k_0_5b_default = {
    "num_problems": 64,              # R1 64题稳定, R2 32题也够
    "num_epochs": 1,                 # 先单epoch验证稳定性
    "learning_rate": 1e-6,           # R2-4验证稳定; 5e-6会导致崩溃
    "temperature": 0.9,              # 0.85-1.0之间; 0.7太低导致entropy崩溃
    "num_generations": 8,            # R3-4验证: gen=8稳定advantage方差
    "batch_size": 1,                 # 0.5B单卡宽松, 但保持保守
    "max_completion_length": 1536,   # R3-4验证: 截断<14%; 最少1024
    "gradient_accumulation_steps": 4, # 等效batch=4
    "max_agent_steps": 4,            # GSM8K需要多步推理
    "max_grad_norm": 1.0,            # 梯度裁剪防止不稳定
}
```

#### 调参路线图

```
Round 1: 用上面默认配置建立基线
  -> 如果截断率>30%: max_completion_length += 512
  -> 如果Step后期崩溃: lr /= 2
  -> 如果pg_loss~0: num_generations += 2

Round 2+: 根据分析报告调整
  -> 如果Val平台+Train上升: 天花板, 换模型或加tool
  -> 如果clipfrac=0: lr *= 2, 或增加ppo_epochs
  -> 如果稳定: epochs += 1, lr /= 2
```

### 训练样本数调整建议

0.5B GSM8K 推荐 num_problems=256:
- R1 (64样本): 尽管有ppo_mini_batch问题，仍完成26步训练
- R3 (32样本): 82步稳定但reward上限仅0.1964
- R4 (32样本, lr提升): reward跃升至0.5200

256样本可提供更丰富的训练信号，在lr优化后获得更好泛化效果。0.5B模型训练时间短，256样本成本可控。

## 0.5B 模型推荐默认配置

基于Round 1和Round 2的实证数据, 0.5B模型的最佳保守配置:

```yaml
# 已验证有效的保守基线 (Round 1: val-acc=0.488)
model: Qwen2.5-0.5B-Instruct
learning_rate: 3e-6        # R1=5e-6有效但entropy下降偏快, 降为3e-6
num_problems: 48            # R1=32有效, R2=512有害, 48为折中
num_epochs: 2               # 多epoch精炼 > 单epoch海量数据
batch_size: 4               # R2=8导致等效batch过大
num_generations: 8          # GRPO最小group size=4, 8提供充足variance
temperature: 0.85           # R1/R2一致, 对0.5B数学推理有效
entropy_coeff: 0.001        # R2未启用导致entropy崩塌, 必须显式启用
max_response_length: 1536   # 足够的推理空间
gradient_checkpointing: false # 0.5B不需要, 增加开销无收益
```

### 关键约束
- **entropy_coeff必须>0**: Round 2实证entropy_coeff=0.0时entropy从0.29崩塌到0.092
- **lr上限**: 5e-6对0.5B偏高 (entropy下降率0.0016/step), 3e-6更安全
- **epochs≥2**: 单epoch = shallow contact → val divergence

## R3 验证安全基线 (2026-06-01)

基于Round 3实证 (session: ae18fdb6), 以下配置被验证为0.5B模型的安全有效基线:

```yaml
# 验证安全基线 (R3: val-acc=0.475, entropy稳定)
model: Qwen2.5-0.5B-Instruct
learning_rate: 3e-6          # R2优化推荐, R3验证有效
num_problems: 48             # R2优化推荐上限64, R3用48安全
num_epochs: 2                # 多epoch精炼
batch_size: 2                # 保守, per-step stability优先
num_generations: 8           # GRPO group size
temperature: 0.7             # 配合entropy_coeff=0.001
entropy_coeff: 0.001         # R2关键发现: 必须启用
max_response_length: 1536
max_prompt_length: 512
gradient_checkpointing: false
backend: verl
```

### 实证指标
| 指标 | R1 (baseline) | R2 (degraded) | R3 (this config) |
|------|--------------|---------------|------------------|
| Val-acc | 0.488 | 0.334 | **0.475** |
| Entropy stability | 0.279→0.218 | 0.289→0.092 | **0.157→0.152** |
| Entropy decline rate | 0.0013 | 0.0015 | **0.0004** |
| PPO_KL | 0.0 | 0.0 | **0.000433** |
| Clipfrac | 0.0 | 0.0 | **0.0053** |
| Grad norm | 2.46 | 1.60 | **2.41** |
| Train score mean | 0.520 | 0.464 | **0.553** |

### 与R2优化建议的对应
- R2 Patch 1 (0.5B上限64): ✅ problems=48 生效
- R2 Patch 2 (lr=3e-6, entropy_coeff=0.001): ✅ 直接采纳, 效果显著
- R2 Patch 3 (entropy监控): ✅ entropy稳定在0.15附近

### 已知限制
- 训练步数过短 (仅13步): 建议下一轮增加num_problems到96
- 基础设施偶发OOM (DataLoader worker, 非GPU OOM)

## entropy_coeff 强制默认值 [VERL源码验证]

### 机制 (losses.py:123-129)
```python
if entropy is not None:
    entropy_loss = agg_loss(entropy, response_mask, loss_agg_mode)
    policy_loss -= entropy_coeff * entropy_loss  # ← 唯一的entropy保护!
```

### 强制规则
- **0.5B 模型: entropy_coeff 必须 >= 0.001 (不可为0)**
- **7B 模型: entropy_coeff 必须 >= 0.0005 (不可为0)**
- 配置生成时检查: 如果 entropy_coeff=0 → **ERROR: 拒绝生成配置**

### 实证证据 (3轮对比)
```
         entropy_coeff  entropy trend          decline_rate  val-acc
R1 (0.5B)     0         0.279→0.218             0.0013       0.488
R2 (0.5B)     0         0.289→0.092 ⚠️崩塌      0.0015       0.334
R3 (0.5B)     0.001     0.157→0.152 ✅稳定      0.0004       0.475
```

### 为什么必须强制
1. GRPO 无 critic → 无 value-based exploration bonus
2. 无 entropy_coeff 时, 策略自然趋向确定性 (entropy 单调下降)
3. R1 在 48 步下降 0.06, R2 在 128 步下降 0.20
4. R3 启用后, entropy 几乎水平 (Δ=0.005 in 12 steps)
5. **entropy 崩塌是不可逆的** — 一旦策略坍缩, 无法通过继续训练恢复

### 7B 模型特别说明
R2 的 3 次 7B 实验中, 即使 entropy_coeff=0.001 (R2-R2, R2-R3), entropy 仍在下降.
7B 对低温 (temp=0.3) 更敏感, 建议 temperature >= 0.5 配合 entropy_coeff >= 0.001.

### 数据源作用域隔离（必须优先执行）

生成或调参配置前，先判定数据源作用域:

| 条件 | 数据源类型 | 可使用的经验规则 |
|---|---|---|
| `dataset_path` 为空 | 内置合成数据 | 可使用 simple/mixed/hard、0.5B mixed problem-count、difficulty scaling 等合成数据经验 |
| `dataset_path` 非空 | 外部 HuggingFace Dataset | 不使用合成数据 difficulty 经验；只根据当前 run 的 config/log/trajectory/analysis 调参 |
| `dataset="deepscaler"` 或路径含 deepscaler | DeepScaler 外部数据集 | 不使用 simple/mixed/hard 调参；只调整通用训练参数和当前样本选择参数 |

硬性规则:
1. 外部数据集下，`difficulty` 不展示、不写入新配置，也不会改变 `load_external_dataset()` 的样本分布。除非训练代码已实现基于数据集字段的过滤，否则不得声称 mixed/hard 会改变 DeepScaler 难度。
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

### 调参策略扩展（4轮GSM8K实证诊断）

以下模式是当前 tuning 策略未覆盖但实际发生的关键问题:

#### 新模式 1: Policy 不更新 (PPO_KL ~ 0, clipfrac = 0)

**症状**: PPO KL divergence 接近 0, clip ratio 始终为 0, train reward 不提升或极慢
**根因**: lr 过低 (如 1e-6) 或 ppo_epochs 不足, 导致 policy 几乎无更新
**修复**:
- 增大 lr (0.5B: 1e-6 -> 5e-6, 观察是否崩溃; 7B: 维持 1e-6 ~ 2e-6)
- 增大 ppo_epochs (默认1 -> 2-4)
- 注意: lr 过高会导致 Step 后期崩溃 (R1: lr=5e-6, Step 17-32 -> reward=0)

#### 新模式 2: 高截断率 (>30%)

**症状**: avg_response_len 持续达到 max_completion_length, reward=0 的比例高
**根因**: max_completion_length 不足, 模型推理链被截断
**修复**:
- **优先增大 max_completion_length** (R1: 256->1024, 截断88%->45%; R3: 1536, 截断<14%)
- 不要先降 temperature — 截断的根因是长度限制, 不是采样温度
- gsm8k 推荐 max_len >= 1024; 复杂推理任务推荐 >= 1536

#### 新模式 3: 验证平台 + 训练上升 (容量天花板)

**症状**: Train reward 持续上升但 Val accuracy 卡在某个值不涨 (R2-4: Train 166->258->291, Val~49%)
**根因**: 模型容量不足以学习更复杂的推理模式
**修复**:
- 检查是否为容量天花板: 连续 2+ round Val 不涨 + Train 上升 -> 天花板
- 触及天花板时: 换大模型 或 启用多轮 tool-calling, 继续调参无效
- 0.5B GSM8K 天花板 ~49%; 1.5B 预期 ~60-70%; 7B 预期 ~80-90%

#### 新模式 4: Epoch 间退化

**症状**: 多 epoch 训练时, epoch 1 上升但 epoch 2+ reward 下降 (R4: epoch1 up, epoch2 down)
**根因**: 多 epoch 导致过拟合或 policy 更新过度
**修复**:
- 多 epoch 时降低 lr (R4: lr=1e-6 仍有退化, 建议 5e-7)
- 添加 early stopping: Val reward 连续下降则停止
- 或增大数据量(num_problems)替代多 epoch

#### 新模式 5: GRPO Advantage 方差不足

**症状**: pg_loss 接近 0, 模型几乎不学习
**根因**: num_generations 太少, advantage 计算失效
**修复**:
- 增大 num_generations (R1-2: 4->R3-4: 8, advantage 方差恢复到 +/-0.935)
- 最少 4, 推荐 8 (若显存允许)

### Policy 未更新 (PPO_KL = 0) 诊断

**症状**: PPO KL divergence 始终为 0, clipfrac = 0
**根因**: num_problems过大稀释梯度信号
**修复**: 减少num_problems(0.5B:max64), 增大lr, 确保2+epochs

**Round 2 案例**: 512 problems, lr=5e-6 -> PPO_KL=0, val=0.3343

## 验证背离检测

### 监控规则
在训练过程中, 每 test_freq 步检查 val 指标:

```python
val_history = []  # [(step, val_acc)]

def check_val_divergence(val_history):
    if len(val_history) < 4:
        return  # 需要至少4个数据点
    recent = val_history[-4:]
    # 检查连续3次下降
    declining_count = sum(1 for i in range(len(recent)-1) if recent[i][1] > recent[i+1][1])
    if declining_count >= 3:
        first_val = val_history[0][1]
        last_val = val_history[-1][1]
        drop_pct = (first_val - last_val) / first_val * 100
        return (
            f"ALERT: val-acc连续下降 ({first_val:.3f} -> {last_val:.3f}, -{drop_pct:.1f}%). "
            f"建议: 检查是否过拟合, 考虑减少num_problems或增加epochs/entropy_coeff."
        )
```

### 触发阈值
- 连续3次val检查下降 → **ALERT**: 验证背离
- 下降幅度 >20% → **EMERGENCY**: 建议停止训练

### Case Study: Round 2验证背离
```
Step 10:  0.444 ← 起始
Step 20:  0.376 ↓
Step 40:  0.400 ↑
Step 60:  0.391 ↓
Step 80:  0.400 ↑
Step 100: 0.359 ↓
Step 110: 0.318 ↓ (连续3降→触发ALERT)
Step 128: 0.334 ↓ (最终-25%)
```
如果在Step 110触发告警, 可避免最后18步的无效训练.

## Entropy崩塌监控规则

### 配置验证阶段
- 如果 `entropy_coeff` 未配置或等于 0:
  → **WARN**: "entropy_coeff=0, entropy可能在训练中崩塌. 建议启用entropy_coeff≥0.001."
  → 引用: Round 2 entropy_coeff=0 → entropy 0.289→0.092

### 训练监控阈值
- `actor/entropy` < 0.15:
  → **WARN**: "Entropy低于0.15警戒线, 探索空间收窄. 检查temperature和entropy_coeff配置."
- `actor/entropy` < 0.10:
  → **EMERGENCY**: "Entropy低于0.10危险阈值, 策略正在坍缩为确定性输出. 建议: 立即停止训练, 增大entropy_coeff(>0.001), 提升temperature(>0.85)."
- `actor/entropy` < 0.05:
  → **CRITICAL**: "Entropy极低, 模型已坍缩. 停止训练, 回滚checkpoint, 大幅调整配置."

### 参考基线
- Round 1 (健康): entropy 0.279→0.218 (48步, Δ=-0.06, 下降率0.0013/step)
- Round 2 (崩塌): entropy 0.289→0.092 (128步, Δ=-0.20, 下降率0.0016/step)
- 崩塌临界值: entropy < 0.10 (诊断参考框架)

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

配置必须包含 package/task 元数据：
- `task_id`: 当前训练任务 ID；未提供时使用 run_id
- `skill_package_id`: 当前使用的 skill package ID；优先使用编排者传入值，其次由 `TrainingConfig` 从 registry 推断
- `skill_package_manifest`: package manifest 路径；可留空，由 `TrainingConfig` 自动补齐

如果用户或编排者显式传入 `skill_package_id`，不得改写为其他 package。

### 硬件信息字段

生成的 config.json 除了 TrainingConfig 标准字段外，附加以下硬件信息字段：

```json
{
  "num_gpus": "auto",
  "gpu_type": "auto"
}
```

- `num_gpus`: 用户明确指定 GPU 数量时写整数；未指定时必须写 `"auto"`，不得默认写死为 1
- `gpu_type`: 用户明确指定 GPU 型号时写型号（A100/H100/4090 等）；未指定时写 `"auto"`
- `resolved_num_gpus`: 不由 rllm-config 写入；由 rllm-run 在启动前探测当前机器后写回本轮 config 或 run metadata
- `resolved_gpu_type`: 不由 rllm-config 写入；由 rllm-run 在启动前探测当前机器后写回
- 这些字段不影响 TrainingConfig 解析（`from_json` 会过滤未知字段），仅供 rllm-run 读取

### GPU 相关预检

当 `num_gpus` 为整数且 `gpu_type` 非空/非 `"auto"` 时，额外执行以下检查：

1. **显存可行性**: 查上方 GPU 显存估算表，检查 per_gpu 显存是否满足模型 + batch + generations 需求
2. **梯度检查点**: 7B 模型自动启用 `gradient_checkpointing: true`（减少 ~30% 显存）

当 `num_gpus="auto"` 时，rllm-config 只做模型级安全参数校验，不假设实际卡数；实际使用多少卡由 rllm-run 在启动阶段动态决定。

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
