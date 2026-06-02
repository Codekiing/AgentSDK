# verl RL Training Parameter Reference (调参指南)

> **适用版本**: verl main branch (2026-05)
> **语言约定**: 参数名保留英文，描述中英双语
> **组织方式**: 按功能模块分类，每个模块包含参数速查表 + 关键参数详解

---

## 目录

1. [Algorithm Configuration (算法配置)](#1-algorithm-configuration-算法配置)
2. [Actor Policy Configuration (策略模型配置)](#2-actor-policy-configuration-策略模型配置)
3. [Rollout Configuration (采样/推理性配置)](#3-rollout-configuration-采样推理性配置)
4. [Data Configuration (数据配置)](#4-data-configuration-数据配置)
5. [Reward Configuration (奖励配置)](#5-reward-configuration-奖励配置)
6. [Trainer Configuration (训练器配置)](#6-trainer-configuration-训练器配置)
7. [Optimizer Configuration (优化器配置)](#7-optimizer-configuration-优化器配置)
8. [Multi-Turn / Agent Configuration (多轮对话/智能体配置)](#8-multi-turn--agent-configuration-多轮对话智能体配置)
9. [Rollout Correction (Off-Policy 修正配置)](#9-rollout-correction-off-policy-修正配置)
10. [Router Replay Configuration (MoE 路由回放配置)](#10-router-replay-configuration-moe-路由回放配置)
11. [Parameter Quick Reference by Scenario (场景速查)](#11-parameter-quick-reference-by-scenario-场景速查)

---

## 1. Algorithm Configuration (算法配置)

**Config path**: `algorithm.*`
**Dataclass**: `AlgoConfig` in [algorithm.py](../verl/trainer/config/algorithm.py)
**Core code**: [core_algos.py](../verl/trainer/ppo/core_algos.py)

### 1.1 参数速查表

| Parameter (参数名) | Default | Range/Options (推荐范围) | Category | 关键问题 |
|---|---|---|---|---|
| `adv_estimator` | `gae` | `gae`, `grpo`, `grpo_vectorized`, `rloo`, `rloo_vectorized`, `reinforce_plus_plus`, `grpo_passk`, `gdpo`, `opo`, `gpg` | 算法选择 | 用什么方式估计 advantage |
| `norm_adv_by_std_in_grpo` | `True` | `True` / `False` | GRPO 变体 | 是否用 std 归一化 advantage |
| `gamma` | `1.0` | `0.9-1.0` | GAE 参数 | 未来奖励的折扣因子 |
| `lam` | `1.0` | `0.95-1.0` | GAE 参数 | GAE 的 bias-variance 权衡 |
| `use_kl_in_reward` | `False` | `True` / `False` | KL 控制 | KL 惩罚加在 reward 上还是 loss 上 |
| `kl_penalty` | `kl` | `kl`, `abs`, `mse`, `low_var_kl`, `full` | KL 控制 | KL 散度的估计方式 |
| `kl_ctrl.type` | `fixed` | `fixed`, `adaptive` | KL 控制 | KL 系数是否自适应调整 |
| `kl_ctrl.kl_coef` | `0.001` | `0.0-0.1` | KL 控制 | KL 惩罚的初始系数 |
| `kl_ctrl.target_kl` | `0.1` | `0.01-1.0` | KL 控制 | 自适应控制的目标 KL 值 |
| `kl_ctrl.horizon` | `10000` | `100-100000` | KL 控制 | 自适应控制的时间窗口 |
| `filter_groups.enable` | `False` | `True` / `False` | DAPO | 是否启用组过滤 (Dynamic Sampling) |
| `filter_groups.metric` | `null` | `acc`, `score`, `seq_reward`, `seq_final_reward` | DAPO | 过滤所用的指标 |
| `filter_groups.max_num_gen_batches` | `0` | `0-N` (0=无上限) | DAPO | 最多生成多少批来凑够有效组 |
| `use_pf_ppo` | `False` | `True` / `False` | 偏好学习 | 是否启用 Preference Feedback PPO |
| `gdpo_reward_keys` | `null` | e.g. `["format_reward", "accuracy_reward"]` | GDPO | 多维度 reward 的 key 列表 |
| `gdpo_reward_weights` | `null` | e.g. `[0.3, 0.7]` | GDPO | 多维度 reward 权重 |

### 1.2 关键参数详解

#### `adv_estimator` — Advantage 估计算法选择 (Advantage Estimator)

**解决的问题**: 如何从稀疏的 outcome reward (每个 response 一个分数) 或稠密的 token-level reward 计算出每个 token 的 advantage，用于策略梯度更新。

**算法对比**:

| Estimator | 原理 | 适用场景 | 优缺点 |
|---|---|---|---|
| `gae` | Generalized Advantage Estimation: 需要 Critic (Value Network) 估计 V(s)，用 TD-error 和 GAE 公式计算 advantage | 有 Critic 的 PPO 训练 | 需要额外训练 Critic，显存开销大，优势估计低方差 |
| `grpo` | Group Relative Policy Optimization: 同一 prompt 的 N 个 response 组成 group，advantage = (r_i - μ_group) / σ_group | **GRPO 首选**，math/coding reasoning | 无需 Critic，简单高效；group size 太小方差大 |
| `grpo_vectorized` | GRPO 的向量化实现，速度更快 | 同 GRPO，推荐替代 `grpo` | 与 `grpo` 等价，计算更快 |
| `rloo` | **Leave-One-Out**: advantage = r_i - mean(r_{j≠i}) | 需要低方差的 GRPO 场景 | 相比 GRPO 减少 bias，方差略低 |
| `rloo_vectorized` | RLOO 的向量化实现 | 同 RLOO，推荐替代 `rloo` | 与 `rloo` 等价 |
| `reinforce_plus_plus` | REINFORCE++ 算法 | 简单 RL 场景 | 无需 group，无需 Critic |
| `grpo_passk` | Pass@K GRPO: 只给 group 内最佳 response 正 advantage | 评测用 Pass@K 的场景 | 鼓励 model 专注于最佳解 |
| `gdpo` | Group Decoupled Normalization: 多维度 reward 分别做 group normalization 后加权 | **多维度 reward 场景** (如 format + accuracy) | 防止某个 reward 维度主导梯度 |
| `opo` | Outcome-based Policy Optimization: 用 response length 加权 baseline | 需要长度正则化的场景 | 减少长度偏差 |
| `gpg` | Guided Policy Gradient | 探索性算法 | 有 Critic 但不需要 GAE 的中间方案 |

**推荐选择**:
- **数学/代码推理 (cold start, 单一 outcome reward)**: `grpo_vectorized` 或 `rloo_vectorized`
- **多维度 reward (format + accuracy + ...)**: `gdpo`，配合 `gdpo_reward_keys`
- **有 Critic 的标准 PPO**: `gae`
- **Agent 多步任务 (如 SWE Agent)**: `rloo_vectorized` (DeepSWE 的选择)

**与其它参数交互**:
- `rollout.n` 影响 group size: group 越大 advantage 估计越准，但计算成本越高
- `norm_adv_by_std_in_grpo` 仅在 `grpo` / `grpo_vectorized` 下生效

---

#### `norm_adv_by_std_in_grpo` — GRPO 中是否用标准差归一化 Advantage (Normalize Advantage by Std in GRPO)

**解决的问题**: 原始 GRPO 对 group 内 reward 做 z-score 归一化 `(r - μ) / σ`。但除以 σ 会**消除难易问题的区分度**：简单题和难题的 advantage 被缩放到同一尺度，模型无法识别哪些任务更值得学习。此外当 group 内所有 response 分数相同时 σ=0，导致数值不稳定。

**配置指南**:

| 值 | 公式 | 效果 | 来源 |
|---|---|---|---|
| `True` (default) | `A_i = (r_i - μ) / σ` | 原始 GRPO：所有任务等权重 | DeepSeek-R1 |
| `False` | `A_i = r_i - μ` | **Dr.GRPO**：保留难度信号 | Dr.GRPO, DeepSWE |

**调参建议**:
- 如果你的 reward 是 0/1 二值的 (如 SWE 的 pass/fail，数学的对/错)，设为 `False` 往往更好，因为保留难度信号有助于课程学习
- 如果你的 reward 有较大动态范围 (如 0-10)，保留 `True` 可以在 group 内做合适的 scaling
- DeepSWE、DAPO 都使用 `False`

---

#### `use_kl_in_reward` vs `use_kl_loss` — KL 散度控制的两种方式 (KL Control Mechanism)

**解决的问题**: 防止 RL 训练中策略偏离初始模型太远导致模型退化 (reward hacking, 语言质量下降)。

**两种机制对比**:

| 维度 | `use_kl_in_reward=True` | `use_kl_loss=True` |
|---|---|---|
| Config path | `algorithm.use_kl_in_reward` | `actor.use_kl_loss` |
| 机制 | KL 作为 reward penalty: `r_effective = r - β·KL(π||π_ref)` | KL 作为 loss term: `L_total = L_PG + β·KL(π||π_ref)` |
| 梯度流向 | 通过 advantage 间接影响 | 直接加入 loss，反向传播 |
| 适用场景 | 标准 PPO (有 Critic) | **GRPO 系列** (无 Critic) |
| 需要 ref model | 是 | 是 |

**调参建议**:
- GRPO 场景用 `use_kl_loss=True` + `kl_loss_coef` (不要用 `use_kl_in_reward`)
- PPO + Critic 场景可以用 `use_kl_in_reward=True`
- DeepSWE 的做法: **两者都关掉** (`kl_loss_coef=0`, `use_kl_in_reward=False`) — 因为从 base model 开始训练，没有 SFT model 约束，KL loss 反而限制探索

---

#### `kl_loss_type` — KL 散度的估计方法 (KL Divergence Estimator)

**解决的问题**: 不同 KL 估计器数值性质不同，影响训练稳定性和计算开销。

| Type | 公式 | 性质 | 推荐场景 |
|---|---|---|---|
| `kl` (k1) | `log(p/q)` | 无偏但高方差，可为负 | 一般不推荐 |
| `abs` | `|log(p/q)|` | 非负，但不光滑 | 实验性 |
| `mse` (k2) | `0.5 * (log(p/q))^2` | 平方惩罚，稳定 | 标准 PPO |
| `low_var_kl` (k3) | `p/q - log(p/q) - 1` | **低方差，始终 ≥0** | **推荐默认值**，GRPO 首选 |
| `full` | 完整 KL 估计 | 最准确但计算最贵 | 研究/对比实验 |

**推荐**: 一般保持默认 `low_var_kl` (k3 估计器)，数值最稳定。

---

#### `filter_groups` — DAPO 风格的组过滤 (Dynamic Sampling / Group Filtering)

**解决的问题**: GRPO 训练中，有些 group 内所有 response 都正确 (or 都错误)，这些组的 advantage 方差为 0，不提供有效学习信号。Dynamic Sampling 过滤掉这些无效组，重新采样直到凑够有效训练数据。

**配置**:
```yaml
algorithm:
  filter_groups:
    enable: True           # 启用组过滤
    metric: "seq_reward"   # 用什么指标判断组内方差
    max_num_gen_batches: 10  # 最多重采样 10 次
```

**适用场景**: reward 稀疏且容易全部正确/全部错误的场景。DAPO 论文中展示了它的效果。DeepSWE 没用这个机制（用了 compact filtering 替代）。

---

## 2. Actor Policy Configuration (策略模型配置)

**Config path**: `actor_rollout_ref.actor.*`
**Dataclass**: `ActorConfig` in [actor.py](../verl/workers/config/actor.py)
**Core code**: [losses.py](../verl/workers/utils/losses.py) (`ppo_loss()`)

### 2.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `clip_ratio` | `0.2` | `0.1-0.3` | PPO Clipping | PPO 对称 clipping 参数 ε |
| `clip_ratio_low` | `0.2` | `0.1-0.2` | PPO Clipping | 下界 clipping (asymmetric) |
| `clip_ratio_high` | `0.2` | `0.2-0.4` | PPO Clipping | 上界 clipping (DAPO clip-higher) |
| `clip_ratio_c` | `3.0` | `2.0-5.0` | PPO Clipping | Dual-clip PPO 下界常数 |
| `loss_agg_mode` | `token-mean` | `token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean`, `seq-mean-token-sum-norm` | Loss 聚合 | loss 如何在 batch/sequence 维度聚合 |
| `loss_scale_factor` | `null` | `null` or constant (e.g. `8192`) | Loss 聚合 | `seq-mean-token-sum-norm` 的归一化分母 |
| `entropy_coeff` | `0` | `0.0-0.01` | 正则化 | 熵正则化系数 |
| `calculate_entropy` | `false` | `true` / `false` | 正则化 | 是否计算熵 (需要模型支持) |
| `use_kl_loss` | `false` | `true` / `false` | KL 控制 | 是否用 KL loss (替代 KL reward penalty) |
| `kl_loss_coef` | `0.001` | `0.0-0.1` | KL 控制 | KL loss 的系数 β |
| `kl_loss_type` | `low_var_kl` | `kl`, `abs`, `mse`, `low_var_kl`, `full` | KL 控制 | KL 估计器类型 |
| `ppo_epochs` | `1` | `1-5` | 训练循环 | 每个 batch 训练几个 epoch |
| `shuffle` | `false` | `true` / `false` | 训练循环 | 跨 epoch 是否 shuffle 数据 |
| `ppo_mini_batch_size` | `256` | `64-512` | Batch | PPO mini batch 大小 |
| `ppo_micro_batch_size_per_gpu` | `null` | e.g. `4`, `8`, `16` | Batch | 每 GPU 的 micro batch 大小 |
| `use_dynamic_bsz` | `false` | `true` / `false` | Batch | 是否启用动态 batch size |
| `ppo_max_token_len_per_gpu` | `16384` | `4096-65536` | Batch | 每 GPU 最大 token 数 (动态 batch) |
| `tau_pos` | `1.0` | `1.0-2.0` | SAPO | SAPO 正奖励平滑参数 |
| `tau_neg` | `1.05` | `1.0-2.0` | SAPO | SAPO 负奖励平滑参数 |
| `policy_loss.loss_mode` | `vanilla` | `vanilla`, `clip-cov`, `kl-cov`, `gpg` | 策略损失 | 策略损失函数变体 |
| `use_torch_compile` | `true` | `true` / `false` | 性能 | 是否用 torch.compile 加速 |
| `use_prefix_grouper` | `false` | `true` / `false` | 性能 | 是否启用 shared-prefix 批量前向 |
| `data_loader_seed` | `42` | any integer | 可复现性 | mini-batch 构造的随机种子 |
| `grad_clip` | `1.0` | `0.5-5.0` | FSDP 优化 | 梯度裁剪阈值 |

### 2.2 关键参数详解

#### `clip_ratio` / `clip_ratio_high` / `clip_ratio_low` — PPO Clipping 参数 (Policy Clipping)

**解决的问题**: PPO 通过 clipping 限制每次更新的幅度，防止策略变化过大导致训练不稳定。原始 PPO 使用对称 clipping `[1-ε, 1+ε]`。

**三种 clipping 模式**:

| 配置 | 效果 | 来源 |
|---|---|---|
| `clip_ratio=0.2`, `clip_ratio_low=0.2`, `clip_ratio_high=0.2` | 对称 clipping `[0.8, 1.2]` | 原始 PPO |
| `clip_ratio_low=0.2`, `clip_ratio_high=0.28` | 不对称 clipping，提高上界 | **DAPO/DeepSWE**: Clip-higher |
| `clip_ratio_low=0.15`, `clip_ratio_high=0.25` | 更宽松的不对称 clipping | 大模型/后期调参 |

**Clip-higher 的原理** (DeepSWE/DAPO):
标准对称 clipping 在 reward 稀疏的场景下，当模型偶然获得正 reward 时，等比例地限制了概率上升。提高 `clip_ratio_high` (`[0.8, 1.28]` vs `[0.8, 1.2]`) 允许正样本更激进地增大概率 → 鼓励探索、稳定熵。

**`clip_ratio_c` (Dual-clip PPO)**: 当 advantage < 0 且 ratio > `clip_ratio_c` 时，进一步 clip loss 为 `-A * clip_ratio_c`，防止极端负样本主导梯度。

**推荐**:
- 标准 GRPO: `clip_ratio=0.2`, `clip_ratio_low=0.2`, `clip_ratio_high=0.2`
- **GRPO++ (推荐)**: `clip_ratio=0.2`, `clip_ratio_low=0.2`, `clip_ratio_high=0.28`
- 训练不稳定时: 降低 `clip_ratio` 到 `0.1`

---

#### `loss_agg_mode` — Loss 聚合模式 (Loss Aggregation Mode)

**解决的问题**: 如何处理 batch 内不同长度序列的 loss 聚合，直接影响训练的动态——长序列是否获得更多梯度权重。

| Mode | 公式 | 效果 | 适用场景 |
|---|---|---|---|
| `token-mean` | `sum(loss * mask) / total_valid_tokens` | **每个 token 等权重**，长序列贡献更多梯度 | 传统的 PPO/GRPO |
| `seq-mean-token-sum` | `mean( sum(loss * mask per seq) )` | **每个序列等权重**，但序列内 token 越多 loss 越大 | 需要均衡序列权重 |
| `seq-mean-token-mean` | `mean( mean(loss * mask per seq) )` | **每个序列等权重**，序列内也归一化 | 极端长度差异场景 |
| `seq-mean-token-sum-norm` | `mean(sum / seq) / scale_factor` | 序列内 sum 后**除以长度**再组间平均 | **Dr.GRPO/DeepSWE**: 消除长度偏差 |

**DeepSWE 的选择** (`seq-mean-token-sum-norm`):

在 SWE agent 场景，错误轨迹往往更长 (反复尝试、兜圈子)。`token-mean` 会给长错误轨迹更大的梯度权重 → 模型被诱导生成越来越长的错误响应。

`seq-mean-token-sum-norm` 将每个序列的 loss 除以其长度，确保长短序列对梯度的贡献等权重 → 消除长度偏差。

**`loss_scale_factor`**: 在 `seq-mean-token-sum-norm` 模式下控制归一化分母。设为 `null` 自动使用 `response_length`；设为常数 (如 `8192`) 确保整个训练过程中归一化稳定一致。

---

#### `entropy_coeff` — 熵正则化系数 (Entropy Regularization)

**解决的问题**: 最大化策略熵防止策略过早坍缩到确定性输出，保持探索能力。

**DeepSWE 的发现 (设为 0)**:
熵损失可能引入不稳定性——熵指数增长最终导致训练崩溃。如果 base model 的 token 级熵在 0.3-1 范围内 (大多数开源模型满足)，**不需要额外的熵正则化**。

**推荐**:
- 一般 GRPO 训练: `entropy_coeff=0` (关闭)
- 如果 entropy 持续下降 (< 0.1)，酌情设置 `0.0001-0.001`
- **注意**: 设置 `calculate_entropy=true` 开启模型输出熵 (增加显存和计算开销)

---

#### `ppo_epochs` — PPO Epoch 数量 (PPO Epochs per Batch)

**解决的问题**: 对同一批数据做几轮梯度更新。多轮可以更充分利用采样数据，但过度训练会导致过拟合和策略退化。

**推荐**:
- 一般 GRPO: `1` (默认)
- 数据量很小时: `2-3`
- DeepSWE: `1` (每个 batch 只用一次)

---

## 3. Rollout Configuration (采样/推理性配置)

**Config path**: `actor_rollout_ref.rollout.*`
**Dataclass**: `RolloutConfig` in worker configs
**File**: [rollout.yaml](../verl/trainer/config/rollout/rollout.yaml)

### 3.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `name` | ?? (required) | `vllm`, `sglang`, `hf`, `trtllm` | 推理引擎 | 用哪个推理引擎 |
| `n` | `1` | `1-64` | 采样 | 每个 prompt 生成几个 response (GRPO group size) |
| `temperature` | `1.0` | `0.0-2.0` | 采样 | 生成多样性控制 |
| `top_p` | `1` | `0.8-1.0` | 采样 | Nucleus sampling 阈值 |
| `top_k` | `-1` | `-1` (vLLM) / `0` (HF) / `1-100` | 采样 | Top-K sampling |
| `response_length` | `${data.max_response_length}` | `1024-65536` | 长度 | 最大生成长度 |
| `prompt_length` | `${data.max_prompt_length}` | `512-32768` | 长度 | Prompt 最大长度 |
| `dtype` | `bfloat16` | `bfloat16`, `fp16` | 推理精度 | 推理时的参数精度 |
| `gpu_memory_utilization` | `0.5` | `0.4-0.9` | 显存 | vLLM/SGLang KV cache 的显存占比 |
| `tensor_model_parallel_size` | `2` | `1-8` | 并行 | 推理时的 TP 大小 |
| `free_cache_engine` | `True` | `True` / `False` | 显存 | 生成后是否释放 KV cache |
| `max_num_batched_tokens` | `8192` | `2048-65536` | 吞吐 | vLLM 单 batch 最大 token 数 |
| `max_num_seqs` | `1024` | `64-2048` | 吞吐 | vLLM 最大并发序列数 |
| `enforce_eager` | `False` | `True` / `False` | 性能 | 是否禁用 CUDA Graph |
| `enable_chunked_prefill` | `True` | `True` / `False` | 性能 | 是否启用分块预填充 |
| `enable_prefix_caching` | `True` | `True` / `False` | 性能 | 是否启用 Prefix Caching |
| `scheduling_policy` | `fcfs` | `fcfs`, `priority` | 调度 | vLLM 调度策略 |
| `load_format` | `dummy` | `dummy`, `auto`, `safetensors`, `hf` | 权重加载 | 从训练引擎同步权重的方式 |
| `ignore_eos` | `False` | `True` / `False` | 采样 | 是否忽略 EOS 继续生成 |
| `do_sample` | `True` | `True` / `False` | 采样 | 训练时是否随机采样 (HF rollout) |
| `calculate_log_probs` | `False` | `True` / `False` | 调试 | 是否计算 rollout 阶段的 log prob |
| `enable_rollout_routing_replay` | `False` | `True` / `False` | MoE | 是否从推理引擎获取路由决策 (配合 R3) |

### 3.2 关键参数详解

#### `n` — 每个 Prompt 的采样数量 (GRPO Group Size)

**解决的问题**: GRPO 需要每个 prompt 生成 N 个 response 构成 group 来计算 relative advantage。N 过小 advantage 方差大，N 过大推理成本高。

**推荐值**:
- 数学推理: `n=4` ~ `n=16` (GRPO 论文用 16)
- 代码生成: `n=8` ~ `n=16`
- **Agent 任务 (DeepSWE)**: `n=8` (平衡方差和成本)
- 简单分类/选择题: `n=4`
- 节省显存/加速: `n=4` (配合 `rloo_vectorized` 补偿方差)

**与其它参数关系**:
- `train_batch_size = N_prompts = total_rollouts / n`
- 如果 `train_batch_size=64, n=8`, 则总共 512 个并行 rollout
- `n` 越大，`adv_estimator` 的 group 越大，advantage 估计越准

---

#### `temperature` — 采样温度 (Sampling Temperature)

**解决的问题**: 控制生成多样性。高温度 → 更多探索 → 更低的 token 级确定性。

**推荐**:
- 训练阶段: `0.8-1.2` (鼓励探索)
- 训练初期: `1.0-1.2` (更多探索)
- 训练后期: 可逐步降低到 `0.6-0.8`
- 验证/评估阶段: `0.0-0.7` (更确定性)
- **Agent 任务**: `temperature=1.0` (DeepSWE 的选择)

---

#### `gpu_memory_utilization` — KV Cache 显存比例 (GPU Memory for KV Cache)

**解决的问题**: vLLM/SGLang 用多少显存做 KV cache。越高 → 能处理更大 batch / 更长序列，但留给模型权重的显存越少。

**推荐**:
- 模型小 (< 7B): `0.7-0.9`
- 模型大 (32B+): `0.4-0.6` (DeepSWE：32B 模型用较低值)
- 长序列 (32K+ tokens): 降低此值，避免 OOM

---

#### `response_length` — 最大响应长度 (Max Response Length)

**解决的问题**: 约束 LLM 最多生成多少 token。需要综合考虑任务的响应分布和显存预算。

**推荐**:
- GSM8K 等数学: `1024-2048`
- LiveCodeBench 等代码: `4096-16384`
- **SWE Agent**: `16384`+ (DeepSWE 评估用 64K total context)
- 原则: 设为训练集中 90-95 分位数，减少超长无效生成

---

## 4. Data Configuration (数据配置)

**Config path**: `data.*`
**File**: [legacy_data.yaml](../verl/trainer/config/data/legacy_data.yaml)

### 4.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `train_files` | `~/data/rlhf/gsm8k/train.parquet` | path(s) | 数据源 | 训练数据路径 |
| `val_files` | `~/data/rlhf/gsm8k/test.parquet` | path(s) | 数据源 | 验证数据路径 |
| `train_batch_size` | `1024` | `32-2048` | Batch | 每次训练的 prompt 数量 |
| `val_batch_size` | `null` | `null` or int | Batch | 验证时的 batch size |
| `max_prompt_length` | `512` | `256-65536` | 长度 | 最大 prompt 长度 |
| `max_response_length` | `512` | `256-65536` | 长度 | 最大 response 长度 |
| `prompt_key` | `prompt` | string | 数据格式 | 数据集中 prompt 字段名 |
| `reward_fn_key` | `data_source` | string | 数据格式 | 用来选择 reward 函数的字段 |
| `train_max_samples` | `-1` | `-1` or N | 数据量 | 最多用多少训练样本 |
| `val_max_samples` | `-1` | `-1` or N | 数据量 | 最多用多少验证样本 |
| `truncation` | `error` | `error`, `left`, `right`, `middle` | 预处理 | prompt 超长时的裁剪策略 |
| `filter_overlong_prompts` | `false` | `true` / `false` | 预处理 | 是否过滤超长 prompt (丢弃) |
| `filter_overlong_prompts_workers` | `1` | `1-32` | 预处理 | 过滤时的并行进程数 |
| `shuffle` | `true` | `true` / `false` | 数据加载 | 是否 shuffle 训练数据 |
| `seed` | `null` | null or int | 数据加载 | shuffle 的随机种子 |
| `dataloader_num_workers` | `8` | `0-32` | 数据加载 | DataLoader 的 worker 数 |
| `return_raw_chat` | `true` | `true` / `false` | 数据格式 | 是否返回原始 chat 格式 |
| `return_raw_input_ids` | `false` | `true` / `false` | 数据格式 | 是否返回未加 chat template 的 input_ids |
| `return_full_prompt` | `false` | `true` / `false` | 数据格式 | 是否返回加完 chat template 的完整 prompt |
| `apply_chat_template_kwargs` | `{}` | dict | 数据格式 | tokenizer.apply_chat_template 的额外参数 |
| `sampler.class_path` | `null` | path or null | 课程学习 | 自定义 sampler (curriculum) 的路径 |
| `trust_remote_code` | `false` | `true` / `false` | 安全 | 是否信任 tokenizer 的远程代码 |

### 4.2 关键参数详解

#### `train_batch_size` — 训练 Batch 大小 (Train Batch Size)

**解决的问题**: 每个 RL step 使用多少个 prompt。结合 `rollout.n` 决定每次的总 rollout 数。

**计算公式**: `total_rollouts_per_step = train_batch_size * rollout.n`

**推荐**:
- 小模型 (< 7B): `128-512`
- 中等模型 (14B-32B): `64-256`
- 大模型 (70B+): `32-128`
- DeepSWE (32B): `train_batch_size=64`, `n=8`, total=512

**调参原则**: 太小 → advantage 估计不准；太大 → 推理时间过长、显存压力大。一般保证每个 step 有 256-1024 个有效 rollout。

---

#### `max_prompt_length` + `max_response_length` — 上下文长度控制 (Context Length)

**解决的问题**: 控制总输入 token 数。`max_prompt_length + max_response_length` 决定了模型的最大上下文窗口。

**推荐**:
- GSM8K: `prompt=256, response=1024`
- 代码推理: `prompt=2048, response=16384`
- SWE Agent: `prompt=16384, response=49152` (total 64K)
- DeepSWE 评估: total 64K (训练时用更小值)

**注意**: 
- `truncation=error` 时超长数据报错而非裁剪
- 如果经常遇到超长数据，考虑 `truncation=left` (保留最后部分) 或调整模型

---

## 5. Reward Configuration (奖励配置)

**Config path**: `reward.*`
**File**: [reward.yaml](../verl/trainer/config/reward/reward.yaml)

### 5.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `num_workers` | `8` | `1-64` | 并行 | 并行计算 reward 的 worker 数 |
| `custom_reward_function.path` | `null` | path or null | 自定义 | 自定义 reward 函数文件路径 |
| `custom_reward_function.name` | `compute_score` | string | 自定义 | 自定义 reward 函数名 |
| `reward_model.enable` | `False` | `True` / `False` | RM | 是否使用生成式/Discriminative RM |
| `reward_model.model_path` | `null` | path | RM | RM 模型路径 |
| `reward_model.rollout.name` | ??? | `vllm`, `sglang`, `hf` | RM 推理 | RM 推理引擎 |
| `sandbox_fusion.url` | `null` | url or null | 代码执行 | 沙盒执行环境 URL |
| `sandbox_fusion.max_concurrent` | `64` | `1-256` | 代码执行 | 最大并发请求数 |
| `sandbox_fusion.memory_limit_mb` | `1024` | `256-8192` | 代码执行 | 每个沙盒内存上限 (MB) |
| `reward_model.overlong_buffer.enable` | `False` | `True` / `False` | DAPO | 是否启用超长惩罚 |
| `reward_model.overlong_buffer.len` | `0` | `1024-8192` | DAPO | 惩罚缓冲区的 token 数 |
| `reward_model.overlong_buffer.penalty_factor` | `0.0` | `0.0-1.0` | DAPO | 惩罚系数 |

### 5.2 关键参数详解

#### `custom_reward_function` — 自定义 Reward 函数 (Custom Reward Function)

**解决的问题**: 定义 reward 计算逻辑。verl 支持通过函数注册或文件路径两种方式指定。

**配置示例**:
```yaml
reward:
  custom_reward_function:
    path: /path/to/my_reward.py
    name: compute_score  # 默认函数名
```

**函数签名**:
```python
def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    # 返回 float (或 dict for GDPO)
    return reward_score
```

**GDPO 多维度 reward**:
```python
def compute_score(...):
    return {
        "format_reward": 1.0,
        "accuracy_reward": 0.8,
    }
```
配合 `algorithm.gdpo_reward_keys` 使用。

---

#### `overlong_buffer` — DAPO 超长惩罚 (Overlong Reward Shaping)

**解决的问题**: 不是直接丢弃超长 response (这会导致训练偏差)，而是给超长部分施加线性递减的负奖励。

**机制**: 在临近 `max_response_length` 的 `len` 个 token 范围内，施加从 0 到 `-penalty_factor` 的线性惩罚。

**示例**: `max_response_length=20480`, `len=4096`, `penalty_factor=1.0`
- Response 长度 0-16384: 无惩罚
- Response 长度 16384-20480: 惩罚从 0 线性增加到 -1.0

**推荐**: 
- 一般场景: 不启用
- 显式需要压制生成长度时: `len=4096`, `penalty_factor=0.5-1.0`
- DeepSWE: 没直接用这个机制 (用了 compact filtering)

---

## 6. Trainer Configuration (训练器配置)

**Config path**: `trainer.*`
**File**: [ppo_trainer.yaml](../verl/trainer/config/ppo_trainer.yaml)

### 6.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `total_training_steps` | `null` | `-1` (auto) or N | 训练流程 | 总训练步数 |
| `total_epochs` | `30` | N | 训练流程 | 总训练 epoch 数 |
| `save_freq` | `-1` | `-1` or N | 保存 | 每隔多少步保存 checkpoint |
| `test_freq` | `-1` | `-1` or N | 验证 | 每隔多少步验证 |
| `val_before_train` | `True` | `True` / `False` | 验证 | 训练前是否先验证一次 |
| `val_only` | `False` | `True` / `False` | 验证 | 是否只验证不训练 |
| `nnodes` | `1` | `1-N` | 分布式 | 训练节点数 |
| `n_gpus_per_node` | `8` | `1-8` | 分布式 | 每节点 GPU 数 |
| `resume_mode` | `auto` | `auto`, `disable`, `resume_path` | 续训 | 如何恢复训练 |
| `resume_from_path` | `null` | path or null | 续训 | 手动指定 checkpoint 路径 |
| `logger` | `["console", "wandb"]` | list | 日志 | 日志后端 |
| `project_name` | `verl_examples` | string | 日志 | 项目名 (wandb) |
| `experiment_name` | `gsm8k` | string | 日志 | 实验名 (wandb) |
| `log_val_generations` | `0` | `0-10` | 日志 | 验证时保存几个生成结果 |
| `critic_warmup` | `0` | `0-100` | PPO (有 Critic) | Critic 预热步数 |
| `balance_batch` | `True` | `True` / `False` | 分布式 | 是否平衡各 worker 的 batch |
| `esi_redundant_time` | `0` | seconds | 容错 | ESI 关机前的冗余时间 |
| `max_actor_ckpt_to_keep` | `null` | null or N | 磁盘 | 最多保留几个 actor checkpoint |
| `max_critic_ckpt_to_keep` | `null` | null or N | 磁盘 | 最多保留几个 critic checkpoint |
| `default_local_dir` | `checkpoints/${project}/${experiment}` | path | 磁盘 | checkpoint 默认保存路径 |
| `ray_wait_register_center_timeout` | `300` | seconds | 系统 | Ray worker 注册超时 |

### 6.2 关键参数详解

#### `total_training_steps` / `total_epochs` — 训练长度控制 (Training Duration)

**计算逻辑**: 如果 `total_training_steps=null`，则根据 `total_epochs` 和数据集大小自动计算。

**推荐**:
- 小数据集 (< 1K): `total_training_steps=200-500`
- 中数据集 (1K-10K): `total_epochs=5-30`
- 大数据集 (10K+): `total_epochs=1-3`
- **DeepSWE (4.5K): `total_training_steps=200`** (Figure 2 显示 200 步达到最佳)

---

#### `save_freq` + `test_freq` — 保存和验证频率 (Save and Validation Frequency)

**推荐**:
- `save_freq: 50` (每 50 步保存，防止训练崩溃丢失进展)
- `test_freq: 20` (每 20 步验证，及时监控性能)
- 快速实验: `save_freq=-1` (只保存最后), `test_freq=-1` (不验证)
- 谨慎训练 (如 70B 模型): `save_freq=10`, `test_freq=10`

---

## 7. Optimizer Configuration (优化器配置)

**Config path**: `actor_rollout_ref.actor.optim.*`
**YAML**: inline in [actor.yaml](../verl/trainer/config/actor/actor.yaml)

### 7.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `lr` | `1e-6` | `1e-7 - 1e-5` | 学习率 | 峰值学习率 |
| `lr_warmup_steps_ratio` | `0.0` | `0.0-0.1` | 学习率 | Warmup 步数占总步数的比例 |
| `lr_warmup_steps` | `-1` | `-1` (delegate to ratio) or N | 学习率 | Warmup 步数 (绝对值) |
| `weight_decay` | `0.01` | `0.0-0.1` | 正则化 | Weight decay 系数 |
| `lr_scheduler_type` | `constant` | `constant`, `cosine`, `linear` | 学习率 | LR schedule 类型 |
| `min_lr_ratio` | `0.0` | `0.0-0.5` | 学习率 | 最低 LR 相对于 peak LR 的比例 |
| `num_cycles` | `0.5` | `0.5-3.0` | 学习率 | Cosine schedule 的周期数 |
| `betas` | `[0.9, 0.999]` | `[0.9, 0.95]` etc. | Adam | Adam betas |
| `clip_grad` | `1.0` | `0.5-5.0` | 优化 | 梯度裁剪 max norm |
| `optimizer` | `AdamW` | `AdamW`, `Adam` | 优化器 | 优化器类型 |

### 7.2 关键参数详解

#### `lr` — 学习率 (Learning Rate)

**解决的问题**: 控制每次参数更新的步长。RL 训练通常需要比 SFT 更小的学习率。

**推荐值** (经验法则):
- **Base model cold start (DeepSWE 风格)**: `1e-6` ~ `2e-6` (小 LR, 纯 RL 探索)
- SFT 初始化后再 RL: `5e-7` ~ `1e-6` (更保守)
- 小模型 (< 7B): `1e-6` ~ `5e-6`
- 大模型 (32B+): `5e-7` ~ `1e-6`
- **如果训练不稳定 (reward 剧烈波动、loss spike)**: 降低 LR 2-5 倍

**DeepSWE 32B 的选择**: `lr=1e-6`

---

#### `lr_warmup_steps_ratio` vs `lr_warmup_steps` — Warmup 配置

**优先级**: `lr_warmup_steps > 0` 时用绝对值，否则用 `lr_warmup_steps_ratio * total_training_steps`

**推荐**: 总步数的 3-5%，如 200 步训练则 `lr_warmup_steps=10`

---

## 8. Multi-Turn / Agent Configuration (多轮对话/智能体配置)

**Config path**: `actor_rollout_ref.rollout.multi_turn.*`
**Dataclass**: `MultiTurnConfig`
**File**: [rollout.yaml](../verl/trainer/config/rollout/rollout.yaml) (lines 164-213)

### 8.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `multi_turn.enable` | `False` | `True` / `False` | 开关 | 是否启用多轮交互 |
| `multi_turn.max_assistant_turns` | `null` | null or N | 控制 | 最多 assistant 发言轮数 |
| `multi_turn.max_user_turns` | `null` | null or N | 控制 | 最多 user/tool 响应轮数 |
| `multi_turn.tool_config_path` | `null` | path | 工具 | 工具配置 JSON 文件路径 |
| `multi_turn.format` | `hermes` | `hermes`, `llama3_json` | 格式 | 多轮交互的序列化格式 |
| `multi_turn.max_parallel_calls` | `1` | `1-10` | 工具 | 单轮最大并行工具调用数 |
| `multi_turn.max_tool_response_length` | `256` | `64-4096` | 工具 | 工具返回内容最大长度 |
| `multi_turn.tool_response_truncate_side` | `middle` | `left`, `middle`, `right` | 工具 | 工具响应超长的裁剪位置 |
| `multi_turn.interaction_config_path` | `null` | path or null | 交互 | 自定义交互逻辑配置 |
| `multi_turn.use_inference_chat_template` | `False` | `True` / `False` | 格式 | 是否用推理格式的 chat template |
| `multi_turn.tokenization_sanity_check_mode` | `strict` | `disable`, `strict`, `ignore_strippable` | 校验 | 分词一致性检查 |
| `multi_turn.num_repeat_rollouts` | `null` | null or N | 采样 | 每个交互的重复 rollout 数 |
| `agent.num_workers` | `8` | `1-64` | Agent | Agent loop worker 数量 |
| `agent.default_agent_loop` | `single_turn_agent` | string | Agent | 默认 agent loop 类型 |
| `agent.agent_loop_config_path` | `null` | path or null | Agent | 自定义 agent loop 配置 |

### 8.2 关键参数详解

#### `multi_turn.enable` — 多轮交互开关 (Multi-Turn Enable)

**解决的问题**: SWE Agent 等场景需要在训练中模拟多轮工具调用，而非单轮生成。

**配置要求**: 启用时需要同时设置 `rollout.name=sglang` (目前多轮主要支持 SGLang)。

---

#### `multi_turn.max_assistant_turns` — 最大 Assistant 轮数 (Max Assistant Turns)

**解决的问题**: 限制 agent 能进行的工具调用轮数。防止 agent 无限循环或超长轨迹。

**推荐**:
- SWE Agent: `30-100` (DeepSWE: 100 max environment steps)
- 简单工具调用: `5-10`
- `null` 表示用 `max_length // 3` 自动计算

---

#### ChatML Masking (Agent Trajectory Loss Masking)

Agent 训练的关键技术点: 多轮对话中，**只对 assistant 生成的内容计算 loss**，mask 掉环境返回的 observation (user messages / tool responses)。

verl 通过 `loss_mask` 机制实现这一点: tokenize 整个对话后，标记哪些 token 属于 assistant response，哪些属于环境返回。这个逻辑在 data 预处理阶段完成。

---

## 9. Rollout Correction (Off-Policy 修正配置)

**Config path**: `algorithm.rollout_correction.*`
**Dataclass**: `RolloutCorrectionConfig` in [algorithm.py](../verl/trainer/config/algorithm.py)

**解决的问题**: RL 训练存在训练-推理不一致: rollout engine (如 vLLM BF16) 和 training engine (如 FSDP FP32) 可能有精度差异；rollout 数据来自旧的 policy checkpoint (staleness)。Rollout Correction 通过 Importance Sampling (IS) 和 Rejection Sampling (RS) 修正这个 off-policy gap。

**核心论文**: ["When Speed Kills Stability"](https://richardli.xyz/rl-collapse)

### 9.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `rollout_is` | `sequence` | `None`, `token`, `sequence` | IS | IS 的聚合级别 |
| `rollout_is_threshold` | `2.0` | `1.5-10.0` | IS | IS weight 的 clip 上界 |
| `rollout_is_batch_normalize` | `False` | `True` / `False` | IS | 是否 batch 内归一化 IS weight |
| `rollout_rs` | `None` | `token_k1`, `token_k2`, `token_k3`, `seq_sum_k1`, `seq_sum_k2`, `seq_sum_k3`, `seq_mean_k1`, `seq_mean_k2`, `seq_mean_k3`, `seq_max_k2`, `seq_max_k3` | RS | Rejection sampling 模式 |
| `rollout_rs_threshold` | `None` | float or `"lower_upper"` | RS | Rejection sampling 的阈值 |
| `bypass_mode` | `False` | `True` / `False` | 模式 | True=2策略模式, False=3策略模式 |
| `loss_type` | `ppo_clip` | `reinforce`, `ppo_clip` | 模式 | bypass 模式下的 loss 类型 |

### 9.2 关键参数详解

#### 模式选择速查 (Preset Quick Reference)

| Preset | bypass_mode | rollout_is | rollout_rs | loss_type | 适用场景 |
|---|---|---|---|---|---|
| `disabled()` | - | None | None | - | 不修正；大多数 GRPO 场景 |
| `bypass_ppo_clip()` | True | None | None | ppo_clip | 最简单：用 PPO ratio 自然处理 IS |
| `bypass_ppo_clip_k3_rs()` | True | None | seq_mean_k3 | ppo_clip | PPO clip + K3 RS 过滤 outlier |
| `bypass_pg_is()` | True | sequence | None | reinforce | REINFORCE + IS weights |
| `decoupled_seq_is()` | False | sequence | None | ppo_clip | 3策略解耦 + Seq-IS |
| `decoupled_geo_rs()` | False | None | seq_mean_k1 | ppo_clip | 3策略 + Geometric Ratio RS |
| `decoupled_k3_rs()` | False | None | seq_mean_k3 | ppo_clip | 3策略 + K3 KL RS |

**推荐**:
- **普通 GRPO 训练**: 不启用 rollout correction (设为 `None`)
- **训练崩溃、off-policy 怀疑严重**: 尝试 `bypass_ppo_clip_k3_rs(rs_threshold=0.01)`
- **3 策略架构 (rollout ≠ old policy)**: 使用 `decoupled_*` 系列

---

## 10. Router Replay Configuration (MoE 路由回放配置)

**Config path**: `actor_rollout_ref.actor.router_replay.*`
**Dataclass**: `RouterReplayConfig` in [actor.py](../verl/workers/config/actor.py)

**解决的问题**: MoE 模型的训练-推理不一致——推理引擎和训练引擎可能将 token 路由到不同 expert。Router Replay 记录并回放路由决策确保一致性。

### 10.1 参数速查表

| Parameter | Default | Range/Options | Category | 关键问题 |
|---|---|---|---|---|
| `mode` | `disabled` | `disabled`, `R2`, `R3` | 开关 | 路由回放模式 |
| `record_file` | `null` | path or null | 文件 | 录制路由决策的文件路径 |
| `replay_file` | `null` | path or null | 文件 | 回放路由决策的文件路径 |

### 10.2 R2 vs R3 对比

| 模式 | 路由决策来源 | 机制 | 适用场景 |
|---|---|---|---|
| `R2` | 训练引擎先 RECORD 再 REPLAY | 需要额外 RECORD 前向 | 独立训练，不需要 rollout 端配合 |
| `R3` | 直接从 rollout engine 获取 | 节省 RECORD 阶段 | RL 场景，需要 rollout engine 支持返回路由决策 |

**使用 R3 的条件**:
1. 模型是 MoE 架构
2. rollout engine (vLLM/SGLang) 支持返回路由决策
3. 设置 `rollout.enable_rollout_routing_replay=True`
4. 设置 `actor.router_replay.mode=R3`

---

## 11. Parameter Quick Reference by Scenario (场景速查)

### 场景 A: 数学推理 GRPO (GSM8K, MATH)

```yaml
algorithm:
  adv_estimator: grpo_vectorized
  norm_adv_by_std_in_grpo: False

actor:
  clip_ratio: 0.2
  clip_ratio_high: 0.28
  loss_agg_mode: seq-mean-token-sum-norm
  entropy_coeff: 0
  use_kl_loss: False
  ppo_epochs: 1

rollout:
  n: 8
  temperature: 1.0
  response_length: 2048

data:
  train_batch_size: 128
  max_prompt_length: 512
  max_response_length: 2048

trainer:
  total_training_steps: 200

actor.optim:
  lr: 1e-6
  lr_warmup_steps: 10
```

### 场景 B: 代码生成 GRPO (LiveCodeBench, 单轮)

```yaml
algorithm:
  adv_estimator: grpo_vectorized
  norm_adv_by_std_in_grpo: False

actor:
  clip_ratio: 0.2
  clip_ratio_high: 0.28
  loss_agg_mode: seq-mean-token-sum-norm
  use_kl_loss: False

rollout:
  n: 16
  temperature: 0.8
  response_length: 16384

data:
  train_batch_size: 64
  max_prompt_length: 2048
  max_response_length: 16384
```

### 场景 C: SWE Agent (多轮, DeepSWE 风格)

```yaml
algorithm:
  adv_estimator: rloo_vectorized
  norm_adv_by_std_in_grpo: False

actor:
  clip_ratio: 0.2
  clip_ratio_high: 0.28
  loss_agg_mode: seq-mean-token-sum-norm
  entropy_coeff: 0
  use_kl_loss: False
  ppo_epochs: 1

rollout:
  name: sglang
  n: 8
  temperature: 1.0
  response_length: 49152
  multi_turn:
    enable: True
    max_assistant_turns: 100
    format: hermes

data:
  train_batch_size: 64
  max_prompt_length: 16384
  max_response_length: 49152

trainer:
  total_training_steps: 200

actor.optim:
  lr: 1e-6
```

### 场景 D: PPO + Critic (标准 RLHF)

```yaml
algorithm:
  adv_estimator: gae
  gamma: 1.0
  lam: 0.95
  use_kl_in_reward: True
  kl_penalty: low_var_kl
  kl_ctrl:
    type: adaptive
    kl_coef: 0.02
    target_kl: 6.0

actor:
  clip_ratio: 0.2
  loss_agg_mode: token-mean
  use_kl_loss: False
  ppo_epochs: 1

critic:
  cliprange_value: 0.2

rollout:
  n: 1          # PPO 每 prompt 只需一个 response
  temperature: 1.0

data:
  train_batch_size: 128
```

### 场景 E: 训练不稳定诊断 (Troubleshooting Unstable Training)

| 症状 | 可能原因 | 调整参数 |
|---|---|---|
| Reward 震荡 | LR 过大 | 降低 `lr` 2-5 倍 |
| Loss spike / NaN | 梯度爆炸 | 降低 `clip_ratio` 到 0.1, 增加 `clip_grad` |
| Entropy 持续下降 | 策略过早坍缩 | 启用 `entropy_coeff=0.0005`, `calculate_entropy=true` |
| KL 发散 | 策略偏离太远 | 启用 `use_kl_loss`, `kl_loss_coef=0.01-0.05` |
| Response 越来越长 | 长度偏差 | 改为 `loss_agg_mode: seq-mean-token-sum-norm` |
| All reward=0 or =1 | 难度不匹配 | 调整数据集难度，启用 `filter_groups` |
| Off-policy 崩溃 | 训练-推理不一致 | 启用 `rollout_correction` |

### 场景 F: 显存不够 (OOM Troubleshooting)

| 调整 | 设置 |
|---|---|
| 降低 rollout batch | 减少 `data.train_batch_size` |
| 减少 group size | 减少 `rollout.n` |
| 减少 response length | 减少 `data.max_response_length` |
| 降低 KV cache | 降低 `rollout.gpu_memory_utilization` 到 0.4 |
| 减少 micro batch | 降低 `actor.ppo_micro_batch_size_per_gpu` |
| 启用 offload | FSDP: `actor.fsdp_config.param_offload=true` |
| 启用 dynamic bsz | `actor.use_dynamic_bsz=true` |
| 释放 KV cache | `rollout.free_cache_engine=true` |
| 降低 TP | 减少 `rollout.tensor_model_parallel_size` |
| 减少 epoch | `actor.ppo_epochs=1` |

---

## Appendix: 参数相互关系图 (Parameter Interaction Map)

```
rollout.n ──── 决定 group size ──── adv_estimator (grpo/rloo 需要 n>1)
                                      │
train_batch_size ─── 决定 prompts 数 ──── total_rollouts = train_batch_size * n
                                                    │
max_prompt_length + max_response_length ── 决定显存 ──── ppo_micro_batch_size_per_gpu
                                                                    │
clip_ratio / clip_ratio_high ── 控制更新幅度 ── loss_agg_mode ── 控制长度偏差
                        │
use_kl_loss / use_kl_in_reward ── KL 控制 ── kl_loss_coef / kl_ctrl.kl_coef
                        │
entropy_coeff ── 控制探索程度 ── temperature ── 控制采样多样性
                        │
router_replay.mode ── MoE 一致性 ── rollout.enable_rollout_routing_replay
```

---

## Reference Papers (相关论文)

| 缩写 | 论文 | 核心贡献 |
|---|---|---|
| GRPO | [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300) | Group Relative Policy Optimization |
| DAPO | [DAPO (2025)](https://arxiv.org/abs/2503.14476) | Clip-higher, Dynamic Sampling, Overlong Filtering |
| Dr.GRPO | [Dr.GRPO (2025)](https://arxiv.org/abs/2503.20783) | 去掉 std 归一化, 长度归一化 |
| RLOO | [LOOP/RLOO (2025)](https://arxiv.org/abs/2502.01826) | Leave-One-Out 优势估计 |
| DeepSWE | [DeepSWE (2025)](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33) | GRPO++ 组合 + Agent RL 训练 |
| SAPO | [SAPO (2025)](https://arxiv.org/abs/2511.20347) | 非对称平滑函数 |
| GDPO | [GDPO (2025)](https://arxiv.org/abs/2601.05242) | 多维度 reward 解耦归一化 |
| Rollout Correction | [RL Collapse (2025)](https://richardli.xyz/rl-collapse) | IS + RS 修正 off-policy gap |
| DeepScaleR | [DeepScaleR (2025)](https://agentica-project.github.io/deepscaler/) | 1.5B 纯 RL 推理模型 |
| DeepCoder | [DeepCoder (2025)](https://agentica-project.github.io/deepcoder/) | 14B 纯 RL 代码模型 |
| R2E-Gym | [R2E-Gym (2025)](https://r2e-gym.github.io/) | SWE 训练环境 + Hybrid Verifier |
