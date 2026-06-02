---
id: verl-source-root-cause-analysis
target_section: domain-knowledge
action: append
description: "VERL源码级根因分析框架: 每个训练指标的源码实现 → 异常模式 → 根因链 → 修复方案. 涵盖GRPO advantage计算、KL/clipfrac/entropy源码、3大VERL特有陷阱"
status: proposed
source: verl-source-analysis
source_sessions: ["verl_latest/verl/trainer/ppo/core_algos.py", "verl_latest/verl/trainer/ppo/metric_utils.py", "verl_latest/verl/workers/utils/losses.py", "verl_latest/verl/trainer/ppo/ray_trainer.py"]
---

## VERL 源码级根因分析 (Source-Level Root Cause Analysis)

> 以下分析基于 verl_latest 源码。每个指标的"异常→根因→修复"链路均标注源码位置和行号。

### 指标源码映射表

| 指标 | 源码文件 | 关键行 | 计算方式 |
|------|---------|--------|---------|
| `actor/ppo_kl` | `core_algos.py` | L1333 | `masked_mean(old_log_prob - log_prob)` |
| `actor/pg_clipfrac` | `core_algos.py` | L1346 | `mean(gt(pg_losses2, pg_losses1))` — clipped loss > unclipped loss的比例 |
| `actor/pg_loss` | `core_algos.py:losses.py` | L1360, L119 | PPO loss via `agg_loss()` + entropy loss + KL loss |
| `actor/entropy` | `ray_trainer.py` | L1546-1553 | `agg_loss(entropys, response_mask)` — 注意: 受 `loss_agg_mode` 影响! |
| `actor/grad_norm` | FSDP/Megatron引擎 | — | `torch.nn.utils.clip_grad_norm_` 后记录 |
| `critic/score/mean` | `metric_utils.py` | L132-140 | `mean(token_level_scores.sum(-1))` over non-aborted |
| `critic/advantages/mean` | `metric_utils.py` | L150-158 | `mean(valid_adv)` after masking |
| `response/aborted_ratio` | `metric_utils.py` | L171 | `mean(response_length==0)` |
| `response_length/clip_ratio` | `metric_utils.py` | L236 | `mean(response_length==max_response_length)` |
| `val-aux/*/reward/mean@N` | `metric_utils.py` | L554-702 | Bootstrap-based estimation, N是采样数(非step数) |

---

### 根因 #1: PPO_KL 始终为 0 — VERL源码级诊断

**源码位置**: `core_algos.py:1333`, `ray_trainer.py:1231-1253`

**计算链路**:
```
1. 生成后: _compute_old_log_prob(batch) → old_log_probs (ray_trainer.py:1542)
2. 更新后: _update_actor → compute_log_prob → log_probs (workers/utils/losses.py:59)
3. KL = masked_mean(old_log_prob - log_prob) / response_mask (core_algos.py:1333)
```

**3种VERL特定根因** (按概率排序):

| 根因 | 源码证据 | 诊断方法 | 修复 |
|------|---------|---------|------|
| **A. ref_in_actor=True** (参考模型=actor) | `ray_trainer.py:1238`: `if self.ref_in_actor: output = self.actor_rollout_wg.compute_log_prob(batch_td)` | 检查run_verl.sh中是否设置了ref_path | 设置独立的ref模型路径或启用use_kl_loss |
| **B. lr过小 → 权重几乎不变** | 经验: lr<1e-7 for 7B或lr<1e-6 for 0.5B时old_log≈log | 观察grad_norm是否>0.1 (梯度是否在流动) | 增大lr到有效范围 |
| **C. KL Loss未启用** | `losses.py:132-142`: KL loss仅在`config.use_kl_loss=True`时加到总loss中 | 检查`use_kl_loss`和`kl_loss_coef`配置 | 启用use_kl_loss=true, kl_loss_coef=0.01 |

**⚠️ VERL陷阱**: `ppo_kl`指标和`kl_loss`是两回事!
- `ppo_kl`: 旧策略与新策略的KL散度(用于监控, 不参与训练)
- `kl_loss`: KL penalty加到loss中(参与训练, 需要ref_log_prob)
- **KL=0不意味着训练无效** — Round 1实证: KL=0但val-acc=0.488, 模型确实在学习

---

### 根因 #2: GRPO Advantage 为零方差组 — VERL源码级诊断

**源码位置**: `core_algos.py:267-331`

**关键代码**:
```python
# L314-323: 单样本组特殊处理
if len(id2score[idx]) == 1:
    id2mean[idx] = torch.tensor(0.0)  # mean=0
    id2std[idx] = torch.tensor(1.0)    # std=1
# L324-326: Advantage计算
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
# 单样本组: advantage = (score - 0) / (1 + eps) = score (NOT zero!)
```

**根因链**:
1. **组大小=1** → mean=0, std=1 → advantage = score (非零, 但不含组内归一化)
2. **组内全部同分** → mean=score, std=0 → advantage = 0/epsilon ≈ 0
3. **norm_adv_by_std_in_grpo=False** → advantage = score - mean (无std归一化, 但mean仍为0)

**诊断方法**: 
- 检查`critic/score/max == critic/score/min` → 零方差组
- 检查`rollout.n`配置 (必须≥4, 推荐8以降低同分概率)
- 零方差组的`critic/advantages/mean`接近0

**修复**: 
- 增大`num_generations` (rollout.n≥8)
- 启用`filter_groups`过滤零方差组
- 使用GDPO替代GRPO (多维度reward降低同分概率)

---

### 根因 #3: Clipfrac=0 — 策略更新幅度过小

**源码位置**: `core_algos.py:1329-1346`

**计算链路**:
```python
ratio = exp(log_prob - old_log_prob)  # L1332
pg_losses1 = -advantages * ratio      # L1335: unclipped
pg_losses2 = -advantages * clamp(ratio, 1-clip, 1+clip)  # L1340-1342: clipped
pg_clipfrac = mean(gt(pg_losses2, pg_losses1))  # L1346: how often clip is triggered
```

**根因链**: clipfrac=0 意味着 `ratio` 始终在 `[1-clip_ratio, 1+clip_ratio]` 范围内:
- 默认 `clip_ratio=0.2` → ratio在[0.8, 1.2]内
- ratio=exp(Δlog_prob), Δlog_prob≈0意味着ratio≈1
- **Δlog_prob≈0的原因**: lr太小, 梯度太弱, 或模型太大参数变化不明显

**诊断决策树**:
```
clipfrac=0
├── grad_norm > 1.0? 
│   ├── YES: 梯度在流动但ratio仍在clip内 → lr偏小但可能OK (Round 1实证)
│   └── NO:  梯度不流动 → 检查optimizer/lr/frozen weights
├── KL > 0.001?
│   ├── YES: 模型在更新但幅度极小 → 考虑增大lr
│   └── NO:  模型未更新 → 检查根因#1
└── score上升?
    ├── YES: 训练有效(如Round 3: clipfrac=0.005但val恢复)
    └── NO:  训练无效 → 综合诊断
```

**VERL陷阱**: **clipfrac=0不一定是坏事**。GRPO中如果advantage信号清晰且ratio在clip范围内, 不需要clipping。Round 1 (clipfrac=0, val=0.488)和Round 3 (clipfrac=0.005, val=0.475)都证明clipfrac=0时可以正常训练。

---

### 根因 #4: Entropy 受 loss_agg_mode 污染

**源码位置**: `ray_trainer.py:1546-1553`, `core_algos.py:1138-1199`

**关键代码**:
```python
# ray_trainer.py:1546-1553
entropy_agg = agg_loss(
    loss_mat=entropys,       # per-token entropy values
    loss_mask=response_masks,
    loss_agg_mode=actor_config.loss_agg_mode,  # ← 影响entropy值!
)
metrics["actor/entropy"] = entropy_agg.detach().item()
```

**VERL陷阱**: `actor/entropy` 不是一个简单的per-token mean! 它的计算受 `loss_agg_mode` 影响:

| loss_agg_mode | 对entropy的影响 | 可能导致误读 |
|---------------|----------------|------------|
| `token-mean` | 长序列权重大 | 长时间生成的数据点主导entropy |
| `seq-mean-token-sum` | 所有序列等权 | entropy值比token-mean略低 |
| `seq-mean-token-sum-norm` | 除以response_length | entropy值可能很低! |

**诊断**: 
- 如果entropy突然变化但response_length分布不变 → 真实entropy变化
- 如果entropy和response_length同时变化 → 可能是loss_agg_mode污染
- 对比`per-token entropy mean` (可用deep_analysis中的raw entropy)

---

### 根因 #5: Val-acc与train score的背离 — 完整根因链

**根源分析**(综合VERL训练流程):

```
train_dataloader → rollout → compute_reward → compute_advantage → update_actor
     ↓                ↓            ↓                ↓                   ↓
  [训练数据]    [生成响应]    [0/1 binary]    [GRPO组内归一化]    [PPO clip+KL]
                                                                    ↓
val_dataloader → rollout → compute_reward → (不更新)
     ↓                ↓            ↓
  [验证数据]    [生成响应]    [val-acc]
```

**3种背离模式**(VERL源码视角):

| 模式 | train score | val-acc | VERL根因 | 诊断 |
|------|------------|---------|---------|------|
| **A. 过拟合** | →震荡 | ↓持续 | train problems数量过多(>128 for 0.5B), 每个问题仅见1次 | seed的prompt分布与val不重叠 |
| **B. Entropy崩塌** | →或↓ | ↓↓ | entropy_coeff=0, temperature过低 | 策略坍缩为固定模式, 该模式在val上无效 |
| **C. Reward hacking** | ↑高 | ↓ | temperature过低 + binary reward | 模型找到格式捷径(e.g., 总是输出相同结构) |

**A/B/C的区分方法**:
- 如果entropy <0.1 → 模式B
- 如果score >0.8但val低 → 模式C
- 如果entropy正常但val降 → 模式A

---

### VERL GRPO 训练流程全景 (含源码行号)

```
每个 training step = 1次完整的数据流:

1. [DataLoader] train_dataloader → batch (ray_trainer.py:1422)
2. [Rollout]   batch.repeat(rollout.n) → generate_sequences → gen_output (L1447-1469)
3. [Reward]    extract_reward(batch) → token_level_scores (L1517-1524)
               reward_tensor = compute_reward_score(response)  ← 你的reward函数
4. [Old LogP]  _compute_old_log_prob(batch) → old_log_probs, entropy (L1541-1556)
               → 此时entropy被记录为 actor/entropy (L1553)
5. [Ref LogP]  _compute_ref_log_prob(batch) → ref_log_prob (仅use_reference_policy时) (L1575-1579)
6. [Advantage] compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
               → advantages = (score - group_mean) / group_std (core_algos.py:267-331)
               → 此时记录 critic/score/mean, critic/advantages/mean (metric_utils.py:89-268)
7. [Update]    _update_actor(batch) → ppo_loss(actor) → clip_grad → optimizer.step (L1292-)
               → pg_loss, pg_clipfrac, ppo_kl 在 core_algos.py:1278-1369 中计算
               → 如果 use_kl_loss=True: kl_loss加到总loss中 (losses.py:132-142)
               → 如果 entropy_coeff>0: entropy_loss加到总loss中 (losses.py:123-129)
8. [Validate]  _validate() → val-aux/*/reward/mean@N (每test_freq步) (ray_trainer.py:1392-1396)
               → Bootstrap @N: N是每prompt的采样数, 非step数!
               → reward/mean@1 = greedy decode的准确率 (单次采样)
9. [Log]       metrics → verl_metrics.jsonl (每步一行JSON)
```

### VERL 训练流程中的3个关键边界

**边界1: rollout.n vs num_generations**

```python
# ray_trainer.py:1447
gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
# rollout_n = config.actor_rollout_ref.rollout.n
# rllm中对应 num_generations
```

- `rollout.n=1`: 每个prompt仅1个response → GRPO无法计算组内advantage!
- **每组至少2个response才能有非零advantage**
- `rollout.n=8`(R1/R2/R3配置): 每组8个response, 组内归一化有效

**边界2: loss_agg_mode 的影响面**

`loss_agg_mode` 影响3个地方:
1. `pg_loss` 聚合 (core_algos.py:1360)
2. `entropy_loss` 聚合 (losses.py:124-128)
3. `actor/entropy` 指标 (ray_trainer.py:1546-1553)

**经验**: 使用`seq-mean-token-sum-norm`可以避免长度偏差, 但entropy值会偏低。

**边界3: ref_in_actor 的连锁效应**

```python
# ray_trainer.py:1238-1244
if self.ref_in_actor:
    output = self.actor_rollout_wg.compute_log_prob(batch_td)  # ← 用actor计算ref!
else:
    output = self.ref_policy_wg.compute_ref_log_prob(batch_td)  # ← 用独立ref模型
```

- `ref_in_actor=True`: actor计算ref_log_prob → KL从计算上就接近0
- `ref_in_actor=False`: 独立ref模型 → KL能反映真实策略偏移
- 当`ref_in_actor=True`且`use_kl_loss=False`时: **完全没有KL约束!** 只有clip_ratio作为约束。
- 这就是R1/R2 KL=0的直接原因! 但clip_ratio本身可以作为足够的约束(如R1实证)。

---

### 从指标异常到VERL根因 → 修复的决策链

```
观察: KL=0 + clipfrac=0 + entropy崩塌
  ↓ 查 ray_trainer.py:1238
根因: ref_in_actor=True + use_kl_loss=False → 无KL约束
  ↓ 查 core_algos.py:267-331
加剧: rollout.n=8 → 组内归一化 → advantage有时为0(同分组)
  ↓ 查 losses.py:123-129
缺失: entropy_coeff=0 → 无探索压力 → entropy自由崩塌
  ↓
修复:
  1. 启用 entropy_coeff=0.001 (最重要 — R3实证)
  2. 启用 use_kl_loss=true + kl_loss_coef=0.01
  3. 或保持 ref_in_actor=True 但加大 clip_ratio
```

```
观察: val-acc持续下降但train score震荡
  ↓ 查 metric_utils.py:554-702
理解: val-acc是bootstrap @N估计, N=rollout.n时的mean accuracy
  ↓ 查 core_algos.py:267-331
根因: 训练数据过多(>128), 每step面对不同problem → gradient noise来自difficulty variance
  ↓ 查 ray_trainer.py:1392-1396
确认: val在每次test_freq触发时都降 → 模型在训练集上过拟合, 验证集上偏斜
  ↓
修复:
  1. 减少num_problems (<128 for 0.5B) — R3实证: 48→val稳定
  2. 增加epochs (≥2) — 确保每个样本多次接触
  3. 添加验证背离检测 — 连续3降→ALERT
```
