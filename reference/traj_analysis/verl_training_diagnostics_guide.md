# verl Training Diagnostics Guide (训练诊断手册)

> **定位**: 当训练指标偏离正常范围时，帮助诊断根因并给出解决方案
> **类比**: 医生看化验单 → 诊断病因 → 开处方
> **配套文档**: [verl_training_parameter_reference.md](verl_training_parameter_reference.md) (参数调参指南)

---

## 目录

1. [核心指标正常范围速查](#1-核心指标正常范围速查-vital-signs)
2. [单指标异常诊断](#2-单指标异常诊断-individual-abnormal-signs)
3. [综合征诊断 (多指标联合)](#3-综合征诊断-多指标联合-differential-diagnosis)
4. [紧急状况处理](#4-紧急状况处理-emergency-procedures)
5. [训练健康检查清单](#5-训练健康检查清单-health-checklist)

---

## 1. 核心指标正常范围速查 (Vital Signs)

> 以下范围基于 GRPO/PPO 训练经验，具体数值因模型大小、任务类型而异。

### 1.1 一级指标 (必看 — 直接反映训练健康状态)

| 指标 (Metric) | 健康范围 | 警戒线 | 危险线 | 趋势要求 |
|---|---|---|---|---|
| **`reward/score`** (或 `critic/score/mean`) | 逐步上升 | 连续 50 步不涨 | 持续下降 | **上升** |
| **`actor/pg_loss`** | 波动但不大幅漂移 | 变化 >5x 初始值 | 变为负值 (GRPO) 或 NaN | **稳定波动** |
| **`actor/ppo_kl`** | 0.001-0.05 | 0.05-0.1 | >0.1 且持续上升 | **稳定或缓慢上升后收敛** |
| **`actor/pg_clipfrac`** | 0.01-0.2 | 0.2-0.4 | >0.4 | **稳定** |
| **`actor/entropy`** | 0.3-1.0 | 0.1-0.3 | <0.1 或 >3.0 | **缓慢下降但不触底** |
| **`actor/grad_norm`** | 0.1-10 | 10-100 | >100 或 NaN | **波动但无尖峰** |
| **`response_length/mean`** | 因任务而异 | 持续增长 >2x | 触顶 (`max_response_length`) | **稳定或轻微增长** |

### 1.2 二级指标 (辅助判断)

| 指标 | 健康范围 | 异常信号 |
|---|---|---|
| **`actor/pg_clipfrac_lower`** | <0.1 | >0.2 说明 dual-clip 频繁触发 |
| **`actor/lr`** | 符合 schedule | 不变 = scheduler 未生效 |
| **`critic/score/max`** | >0 且偶尔达到满分 | 长期为 0 = 所有样本都失败 |
| **`response/aborted_ratio`** | <0.1 | >0.2 = 大量生成被中止 |
| **`response_length/clip_ratio`** | <0.1 | >0.3 = 太多达最大长度的响应 |
| **`perf/mfu/actor`** | 0.3-0.7 | <0.2 = 训练效率低 |
| **`perf/throughput`** | 稳定 | 突变 = 硬件异常或负载变化 |
| **`rollout_corr/kl`** | <0.1 | >0.5 = 严重的 train-rollout 策略差距 |
| **`rollout_corr/rollout_is_eff_sample_size`** | >0.5 | <0.2 = IS weights 退化严重 |

### 1.3 Agent 特有指标

| 指标 | 健康范围 | 异常信号 |
|---|---|---|
| **`num_turns/mean`** | 逐渐增长 | 突然跳跃或归零 |
| **`num_turns/max`** | 接近但不超过 `max_assistant_turns` | 大量触顶 = agent 学不会 stop |
| **`tool_call_counts/mean`** | 稳定 | 突变可能意味 tool 使用模式变化 |

---

## 2. 单指标异常诊断 (Individual Abnormal Signs)

### 2.1 `actor/pg_loss`

#### 正常表现
- 在 GRPO 中为正，波动不剧烈
- 随训练推进可能有轻微上升（因为 advantage 变大）
- step 之间的变化 < 5x 前一 step 的值

#### 异常模式 A: pg_loss 持续下降趋于 0

```
症状: actor/pg_loss 从 0.XX 持续下降到 0.001 级别
危险等级: ⚠️ 中
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **策略已收敛，所有 advantage 接近 0** | 查看 `critic/advantages/mean` 和 `critic/advantages/std`，std 趋近 0 | 1) 增加 `rollout.n` 提高 advantage 方差；2) 检查 reward 是否都变成一样 (0 或 1) |
| **Reward hacking: model 找到捷径获得 reward** | 查看生成的文本是否有重复/无意义模式 | 1) 加入 format reward 约束; 2) 使用 GDPO 多维度 reward |
| **所有 response 完全相同** | 查看 `response_length/std` 是否为 0 | 1) 提高 temperature; 2) 检查 `do_sample` 是否为 True; 3) 增加 entropy_coeff |
| **GRPO: 每个 group 内所有 reward 相同** | `critic/score/std` 接近 0 | 1) 启用 `filter_groups`; 2) 增加 `n` 使 group 更大; 3) 换更难的训练数据 |

#### 异常模式 B: pg_loss 剧烈震荡

```
症状: actor/pg_loss spike 到初始值的 10-100x，然后回落
危险等级: 🔴 高 (可能导致训练崩溃)
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **单个 batch 中有极端 outlier** | 查看 `critic/rewards/max` 是否有异常高值 | 1) 检查 reward 函数是否有 bug; 2) 对 reward 做 clip (如 [-1, 1]) |
| **策略突变 (`actor/grad_norm` spike)** | 同时查看 `actor/grad_norm` 是否 spike | 1) 减小 `clip_ratio`; 2) 减小 `lr`; 3) 增大 `clip_grad` |
| **KL divergence 爆炸** | `actor/ppo_kl` 同时显著升高 | 1) 启用 `use_kl_loss` + `kl_loss_coef=0.01-0.05`; 2) 减小 `clip_ratio` |
| **off-policy gap 过大** | `rollout_corr/kl` > 0.5 | 1) 启用 rollout_correction 的 IS/RS; 2) 增加 rollout 和 training 的同步频率 |

#### 异常模式 C: pg_loss 变为负值

```
症状: actor/pg_loss 变成负数 (GRPO 中不应该出现)
危险等级: 🔴 高
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **`loss_agg_mode` 或 `loss_scale_factor` 配置错误** | 检查配置是否有误 | 确认 `loss_agg_mode: seq-mean-token-sum-norm`, `loss_scale_factor` 正确 |
| **advantage 符号与 ratio 方向不一致** | `critic/advantages/mean` 和 `critic/rewards/mean` 符号相反 | 1) 检查 reward 计算; 2) 确认 `norm_adv_by_std_in_grpo` 设置是否符合预期 |
| **数值精度问题导致 log_prob 计算异常** | `actor/ppo_kl` 异常大 (>10) | 1) 确认 `dtype=bfloat16`; 2) 检查是否有 FP16 overflow |

---

### 2.2 `actor/ppo_kl` or `actor/kl_loss`

#### 正常表现
- GRPO + `use_kl_loss=False`: 值较小但存在 (仅作为 monitoring metric)
- PPO + `use_kl_in_reward=True`: 应稳定在 target_kl 附近
- 整体趋势: 温和上升后收敛

#### 异常模式 A: KL 发散 (持续快速上升)

```
症状: actor/ppo_kl 从 0.01 → 0.1 → 0.5 → 1.0+ 单调递增
危险等级: 🔴 高 — 这是训练崩溃的最常见前兆
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **学习率过大** | `actor/lr` 是否偏高 | **首要操作**: 降低 `lr` 2-5x |
| **clip_ratio 过大** | `actor/pg_clipfrac` 是否偏高 | 降低 `clip_ratio` 到 0.1，降低 `clip_ratio_high` |
| **没有 KL 约束 (GRPO cold start)** | `use_kl_loss=False` | 启用 `use_kl_loss=True`, `kl_loss_coef=0.01` |
| **某个 batch 的 advantage 极端** | `critic/advantages/max` 异常 | 对 advantage 做 clip |
| **ppo_epochs > 1 导致过度更新** | `ppo_epochs` 设置 | 减小 `ppo_epochs` 到 1 |
| **训练数据太少，过拟合** | 检查有效样本数 | 增加数据集大小或启用数据增强 |

#### 异常模式 B: KL 持续为 0 或接近 0

```
症状: actor/ppo_kl ≈ 0 持续很多步
危险等级: ⚠️ 中
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **参数未更新 (梯度消失)** | `actor/grad_norm` ≈ 0 | 1) 检查 optimizer state; 2) 增大 lr; 3) 检查 loss_agg_mode 是否正确 |
| **Ref model 就是当前 model (配置错误)** | 检查 ref model 是否正确加载 | 确保 ref model 的权重独立于 actor |
| **模型容量上限** | reward 也不涨 | 1) 换更大的 base model; 2) 增加训练数据多样性 |

---

### 2.3 `actor/pg_clipfrac`

#### 正常表现
- 通常在 0.01-0.2
- DAPO clip-higher 设置下略高也正常 (0.1-0.3)
- 长期趋势: 持续 >0.3 需要关注

#### 异常模式: Clip Fraction 过高 (>0.3)

```
症状: actor/pg_clipfrac > 0.3 持续
危险等级: ⚠️ 中 — 大量更新被 clip，训练效率低
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **clip_ratio 过小** | 配置检查 | 增大 `clip_ratio` 到 0.25-0.3 |
| **advantage 量级太大** | `critic/advantages/max` 偏大 | 1) `norm_adv_by_std_in_grpo=True`; 2) 对 advantage 做 global clip |
| **模型更新幅度过大** | `actor/grad_norm` 也高 | 1) 降低 `lr`; 2) 增大 `clip_grad` |
| **学习率过大** | 综合判断 | 降低 lr |

**clipfrac 各区间解读**:
- `0-0.1`: 更新太小，模型可能学习缓慢
- `0.1-0.25`: **最佳区间**: 有约束但不过度限制
- `0.25-0.4`: 偏大，调整 clip_ratio 或 lr
- `>0.4`: 大量更新被截断，训练效率极低

---

### 2.4 `actor/entropy`

#### 正常表现
- Base model token 级熵: 0.3-1.0
- 随训练进行缓慢下降是正常的 (策略从探索到收敛)
- 下降速率为 0.01-0.05 / 100 steps

#### 异常模式 A: 熵崩溃 (<0.1)

```
症状: actor/entropy 快速降至 0.1 以下
危险等级: 🔴 高 — 策略过早收敛，失去探索能力
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **训练数据太简单** | reward 快速达到接近满分 | 1) 换更难的数据; 2) 增加数据多样性 |
| **temperature 太低** | rollout temperature 设置 | 提高 `temperature` 到 1.2-1.5 |
| **reward 信号太强/太一致** | `critic/rewards/std` ≈ 0 | 1) 增加 reward 噪声; 2) 使用多维度 reward |
| **没有熵正则化** | `entropy_coeff=0` | 启用 `entropy_coeff=0.0005`, `calculate_entropy=true` |
| **模型太小，容量不足** | reward 不涨但熵在降 | 换更大的模型 |

#### 异常模式 B: 熵爆炸 (>3.0 或快速增长)

```
症状: actor/entropy 快速增长到 3.0+ 或指数增长
危险等级: 🔴 极高 — 一定会导致训练崩溃
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **学习率过大导致策略发散** | `actor/grad_norm` spike, `actor/ppo_kl` spike | **立即**: 大幅降低 lr (5-10x), 回滚到上一个 checkpoint |
| **entropy_coeff 设置过大** | 检查配置 | 减小或关闭 `entropy_coeff` |
| **KL 约束关闭 + lr 大** | `use_kl_loss=False`, `use_kl_in_reward=False` | 启用 KL 约束 (至少一种) |
| **reward 设计问题导致正反馈循环** | 冗长但无意义的 response 获得高 reward | 1) 加入 length penalty; 2) 用 `seq-mean-token-sum-norm` |

**DeepSWE 的经验**: 熵损失可能引入不稳定性并导致指数熵增长 → 如果 base model 熵在 0.3-1，**建议关闭 entropy_coeff**。

---

### 2.5 `actor/grad_norm`

#### 正常表现
- 通常在 0.1-10
- 随训练可能有小幅上升 (模型变 "陡")
- 单个 spike 后回落可接受

#### 异常模式 A: 梯度爆炸 (>100 或 NaN)

```
症状: actor/grad_norm 突然 spike 到 >100 或变成 NaN
危险等级: 🔴 极高 — 训练可能已经崩溃
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **单个 batch 包含极端样本** | `critic/rewards/max` 异常 | 1) 对 reward 做 clip; 2) 过滤极端数据 |
| **LR 过大** | `actor/lr` 偏高 | 降低 lr 2-5x |
| **loss_agg_mode 配置错误** | 导致 loss scale 异常 | 检查 loss_agg_mode 和 loss_scale_factor |
| **混合精度问题** | FP16 下溢出 | 改用 `dtype=bfloat16` |
| **长序列的 log_prob 计算溢出** | `response_length/mean` 很大 | 降低 `max_response_length`, 确保 attention mask 正确 |

**急救措施**:
1. **立即回滚** 到上一个 checkpoint
2. 降低 `lr` 5x 后从 checkpoint 继续
3. 增大 `clip_grad` 到 1.0-5.0
4. 检查是否有 NaN 的数据样本

#### 异常模式 B: 梯度消失 (≈0)

```
症状: actor/grad_norm 持续 <0.001
危险等级: ⚠️ 中 — 模型不学习
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **所有 advantage 为 0 (GRPO group 内 reward 全相同)** | `critic/advantages/std` ≈ 0 | 1) 增加 `n`; 2) 提高 temperature; 3) 启用 filter_groups |
| **Optimizer state 损坏** | lr 正常但 grad norm 为 0 | 1) 重启训练; 2) 检查 checkpoint 完整性 |
| **loss_response_mask 全为 0** | 所有 response 被 mask 掉 | 检查 compact filtering / overlong filtering 逻辑 |

---

### 2.6 Reward / Score 指标

#### 正常表现
- `critic/score/mean`: 逐步上升
- `critic/score/max` > 0: 至少有模型能解决问题
- `critic/rewards/mean` 与 score 强相关

#### 异常模式 A: Reward 不涨 (Plateau)

```
症状: reward 在训练初期后就停滞，连续 50+ step 不变
危险等级: ⚠️ 中
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **探索不足** | `actor/entropy` 偏低, `rollout/temperature` 偏低 | 1) 提高 temperature; 2) 减小 clip_ratio; 3) 启用 entropy_coeff |
| **训练数据太简单** | score 已经接近满分 | 1) 增加更难的数据; 2) 使用 curriculum learning |
| **训练数据太难** | score 持续接近 0 | 1) 增加更简单的 warmup 数据; 2) 降低难度 |
| **模型太小** | KL 也不变 | 使用更大模型 |
| **reward 信号太稀疏** | 大部分 score 为 0 | 1) 增加中间 reward (process reward); 2) 增加 response length 给更多 token 空间 |

#### 异常模式 B: Reward 塌陷 (Collapse)

```
症状: reward 前期涨得很好，突然快速下降
危险等级: 🔴 高
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **Reward Hacking 后崩塌** | 之前 reward 虚高 (模型找到 hack 而非真正解决) | 1) 增强 reward 函数的鲁棒性; 2) 用 `gdpo` 多维度 reward |
| **KL 发散** | `actor/ppo_kl` 同时上升 | 参见 KL 发散的处理 |
| **熵崩溃后的连锁反应** | `actor/entropy` 已经很低 | 参见熵崩溃的处理 |
| **训练数据被污染** | 检查最近的 batch 是否存在数据问题 | 检查数据质量 |

---

### 2.7 `response_length/mean`

#### 正常表现
- 因任务不同正常值差异很大
- 数学推理: 100-500 tokens
- 代码任务: 500-3000 tokens
- SWE Agent: 可达数千 tokens
- 趋势: 轻微增长或稳定，不应指数增长

#### 异常模式 A: 长度失控 (Runaway)

```
症状: response_length/mean 快速增长 (每 10 步 >10%)
危险等级: 🔴 高 — 将导致显存溢出和训练低效
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **`loss_agg_mode: token-mean` + 长 response reward 高** | 查看长 response 是否系统性获得更高 reward | **改用 `seq-mean-token-sum-norm`** |
| **模型学会 "拖延"策略 (agent)** | `num_turns/mean` 也在增加, 但 `reward` 不涨 | 1) 限制 max_turns; 2) 加入 length penalty in reward; 3) 启用 compact filtering |
| **temperature 过高 + 无 EOS 约束** | 检查 temperature 和 ignore_eos | 1) 降低 temperature; 2) `ignore_eos=False` |

#### 异常模式 B: 大量 Response 触顶

```
症状: response_length/clip_ratio > 0.3
危险等级: ⚠️ 中
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **max_response_length 太小** | 对比任务实际需要的长度 | 增大 `max_response_length` |
| **模型学不会 EOS** | `ignore_eos=True`? 检查生成的结束模式 | 1) `ignore_eos=False`; 2) 增加 EOS penalty |
| **Agent 不会调用 finish tool** | `num_turns/mean` 触顶 | 1) 在 reward 中鼓励 timely submit; 2) compact filtering |

---

### 2.8 `response/aborted_ratio`

```
正常: <0.1
警戒: 0.1-0.3
危险: >0.3
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **环境超时过多 (Agent)** | 单个 step 执行太慢 | 1) 优化环境/工具执行速度; 2) 增加超时时间 |
| **生成超时** | 推理引擎负载过高 | 1) 减少并发; 2) 降低 `max_num_seqs` |
| **max_response_length 太小** | 大量 response 被截断 | 增大 `max_response_length` |

---

### 2.9 `rollout_corr/kl` 和 Off-Policy 指标

#### 正常表现
- `rollout_corr/kl` < 0.05: 策略几乎一致，很好
- `rollout_corr/kl` 0.05-0.2: 可接受
- `rollout_corr/rollout_is_eff_sample_size` > 0.7: IS weights 有效

#### 异常: 严重的 Off-Policy Gap

```
症状: rollout_corr/kl > 0.5, rollout_is_eff_sample_size < 0.3
危险等级: 🔴 高
```

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **Rollout 策略和训练策略精度不匹配** | rollout (BF16) vs training (FP32) | 1) 对齐精度; 2) 启用 rollout_correction |
| **模型更新太快 (staleness)** | 两次 rollout 之间 update 太多步 | 1) 增加 rollout 频率; 2) 减少 `ppo_epochs` |
| **学习率过大** | 综合指标 | 降低 lr |

---

## 3. 综合征诊断 (Differential Diagnosis)

> 多个指标同时异常 → 依据症状组合定位根因

### 综合征 #1: "策略发散" (Policy Divergence)

**症状组合** (至少 3 项同时出现):
- ✅ `actor/ppo_kl` 持续上升 (>0.1)
- ✅ `actor/entropy` 上升或剧烈波动
- ✅ `actor/grad_norm` 出现 spike
- ✅ `actor/pg_loss` 震荡
- ✅ `reward` 可能正在下降

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **LR 过大** | ★★★★★ | 检查 lr 是否 > 2e-6 (32B model) |
| **没有 KL 约束** | ★★★★☆ | `use_kl_loss=False` 且 `use_kl_in_reward=False` |
| **clip_ratio 过大** | ★★★☆☆ | `actor/pg_clipfrac` > 0.3 |

**处方**:
1. **立即**: 降低 `lr` 2-5x
2. **同时**: 如果 `use_kl_loss=False`，改为 `use_kl_loss=True`, `kl_loss_coef=0.01`
3. **同时**: 降低 `clip_ratio` 和 `clip_ratio_high` 到 0.1/0.15
4. **兜底**: 回滚到上一个 checkpoint，用更保守的参数重新训练

---

### 综合征 #2: "探索枯竭" (Exploration Exhaustion)

**症状组合**:
- ✅ `actor/entropy` 持续下降 (<0.2)
- ✅ `reward` 停滞不涨
- ✅ `actor/pg_clipfrac` 偏低 (<0.05)
- ✅ `critic/score/std` 很小 (所有 sample 结果相似)
- ✅ `response_length/mean` 可能也在下降

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **温度过低** | ★★★★★ | `temperature < 0.8` |
| **所有 response 趋同 (模式坍缩)** | ★★★★☆ | `response_length/std` → 0 |
| **训练数据太简单** | ★★★☆☆ | reward 已经很高 |
| **entropy_coeff 关闭 + 没有噪声注入** | ★★★☆☆ | 检查配置 |

**处方**:
1. **首先**: 提高 `temperature` 到 1.2-1.5
2. **同时**: 如果 entropy 低于 0.1，启用 `entropy_coeff=0.0005`
3. **检查**: `clip_ratio_high` 是否 < 0.2 → 提高到 0.28 (DAPO clip-higher)
4. **数据**: 如果 reward 已经很高，考虑加入更难的数据或 curriculum
5. **采样**: 确保 `do_sample=True`, `top_p=0.95`, 不要用 greedy sampling

---

### 综合征 #3: "Reward Hacking" (奖励欺骗)

**症状组合**:
- ✅ `reward` 前期很高/快速增长
- ✅ `reward` 突然开始下降
- ✅ `actor/entropy` 可能先降后升
- ✅ `response_length/mean` 可能快速增长
- ✅ 检查生成样本：模型在生成重复/无意义内容但获得了高 reward

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **Reward 函数有漏洞** | ★★★★★ | 手动检查高分样本，发现 hack 模式 |
| **单一维度的 reward** | ★★★★☆ | 只有 outcome reward，无 process reward |
| **长 response 获得系统性的高 reward** | ★★★☆☆ | `response_length/mean` 与 reward 强相关 |

**处方**:
1. **修复 reward 函数**: 加入 format/quality 检查
2. **多维度 reward**: 用 `gdpo` + `gdpo_reward_keys` 解耦多个 reward 维度
3. **长度控制**: 用 `loss_agg_mode: seq-mean-token-sum-norm` 消除长度偏差
4. **过滤机制**: Agent 场景用 compact filtering 过滤未主动提交的 trajectory
5. **Reward clip**: 对 reward 做 `max` clip，防止单个高分样本主导

---

### 综合征 #4: "长度偏差污染" (Length Bias Contamination)

**症状组合**:
- ✅ `response_length/mean` 快速上升
- ✅ 长 response 系统性获得更高/更低的 `reward`
- ✅ `actor/pg_loss` 可能震荡
- ✅ `response_length/clip_ratio` 上升 (更多样本触顶)

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **`loss_agg_mode: token-mean`** | ★★★★★ | 长序列获得更多梯度权重 |
| **reward 没有长度归一化** | ★★★★☆ | reward 随长度线性增长 |

**处方**:
1. **关键**: 改用 `loss_agg_mode: seq-mean-token-sum-norm`
2. **reward 端**: 在 reward 计算中除以 response length 或使用长度无关的 metric
3. **采样端**: 适当提高 `temperature`, 避免短 response 的确定性格式
4. **overlong**: 可选启用 DAPO overlong_buffer 惩罚过长的 response

---

### 综合征 #5: "离线-在线策略鸿沟" (Off-Policy Gap)

**症状组合** (仅当 rollout_correction 启用时有指标):
- ✅ `rollout_corr/kl` > 0.3 (或上升趋势)
- ✅ `rollout_corr/rollout_is_eff_sample_size` < 0.3
- ✅ `actor/ppo_kl` 也在上升
- ✅ 训练出现不稳定 (pg_loss spike)
- ✅ `actor/pg_clipfrac` 升高

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **Rollout 和 Training 精度不匹配** | ★★★★★ | rollout BF16, training FP32 |
| **模型更新太快** | ★★★★☆ | 高 `lr` + `ppo_epochs > 1` |
| **Rollout engine 的模型版本太旧** | ★★★☆☆ | 检查 weights sync 频率 |

**处方**:
1. **首要**: 启用 rollout correction — `RolloutCorrectionConfig.bypass_ppo_clip_k3_rs(rs_threshold=0.01)`
2. **降低更新幅度**: 减小 `lr` 和/或 `ppo_epochs`
3. **增加同步频率**: 减少两次 weight sync 之间的 step 数
4. **精度对齐**: 如果可能，rollout engine 和 training engine 使用相同精度

---

### 综合征 #6: "Agent 无限循环" (Agent Infinite Loop)

**症状组合** (Agent 多轮场景):
- ✅ `num_turns/mean` 快速增长且触顶
- ✅ `response_length/mean` 随之增长
- ✅ `reward` 不涨甚至下降
- ✅ `response/aborted_ratio` 上升 (超时/超步数)

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **Agent 没学会调用 finish/submit 工具** | ★★★★★ | 检查 trajectory：没有 submit 动作 |
| **环境反馈让 agent 一直重试** | ★★★★☆ | 检查 tool response 是否触发重试循环 |
| **temperature 太高导致无意义探索** | ★★★☆☆ | agent 随机探索，永远不收敛 |

**处方**:
1. **Compact filtering**: 过滤掉达到 max steps/timeout 的 trajectory (DeepSWE 方法)
2. **Reward 设计**: 对提前 submit 给予微小正向奖励，鼓励及时结束
3. **限制回合数**: 减小 `max_assistant_turns` 到合理值
4. **降低 temperature**: Agent 需要更多确定性而非无限探索
5. **增加 submit 行为的训练样本**: 在数据中增加正确 submit 的例子

---

### 综合征 #7: "训练完全不学" (No Learning At All)

**症状组合**:
- ✅ `reward` 从 step 0 就不涨
- ✅ `actor/ppo_kl` ≈ 0 持续
- ✅ `actor/grad_norm` ≈ 0 或极小
- ✅ `actor/pg_loss` 几乎不变

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **配置错误: 参数不更新** | ★★★★★ | optimizer 没连接, frozen weights |
| **GRPO: group 内所有 reward 相同** | ★★★★☆ | `critic/score/std` ≈ 0 |
| **所有样本都失败 (reward=0)** | ★★★★☆ | `critic/score/max` = 0 |
| **训练数据格式错误** | ★★★☆☆ | 检查 tokenization, prompt 是否正确 |

**处方**:
1. **Debug 配置**: 检查 `actor.strategy`, optimizer config, 确保 `lr > 0`
2. **Check reward**: 手动运行 reward 函数，确认能返回非零值
3. **增加 exploration**: 提高 temperature, 增大 `n`
4. **简化数据**: 先用极简数据 (如单一样本) 验证训练循环是否 work
5. **检查数据加载**: 确认 prompt 被正确 tokenize 且 chat template 正确

---

### 综合征 #8: "Agent 过早收敛于捷径" (Premature Convergence to Shortcuts)

**症状组合**:
- ✅ `reward` 快速上升后停滞
- ✅ `actor/entropy` 快速下降
- ✅ `response_length/mean` 很短或为固定模式
- ✅ `num_turns/mean` 很少 (Agent)
- ✅ 检查样本：模型学到捷径 (如直接返回预设答案)

| 根因 | 置信度 | 排除线索 |
|---|---|---|
| **Reward 信号过于简单/可预测** | ★★★★★ | 检查 reward 是否太容易获得 |
| **训练数据缺乏多样性** | ★★★★☆ | 所有样本有相似 pattern |
| **temperature 太低** | ★★★☆☆ | 模型一直走最可能的路径 |

**处方**:
1. **增加数据多样性**: 混合不同类型的训练样本
2. **强化 reward**: 加入 format/quality/process reward
3. **增加 exploration**: 提高 temperature, 启用少量 entropy_coeff
4. **更难的 curriculum**: 在模型收敛到捷径前切换到更难的数据

---

## 4. 紧急状况处理 (Emergency Procedures)

### 4.1 Loss = NaN

**症状**: `actor/pg_loss` 变成 NaN, 通常伴随 `actor/grad_norm` = NaN

**立即操作**:
1. ❌ **不要继续训练** — NaN 会传播并损坏 optimizer state
2. ✅ 回滚到上一个有效 checkpoint
3. ✅ 降低 `lr` 5-10x 后重新开始
4. ✅ 检查并修复可能原因 (见下方)

**排查清单**:
- [ ] `max_response_length` + `max_prompt_length` < model max context?
- [ ] 是否有 attention_mask 全 0 的序列?
- [ ] `response_length/mean` 是否接近 `max_response_length`?
- [ ] `actor/ppo_kl` 在 NaN 之前是否已经很高 (>1.0)?
- [ ] 数据类型是否有 FP16 overflow 风险? (用 BF16!)
- [ ] `loss_agg_mode` 的分母会不会是 0?
- [ ] `norm_adv_by_std_in_grpo=True` 且 group std=0?

**修复后重试配置**:
```yaml
actor:
  clip_ratio: 0.1
  clip_ratio_high: 0.15
  clip_grad: 1.0
  loss_agg_mode: seq-mean-token-sum-norm

algorithm:
  norm_adv_by_std_in_grpo: False  # 避免除以 0
```

---

### 4.2 GPU Out of Memory (OOM)

**症状**: CUDA OOM error, 训练中断

**立即操作**:
1. 检查是哪个阶段 OOM (rollout? training? ref log prob?)
2. 针对性减小对应阶段的 batch/memory

**按阶段排查**:

| OOM 阶段 | 核心调整参数 | 次要调整 |
|---|---|---|
| **Rollout (推理)** | `rollout.gpu_memory_utilization` ↓, `rollout.max_num_batched_tokens` ↓, `rollout.max_num_seqs` ↓ | `rollout.tensor_model_parallel_size` ↑, `rollout.free_cache_engine=true` |
| **Actor Training** | `actor.ppo_micro_batch_size_per_gpu` ↓, `actor.ppo_max_token_len_per_gpu` ↓ | `actor.use_dynamic_bsz=true`, `actor.fsdp_config.param_offload=true`, `actor.use_remove_padding=true` |
| **Ref Log Prob** | `ref.log_prob_micro_batch_size_per_gpu` ↓, `ref.log_prob_max_token_len_per_gpu` ↓ | 同 actor |
| **Reward Model** | `reward_model.rollout.gpu_memory_utilization` ↓ | `reward_model.rollout.tensor_model_parallel_size` ↑ |

**通用显存优化**:
```yaml
actor:
  ppo_micro_batch_size_per_gpu: 1       # 最小 micro batch
  use_dynamic_bsz: true                  # 启用动态 bsz
  ppo_max_token_len_per_gpu: 8192        # 降低 token 上限
  fsdp_config:
    param_offload: true                  # 参数 offload 到 CPU
    optimizer_offload: true              # 优化器状态 offload
  use_torch_compile: false               # 关闭 compile 省显存
  calculate_entropy: false               # 不计算熵
  use_kl_loss: false                     # 不需要 ref log prob

rollout:
  gpu_memory_utilization: 0.35
  free_cache_engine: true
  enforce_eager: true                    # 关闭 CUDA graph 省显存
  tensor_model_parallel_size: 4

data:
  max_response_length: 8192              # 减少 response 长度
  train_batch_size: 32                   # 减少 batch
```

---

### 4.3 训练速度异常慢

**症状**: `perf/throughput` 很低, `timing_s/gen` 过长

| 可能原因 | 检测方法 | 解决方案 |
|---|---|---|
| **Rollout 瓶颈** | `timing_s/gen` 占大头 | 1) 增大 `max_num_batched_tokens`; 2) `enable_chunked_prefill=True`; 3) `enable_prefix_caching=True` |
| **Weight sync 慢** | `timing_s/update_weights` 大 | 1) 用 `nccl` backend; 2) 减少 weight sync 频率 |
| **数据加载慢** | `timing_s/` 之间有大的间隔 | 增加 `dataloader_num_workers` |
| **Checkpoint 保存慢** | `timing_s/save_checkpoint` 大 | `async_save=True` |
| **MFU 低** | `perf/mfu/actor` < 0.2 | 1) 增加 micro_batch_size; 2) `use_fused_kernels=true`; 3) 减少 TP/DP |

---

## 5. 训练健康检查清单 (Health Checklist)

### 训练启动前
- [ ] `max_prompt_length + max_response_length` ≤ model 支持的最大 context
- [ ] `train_batch_size` ≥ `ppo_mini_batch_size` (否则 validation 报错)
- [ ] `rollout.n` > 1 且 ≥ 4 (GRPO)
- [ ] `dtype=bfloat16` (非 FP16, 避免 overflow)
- [ ] reward 函数经手动测试，能返回合理的 reward 值
- [ ] 检查数据加载路径是否正确

### 训练前 10 步
- [ ] `actor/pg_loss` 在正常范围，非 NaN
- [ ] `actor/grad_norm` > 0.001 (参数在更新)
- [ ] `critic/score/max` > 0 (至少有一个样本成功)
- [ ] `response_length/mean` 在预期范围
- [ ] 没有一个指标出现指数增长趋势

### 训练中 (每 50 步检查)
- [ ] `reward` 整体趋势向上
- [ ] `actor/entropy` > 0.1 (没有崩溃)
- [ ] `actor/ppo_kl` < 0.1 或在可控范围
- [ ] `actor/pg_clipfrac` < 0.3
- [ ] `response_length/mean` 没有指数增长
- [ ] `actor/grad_norm` 没有持续 spike
- [ ] 验证集指标与训练集趋势一致

### Agent 训练额外检查
- [ ] `num_turns/mean` 在合理范围，没有普遍触顶
- [ ] `response/aborted_ratio` < 0.2
- [ ] `tool_call_counts/mean` 稳定
- [ ] 随机抽查 trajectory，确认 agent 在正常交互

---

## Appendix A: 指标依赖关系速查 (Metric Dependency Map)

```
reward/score ──→ advantages ──→ pg_loss ──→ grad_norm
     │                │              │
     │           pg_clipfrac ←── clip_ratio
     │                │
     └──→ actor/entropy (通过策略梯度)
              │
         ppo_kl ←── old_log_prob vs log_prob
              │
         kl_loss (if use_kl_loss=True)
              │
         ref_log_prob (requires ref model forward)
```

## Appendix B: 紧急联系人参数

当以下 3 个指标同时出现异常时，**几乎一定**有训练问题需要立即介入:

1. `actor/ppo_kl` > 0.5 **且**
2. `actor/entropy` < 0.1 或 > 3.0 **且**
3. `actor/grad_norm` > 50 或 NaN

→ **立即回滚 checkpoint，按综合征 #1 处理**

---

## Appendix C: 已知的 healthy training 示例 (参考)

### DeepSWE 风格 (Qwen3-32B, GRPO++, Agent):
```
actor/pg_loss:        0.05-0.15, 温和波动
actor/ppo_kl:         0.005-0.03, 缓慢上升
actor/pg_clipfrac:    0.05-0.25
actor/entropy:        0.4-0.8, 缓慢下降
actor/grad_norm:      0.5-5.0
critic/score/mean:    从 0.05 → 0.15 (200步)
response_length/mean: 2000-5000, 稳定
num_turns/mean:       5-15, 逐步增长
```

### 标准数学 GRPO (Qwen2.5-7B, GSM8K):
```
actor/pg_loss:        0.01-0.05
actor/ppo_kl:         0.001-0.01
actor/pg_clipfrac:    0.05-0.2
actor/entropy:        0.3-0.6
actor/grad_norm:      0.1-2.0
critic/score/mean:    从 0.0 → 0.7-0.9
response_length/mean: 100-300
```
