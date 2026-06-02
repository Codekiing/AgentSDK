# VERL 训练 Reward 提升慢 / 精度不涨排查指导原则

本文面向无人工干预的自动诊断流程，整理 VERL 训练过程中 Reward 提升慢、精度不涨、pass@k 偏低时的排查方法、关键观察指标、参考正常范围、异常原因和参数优化入口。

核心思路不是一上来调学习率，而是按层定位：

```text
指标与参数层
  -> 轨迹与工程异常层
  -> Reward 设计层
  -> 数据难度与 curriculum 层
```

参考源码路径：

```text
/lixiang/project/huxing/Project_test/AgentSDKMetricSkill/verl_latest
```

---

## 1. 总体判断流程

当 RL 训练 Reward 提升慢时，先按下面顺序判断。

### Step 1：Reward / Score 是否真的没涨

优先看：

```text
critic/score/mean
critic/score/max
critic/score/min
critic/rewards/mean
critic/rewards/max
critic/rewards/min
val-core/*/reward/mean@N
val-core/*/acc/mean@N
```

源码来源：

```text
verl_latest/verl/trainer/ppo/metric_utils.py:89
verl_latest/verl/trainer/ppo/metric_utils.py:216
verl_latest/verl/trainer/ppo/metric_utils.py:554
```

判断：

| 现象 | 说明 |
|---|---|
| train reward 不涨，val acc 也不涨 | 优先怀疑 reward、轨迹、数据难度 |
| train reward 涨，val acc 不涨 | 可能 reward hacking、训练/验证分布不一致 |
| score/max 涨但 score/mean 不涨 | 少数样本变好，但整体不稳定 |
| score/reward 的 min、max、mean 都几乎不变 | 学习信号可能非常弱 |

注意：

```text
critic/score/* 通常是原始 score。
critic/rewards/* 通常是扣除 KL penalty 后的训练 reward。
```

所以 `score` 涨但 `reward` 不涨时，不要误判为 reward function 没效果，可能是 KL 把收益抵消了。

### 通用自动检测阈值建议

以下阈值用于自动告警和初步归因，不应作为硬性失败条件。不同模型规模、任务难度、reward scale 和采样配置会改变正常区间，自动诊断系统应结合连续多个 step 的趋势判断。

| 指标 | 常见健康范围 / 趋势 | 异常信号 | 常见原因 | 自动动作 |
|---|---|---|---|---|
| `critic/score/mean` | 有缓慢上升趋势，或在 curriculum 阶段阶梯式上升 | 长期持平且 `score/max` 也不涨 | reward 无区分度、任务过难、轨迹失败 | 进入 reward / 轨迹 / 数据难度检测 |
| `critic/score/max` | 早于 mean 上升，表示存在少量成功样本 | 长期为 0 或接近初始值 | 完全没有成功轨迹、答案解析失败 | 抽取高分/低分样本做自动一致性检测 |
| `critic/rewards/mean` | 与 `score/mean` 同向，允许因 KL 略低 | `score/mean` 升但 reward 不升 | KL penalty 过强、reward 被扣抵 | 进入 KL 压制检测 |
| `critic/rewards/min/max` | min/max 有可见差距 | min/max/mean 几乎相等 | reward 全 0、全 1 或 reward scale 过小 | 进入组内方差和 reward 设计检测 |
| `critic/advantages/max/min` | max > 0 且 min < 0，存在组内对比 | max/min 都接近 0 | GRPO 组内无差异、reward 方差过低 | 检查 rollout.n、reward std、数据难度 |
| `actor/ppo_kl` | 小幅非零并平稳波动 | 长期接近 0 | policy 未更新、lr 太低、KL 约束过强 | 建议增大 lr / PPO epoch 或降低 KL |
| `actor/ppo_kl` | 无持续尖峰 | 突然尖峰或连续升高 | 更新过猛、lr 太高、batch 太小 | 建议降低 lr、增大 batch、增强 KL |
| `actor/pg_clipfrac` | 约 0.05 - 0.30 常见较健康 | > 0.5 | 大量 update 被 clip、策略更新过猛 | 降低 lr、增大 batch、调整 clip range |
| `actor/pg_clipfrac` | 约 0.05 - 0.30 常见较健康 | < 0.01 且 KL/grad 都低 | 更新太弱或 advantage 太小 | 增大 lr / epoch，检查 reward 信号 |
| `actor/grad_norm` | 非零、无持续爆炸 | 接近 0 | reward / advantage 信号弱、参数未有效更新 | 进入 reward 方差和 KL 检测 |
| `actor/grad_norm` | 无连续尖峰 | 连续尖峰或爆炸 | lr 过高、reward scale 异常、batch 太小 | 降 lr、检查 reward scale、增大 batch |
| `actor/entropy` | 缓慢下降或稳定 | 快速坍塌 | 探索不足、temperature 低、entropy_coeff 低 | 提高采样温度或 entropy 激励 |
| `prompt_length/clip_ratio` | 接近 0 | > 1% 需告警，持续 > 5% 高风险 | prompt 被截断、样本过长、模板过长 | 增大 prompt length 或过滤超长样本 |
| `response_length/clip_ratio` | 接近 0 | > 1% 需告警，持续 > 5% 高风险 | response 被截断、max_response_length 太小 | 增大 response length 或约束输出 |
| `response/aborted_ratio` | 接近 0 | > 1% 需告警，持续 > 5% 高风险 | rollout 失败、EOS/template/engine 异常 | 进入 rollout engine 与模板检测 |
| `timing_s/reward / timing_s/step` | reward 耗时占比稳定且不过半 | reward 占比长期 > 30% | reward function 慢、外部工具慢、串行计算 | 启用并行、缓存或简化 reward |
| `perf/throughput` | 随 batch/token 配置稳定 | 持续下降 | 生成瓶颈、显存压力、数据/日志阻塞 | 定位 timing_s 最大项 |
| `val-core/*/acc/mean@N` | 与 train score 长期同向 | train 涨但 val 不涨 | reward hacking、训练/验证分布不一致 | 进入 validation 样本与 reward 一致性检测 |

---

## 2. 先区分 score 和 reward：是否被 KL 压制

重点看：

```text
critic/score/mean
critic/rewards/mean
actor/reward_kl_penalty
actor/reward_kl_penalty_coeff
actor/kl_loss
actor/kl_coef
actor/ppo_kl
```

源码来源：

```text
verl_latest/verl/trainer/ppo/ray_trainer.py:76
verl_latest/verl/trainer/ppo/ray_trainer.py:113
verl_latest/verl/trainer/ppo/core_algos.py:153
verl_latest/verl/trainer/ppo/core_algos.py:1122
verl_latest/verl/trainer/ppo/core_algos.py:2126
```

判断标准：

| 现象 | 可能原因 | 建议 |
|---|---|---|
| `score/mean` 上升，但 `rewards/mean` 不升 | KL penalty 太强 | 降低 KL 系数 |
| `actor/ppo_kl` 长期接近 0 | policy 几乎没动 | 增大学习率、增加 PPO epoch、降低 KL 约束 |
| `actor/ppo_kl` 突然升高 | 更新过猛 | 降低 lr、增大 batch、增强 KL |
| `actor/reward_kl_penalty` 较大 | reward 被 KL 扣太多 | 检查 `use_kl_in_reward` 和 `kl_coef` |

相关配置：

```yaml
algorithm.use_kl_in_reward
algorithm.kl_ctrl.type
algorithm.kl_ctrl.kl_coef
algorithm.kl_ctrl.target_kl
algorithm.kl_ctrl.horizon

actor_rollout_ref.actor.use_kl_loss
actor_rollout_ref.actor.kl_loss_coef
actor_rollout_ref.actor.kl_loss_type
```

建议参考范围：

| 指标 | 参考判断 |
|---|---|
| `actor/ppo_kl` | 长期接近 0 通常说明更新太弱 |
| `actor/ppo_kl` | 突然尖峰通常说明更新不稳定 |
| `actor/reward_kl_penalty_coeff` | 如果持续升高且 reward 不涨，优先降低 KL 压力 |

---

## 3. GRPO 场景：检查组内差异是否足够

GRPO 的关键不是单条样本 reward，而是同一个 prompt 下多个 response 的相对差异。

源码逻辑：

```text
verl_latest/verl/trainer/ppo/core_algos.py:268
```

核心逻辑近似为：

```text
scores = token_level_rewards.sum(dim=-1)
按 uid 分组
advantage = score - group_mean
如果 norm_adv_by_std_in_grpo=True:
  advantage = (score - group_mean) / group_std
```

重点看：

```text
actor_rollout_ref.rollout.n
critic/advantages/mean
critic/advantages/max
critic/advantages/min
critic/rewards/mean
critic/rewards/max
critic/rewards/min
```

判断：

| 现象 | 说明 |
|---|---|
| `rollout.n = 1` | GRPO 基本没有组内比较意义 |
| 同组 reward 全 0 | 任务太难、reward 太严、轨迹失败 |
| 同组 reward 全 1 | 任务太简单、reward 太松 |
| reward 方差接近 0 | advantage 弱，梯度信号弱 |
| `critic/advantages/max/min` 很接近 0 | policy 几乎没有有效学习信号 |

建议配置：

```yaml
algorithm.adv_estimator: grpo
actor_rollout_ref.rollout.n: 4-16
algorithm.norm_adv_by_std_in_grpo: true
```

经验建议：

| 场景 | 建议 |
|---|---|
| reward 稀疏、全 0 多 | 增大 `rollout.n`，降低题目难度，增加 partial reward |
| reward 全 1 多 | 提高数据难度，收紧 reward |
| 方差低但平均 reward 中等 | 检查 reward 是否区分度不足 |

---

## 4. PPO 更新是否太弱或被 clipping 限制

重点看 actor 更新指标：

```text
actor/pg_loss
actor/ppo_kl
actor/pg_clipfrac
actor/pg_clipfrac_lower
actor/grad_norm
actor/lr
actor/entropy
actor/entropy_loss
```

源码来源：

```text
verl_latest/verl/workers/utils/losses.py:57
verl_latest/verl/workers/utils/losses.py:119
verl_latest/verl/workers/utils/losses.py:129
verl_latest/verl/trainer/ppo/core_algos.py:1279
```

判断标准：

| 指标 | 正常参考 | 异常含义 |
|---|---|---|
| `actor/pg_clipfrac` | 约 0.05 - 0.30 常见较健康 | 太高说明大量 update 被 clip |
| `actor/pg_clipfrac > 0.5` | 偏高 | 更新过猛或 clip 太严 |
| `actor/pg_clipfrac < 0.01` | 偏低 | 更新可能太弱 |
| `actor/ppo_kl ≈ 0` | 偏低 | policy 几乎不动 |
| `actor/grad_norm ≈ 0` | 偏低 | reward / advantage 信号弱 |
| `actor/grad_norm` 爆炸 | 偏高 | lr、reward scale、batch 可能异常 |

调参入口：

```yaml
actor_rollout_ref.actor.optim.lr
actor_rollout_ref.actor.ppo_epochs
actor_rollout_ref.actor.ppo_mini_batch_size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu

actor_rollout_ref.actor.clip_ratio
actor_rollout_ref.actor.clip_ratio_low
actor_rollout_ref.actor.clip_ratio_high
actor_rollout_ref.actor.clip_ratio_c
```

建议：

| 现象 | 调参方向 |
|---|---|
| `ppo_kl` 低、`pg_loss` 小、`grad_norm` 小 | 增大 lr 或 PPO epoch，降低 KL |
| `pg_clipfrac` 高、`ppo_kl` 高 | 降低 lr，增大 batch，放宽或调整 clip |
| `grad_norm` 爆炸 | 降低 lr，检查 reward scale，增大 batch |

---

## 5. Critic 是否拖累训练

如果使用 PPO / GAE，需要关注 critic。GRPO 通常不依赖 critic，优先级低一些。

重点看：

```text
critic/vf_loss
critic/vf_clipfrac
critic/vpred_mean
critic/vf_explained_var
critic/grad_norm
critic/lr
```

源码来源：

```text
verl_latest/verl/workers/utils/losses.py:147
verl_latest/verl/workers/utils/losses.py:180
verl_latest/verl/trainer/ppo/metric_utils.py:89
```

判断：

| 现象 | 说明 |
|---|---|
| `critic/vf_explained_var` 接近 1 | critic 能较好解释 return |
| `critic/vf_explained_var` 低或为负 | critic 学不好，advantage 噪声大 |
| `critic/vf_loss` 高且不降 | critic 训练失败或 reward 分布不稳定 |
| `critic/vf_clipfrac` 高 | value 更新被大量 clip |

调参入口：

```yaml
critic.enable
critic.optim.lr
critic.ppo_epochs
critic.ppo_mini_batch_size
critic.ppo_micro_batch_size_per_gpu
critic.cliprange_value
```

建议：

| 场景 | 建议 |
|---|---|
| GAE + critic 学不好 | 降低 critic lr，增大 critic batch，检查 reward scale |
| GRPO | 不要优先纠结 critic 指标，先看组内 reward 方差 |

---

## 6. 探索不足：Entropy 和采样参数

重点看：

```text
actor/entropy
actor/entropy_loss
critic/rewards/max
critic/rewards/mean
val-core/*/best@N/mean
val-core/*/mean@N
```

源码来源：

```text
verl_latest/verl/trainer/ppo/ray_trainer.py:1719
verl_latest/verl/workers/utils/losses.py:129
verl_latest/verl/trainer/ppo/metric_utils.py:554
```

判断：

| 现象 | 说明 |
|---|---|
| entropy 快速下降 | 策略过早收敛，探索不足 |
| best@N 上升但 mean@N 不升 | 模型偶尔能答对，但采样不稳定 |
| reward/max 上升但 reward/mean 不升 | 少数轨迹有效，整体探索还不稳定 |

调参入口：

```yaml
actor_rollout_ref.rollout.temperature
actor_rollout_ref.rollout.top_p
actor_rollout_ref.rollout.top_k
actor_rollout_ref.rollout.n

actor_rollout_ref.actor.entropy_coeff
actor_rollout_ref.actor.calculate_entropy
```

建议：

| 场景 | 调参方向 |
|---|---|
| 探索不足 | 提高 temperature / top_p，适当增加 entropy_coeff |
| 输出太随机 | 降低 temperature / top_p |
| pass@k 高但 pass@1 低 | 训练稳定性不足，可降低采样温度并增强有效 reward 区分 |

---

## 7. 输出截断、空响应、Prompt 截断

很多 reward 慢不是算法问题，而是模型根本没看到完整题目，或答案被截断。

重点看：

```text
prompt_length/mean
prompt_length/max
prompt_length/clip_ratio

response_length/mean
response_length/max
response_length/clip_ratio
response_length_non_aborted/clip_ratio
response/aborted_ratio
```

源码来源：

```text
verl_latest/verl/trainer/ppo/metric_utils.py:89
verl_latest/verl/trainer/ppo/metric_utils.py:236
```

判断：

| 指标 | 异常含义 |
|---|---|
| `prompt_length/clip_ratio` 高 | prompt 被截断，模型没看到完整输入 |
| `response_length/clip_ratio` 高 | response 被截断，答案可能没输出完 |
| `response/aborted_ratio` 高 | rollout 失败或生成空响应 |
| `response_length/max` 经常顶到上限 | max response length 太小或输出太啰嗦 |

调参入口：

```yaml
data.max_prompt_length
data.max_response_length

actor_rollout_ref.rollout.prompt_length
actor_rollout_ref.rollout.response_length
actor_rollout_ref.rollout.max_model_len
```

建议参考：

| 现象 | 建议 |
|---|---|
| prompt clip 高 | 增大 prompt length，或清洗过长样本 |
| response clip 高 | 增大 response length，或约束输出格式 |
| aborted 高 | 检查 rollout engine、EOS、模板、采样参数 |

---

## 8. 吞吐瓶颈：Reward 看起来慢，可能只是训练步数太少

重点看：

```text
timing_s/gen
timing_s/reward
timing_s/ref
timing_s/values
timing_s/update_actor
timing_s/update_critic
timing_s/step

perf/total_num_tokens
perf/time_per_step
perf/throughput
```

源码来源：

```text
verl_latest/verl/trainer/ppo/metric_utils.py:271
verl_latest/verl/trainer/ppo/metric_utils.py:313
verl_latest/verl/trainer/ppo/metric_utils.py:345
```

判断：

| 现象 | 说明 |
|---|---|
| `timing_s/gen` 高 | rollout 生成瓶颈 |
| `timing_s/reward` 高 | reward function / reward model 慢 |
| `timing_s/update_actor` 高 | actor 训练瓶颈 |
| `perf/throughput` 低 | 同样 wall-clock 下训练 step 少，reward 看起来慢 |

调参入口：

```yaml
data.train_batch_size

actor_rollout_ref.rollout.max_num_batched_tokens
actor_rollout_ref.rollout.max_num_seqs
actor_rollout_ref.rollout.gpu_memory_utilization
actor_rollout_ref.rollout.tensor_model_parallel_size

actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
actor_rollout_ref.actor.ppo_max_token_len_per_gpu
actor_rollout_ref.actor.use_dynamic_bsz
```

建议：

| 瓶颈 | 优化方向 |
|---|---|
| generation 慢 | 增大 batch 并发，调 rollout engine 参数 |
| reward 慢 | 并行化 reward，缓存 deterministic reward |
| actor update 慢 | 调 micro batch、dynamic batch、token len per GPU |
| 显存接近上限 | 降低 batch/token 上限或开启更保守的 rollout memory 配置 |

---

## 9. 显存和 OOM 风险

重点看：

```text
actor/perf/max_memory_allocated_gb
actor/perf/max_memory_reserved_gb
actor/perf/cpu_memory_used_gb

critic/perf/max_memory_allocated_gb
critic/perf/max_memory_reserved_gb
critic/perf/cpu_memory_used_gb
```

源码来源：

```text
verl_latest/verl/workers/engine_workers.py
```

判断：

| 现象 | 说明 |
|---|---|
| allocated 接近设备上限 | OOM 风险高 |
| reserved 远高于 allocated | 可能存在 cache / fragmentation |
| CPU memory 持续升高 | dataloader、reward、日志或轨迹保存可能有内存压力 |

调参入口：

```yaml
actor_rollout_ref.rollout.gpu_memory_utilization
actor_rollout_ref.rollout.free_cache_engine
actor_rollout_ref.rollout.max_num_batched_tokens
actor_rollout_ref.rollout.max_num_seqs

actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
critic.ppo_micro_batch_size_per_gpu
```

---

## 10. Validation 指标：区分“训练 reward 涨”和“真实精度涨”

重点看：

```text
val-core/{data_source}/{var}/mean@N
val-core/{data_source}/{var}/std@N
val-core/{data_source}/{var}/best@N/mean
val-core/{data_source}/{var}/worst@N/mean
val-core/{data_source}/{var}/maj@N/mean
```

源码来源：

```text
verl_latest/verl/trainer/ppo/metric_utils.py:554
```

判断：

| 现象 | 说明 |
|---|---|
| `mean@N` 不升 | 平均验证表现没有提升 |
| `best@N` 升但 `mean@N` 不升 | 偶尔能答对，但不稳定 |
| `std@N` 高 | 同题多采样差异大 |
| `maj@N` 低 | 多数投票仍不稳定，或 pred 提取有问题 |
| train reward 升、val acc 不升 | 可能 reward hacking 或分布不一致 |

建议：

```yaml
trainer.val_before_train: true
trainer.test_freq: 合理设置
trainer.log_val_generations: 建议开启少量样本
trainer.validation_data_dir: 保存验证输出
```

---

## 11. Reward function 本身是否有问题

VERL reward 入口：

```text
verl_latest/verl/trainer/ppo/reward.py
verl_latest/verl/workers/reward_manager/naive.py
verl_latest/verl/workers/reward_manager/dapo.py
verl_latest/verl/utils/reward_score/__init__.py
```

关键配置：

```yaml
reward.custom_reward_function.path
reward.custom_reward_function.name
reward.custom_reward_function.reward_kwargs

reward.reward_manager.source
reward.reward_manager.name
```

自动检测规则：

| 检测项 | 异常信号 | 常见原因 | 自动动作 |
|---|---|---|---|
| reward 区分度 | reward 全 0、全 1 或 std 接近 0 | 任务过难/过易、reward 过严/过松 | 调整数据难度、增加 partial reward 或收紧 reward |
| reward 与 eval accuracy 一致性 | reward 高但 `acc=0`，或 reward 低但 `acc=1` | reward hacking、答案解析和验证逻辑不一致 | 标记冲突样本并进入解析一致性检测 |
| reward 严格度 | 大量格式错误直接 0，且正确性无法区分 | 格式 reward 过硬、早期探索不足 | 降低格式权重或分阶段收紧格式约束 |
| reward 宽松度 | 无正确推理也能获得高分 | 格式漏洞、关键词刷分、解析规则过松 | 加强答案校验和反作弊规则 |
| answer extraction 稳定性 | `pred` 为空、解析异常、同一答案多种解析结果 | 正则/模板脆弱、输出格式漂移 | 记录解析错误类型并回退到更鲁棒解析器 |
| extra info 完整性 | 缺少 `acc`、`pred`、`gold`、`error_type` 等字段 | reward 日志不可诊断 | 自动标记为不可解释 reward，降低调参建议置信度 |

建议 reward extra info 至少包含：

```text
acc
pred
gold
format_reward
answer_reward
tool_success
error_type
```

---

## 12. 轨迹异常：参数调不动时必须看样本

如果以下情况出现，不要继续只调参数：

```text
pass@k = 0
reward/std 接近 0
advantage 接近 0
response/aborted_ratio 高
prompt/response clip_ratio 高
train reward 涨但 val acc 不涨
```

应自动读取 rollout / validation 样本并执行规则检测：

```yaml
trainer.rollout_data_dir
trainer.validation_data_dir
trainer.log_val_generations
```

重点自动检测：

| 检测项 | 异常信号 |
|---|---|
| prompt 完整性 | 输入字段缺失、prompt 被截断、chat template 渲染异常 |
| response 完整性 | 空响应、被截断、未出现最终答案、格式不匹配 |
| 答案解析一致性 | `pred` 为空、解析异常、同一 response 多种解析结果不一致 |
| reward 与答案一致性 | 正确答案被打 0、错误答案被打 1、reward 与 `acc` 冲突 |
| reward hacking 模式 | 固定模板刷分、绕过格式约束、非解题内容获得高 reward |
| 多轮工具任务稳定性 | tool call 失败、timeout、observation 过长、工具错误未进入 `error_type` |

---

## 13. 数据难度问题

当指标显示：

```text
同组 reward 全 0
val best@N 也接近 0
pass@k = 0
reward/max 长期不涨
```

通常说明模型没有成功轨迹，GRPO/PPO 很难从纯失败样本中学习。

建议：

| 场景 | 处理 |
|---|---|
| 任务太难 | 先用更简单数据 curriculum |
| reward 太稀疏 | 加 partial reward |
| 模型初始能力太弱 | 换更强 base model 或先 SFT |
| rollout.n 太小 | 增大到 4-16 |
| answer 格式过严 | 早期放宽格式 reward，再逐步收紧 |

如果：

```text
同组 reward 全 1
solve_all 很多
```

说明任务太简单或 reward 太松，也会让 GRPO 缺少有效对比。

---

## 14. 常见问题到调参动作速查表

| 问题现象 | 优先看 | 可能原因 | 建议动作 |
|---|---|---|---|
| reward 不涨 | `score/mean`, `rewards/mean` | reward 无信号 | 查 reward、轨迹、数据 |
| score 涨 reward 不涨 | KL 指标 | KL penalty 太强 | 降低 KL 系数 |
| GRPO 没效果 | reward 方差、advantages | 组内差异小 | 增大 rollout.n，调 reward |
| policy 不动 | `ppo_kl≈0`, `grad_norm≈0` | 更新太弱 | 增大 lr / PPO epoch，降低 KL |
| 更新不稳定 | `ppo_kl` spike, `clipfrac` 高 | 更新过猛 | 降 lr，增 batch |
| 输出被截断 | length clip ratio | max length 太小 | 增大 max length |
| 空响应多 | aborted ratio | rollout/template/EOS 问题 | 查生成配置和模板 |
| 训练慢 | throughput/timing | 生成或 reward 瓶颈 | 优化 rollout/reward 并发 |
| train 涨 val 不涨 | validation 指标、样本 | reward hacking | 查 reward 与验证分布 |
| pass@k=0 | validation + 轨迹 | 太难或工程异常 | 看样本、降难度、加 partial reward |

---

## 15. 推荐的默认排查顺序

```text
1. 看 reward/score 曲线
2. 看 KL 是否压制 score
3. 看 GRPO 组内 reward 方差和 advantage
4. 看 PPO 更新强度：ppo_kl / clipfrac / grad_norm / lr
5. 看 entropy 和采样探索
6. 看 prompt/response 是否截断或空响应
7. 看 validation mean@N / best@N / maj@N
8. 看 rollout 样本和 reward extra info
9. 看数据难度：全 0、全 1、partial 比例
10. 最后再系统性调 batch、lr、KL、rollout.n、temperature
```

一句话原则：

```text
Reward 提升慢时，先确认“有没有有效学习信号”，再判断“policy 有没有被允许更新”，最后才调吞吐和训练效率。
```

如果 `pass@k=0`、同组 reward 全 0、轨迹大量异常，只调学习率、batch size、PPO epoch 通常没有用，应优先处理 reward、轨迹和数据难度。
