# verl 训练典型错误场景速查

> 基于 verl 实际记录的指标，列出常见训练失败模式的识别、根因和修复。

---

## 场景 1: Reward 完全不涨 (Flatline)

**日志表现**:
```
critic/score/mean: 持续 ≈ 0 (或某个固定值), 50+ step 不变
critic/score/max:  持续 = 0
actor/pg_loss:     很低且稳定 (0.001 级别)
actor/grad_norm:   接近 0
actor/ppo_kl:      接近 0
```

**直接原因**: 所有 rollout 的 reward 都相同 → advantage 全为 0 → 梯度为 0。

**排查**:

| 检查项 | 命令/方法 |
|---|---|
| reward 函数能正常返回非零值吗 | 手动跑一次 `compute_score`，看是否返回 0 |
| 数据是否太难，模型完全解不出来 | 看 `critic/score/max`，如果训练 20 步仍为 0，数据太难 |
| rollout 采样是否用 greedy (do_sample=false) | 检查 `rollout.do_sample` 和 `temperature` |
| 数据 tokenization 是否正确 | 检查 prompt 在 tokenize 后是否被截断或损坏 |
| `n=1` 但用了 GRPO | `rollout.n=1` 时 group 只有 1 个样本，所有 advantage = 0 |

**修复**:
- 数据太难 → 加入简单 warmup 样本或换更简单的数据
- 数据 tokenization 错误 → 检查 chat template，修复预处理
- `n=1` → GRPO 需要 `n ≥ 4`
- greedy sampling → `do_sample=True`, `temperature=1.0`

---

## 场景 2: Reward 先涨后崩 (Rise then Collapse)

**日志表现**:
```
step 0-50:   critic/score/mean 稳步上升 0 → 0.5+
step 50-80:  critic/score/mean 到达峰值
step 80+:    critic/score/mean 快速下降
同时:
  actor/ppo_kl:      在 score 下降前已经开始上升
  actor/entropy:     可能在下降 (模式坍缩) 或在上升 (策略发散)
  actor/pg_clipfrac: 升高
```

**直接原因**: 模型学到了 reward 函数的漏洞 (reward hacking)，而非真正能力，漏洞被后续训练纠正时 reward 崩塌。

**排查**:

| 检查项 | 方法 |
|---|---|
| 抽查 collapse 前后的生成样本 | 看是否出现重复模式、特定 escape 序列 |
| reward 是单维度还是多维度 | 单维度更容易被 hack |
| response_length/mean 是否在涨 | 模型可能在「水长度」获取得分 |

**修复**:
- reward 函数加固：增加 format 检查、禁止的模式过滤
- 用 `gdpo` + `gdpo_reward_keys` 拆成多维度 reward
- 用 `loss_agg_mode: seq-mean-token-sum-norm` 消除长度偏差
- Agent 场景：启用 compact filtering（过滤未主动 submit 的 trajectory）

---

## 场景 3: KL 发散 (KL Runaway)

**日志表现**:
```
actor/ppo_kl: 0.01 → 0.05 → 0.2 → 0.8 → 2.0+   (单调加速上升)
actor/grad_norm: 伴随 spikes
actor/pg_clipfrac: 从 0.1 升到 0.4+
critic/score/mean: 可能同时下降
```

**直接原因**: 每次更新的步长太大，策略迅速偏离旧策略，agent 行为与 rollout 时完全不同。

**排查**:

| 检查项 | 方法 |
|---|---|
| lr 是否偏高 | 32B model: `lr > 2e-6` 偏高, 7B model: `lr > 5e-6` 偏高 |
| 是否有 KL 约束 | `use_kl_loss=False` 且 `use_kl_in_reward=False` 则完全无约束 |
| ppo_epochs > 1 | 每 batch 多次更新，积累偏离 |
| clip_ratio 是否偏大 | `clip_ratio > 0.3` |
| 训练步数很少就出现 | 可能是某个 batch 有 outlier 样本 |

**修复**:
```yaml
# 立即
actor.optim.lr: 降低 2-5x
actor.clip_ratio: 0.1
actor.clip_ratio_high: 0.15
actor.use_kl_loss: true
actor.kl_loss_coef: 0.01

# 如果已发散严重：回滚 checkpoint + 以上修改后重训
```

---

## 场景 4: 所有 Response 趋同 (Mode Collapse)

**日志表现**:
```
actor/entropy:      持续下降到 < 0.1
response_length/mean: 稳定在一个值，std 接近 0
critic/score/max == critic/score/min:  所有样本得分完全相同 (但非零)
actor/pg_clipfrac:  很低 (< 0.02)
actor/ppo_kl:       很低
```

**直接原因**: 策略坍缩到确定性输出，不管什么 prompt 都生成几乎相同的 response。

**排查**:

| 检查项 | 方法 |
|---|---|
| temperature 是否过低 | `temperature < 0.6` 或 `do_sample=False` |
| entropy_coeff 是否关闭 | `entropy_coeff=0` |
| 训练数据是否太单一 | 看数据是否全是同类 prompt |
| clip_ratio_high 是否过低 | `clip_ratio_high=0.2` 限制了正样本的探索 |

**修复**:
```yaml
rollout.temperature: 1.2
rollout.top_p: 0.95
rollout.do_sample: true
actor.clip_ratio_high: 0.28       # DAPO clip-higher
actor.entropy_coeff: 0.0005       # 临时启用，不要长期开
actor.calculate_entropy: true
```

---

## 场景 5: Response 长度失控 (Length Runaway)

**日志表现**:
```
response_length/mean:   每 10 step 增长 > 20%
response_length/clip_ratio:  上升，> 0.3
response_length/max:    持续触顶 (max_response_length)
perf/throughput:        下降 (因为更长序列)
```

**直接原因**: `loss_agg_mode: token-mean` 下长 response 获得更多梯度权重 → 正反馈循环。

**排查**:

| 检查项 | 方法 |
|---|---|
| loss_agg_mode | 如果是 `token-mean`，长序列天然获得更多梯度 |
| reward 是否与长度正相关 | 散点图看 response_length vs reward |
| ignore_eos 是否被设为 True | 模型被允许无限生成 |

**修复**:
```yaml
# 核心修复
actor.loss_agg_mode: seq-mean-token-sum-norm

# 辅助
rollout.ignore_eos: false
# 可在 reward 中加入微小的长度惩罚
```

---

## 场景 6: 梯度爆炸 / Loss NaN (Gradient Explosion)

**日志表现**:
```
actor/grad_norm:  突然 spike 到 100+ 或直接 NaN
actor/pg_loss:    变为 NaN
后续所有指标:     全部 NaN
```

**直接原因**: 某个 batch 触发了数值溢出，通常从 grad_norm 的 spike 开始，传播到 loss 和后续参数。

**源码中 NaN 可能出现的位置**:
- `torch.exp(negative_approx_kl)` — 如果 `log_prob - old_log_prob` 偏差过大 (core_algos.py:1332)
- `score / (std + epsilon)` — GRPO 归一化时 `norm_adv_by_std_in_grpo=True` 且 epsilon 不够大 (core_algos.py:326)
- FP16 精度下 `log_prob` 计算溢出

**排查**:

| 检查项 | 方法 |
|---|---|
| NaN 发生前 actor/ppo_kl 是否已经很高 | > 1.0 → 策略已经发散了 |
| NaN 发生前 actor/grad_norm 是否有 spike | > 50 → 那个 batch 就是导火索 |
| dtype 是 FP16 还是 BF16 | FP16 更容易溢出 |
| 是否用了 `norm_adv_by_std_in_grpo=True` | True 时如果 group std=0，除以 epsilon |

**修复**:
1. **回滚**到最近的有效 checkpoint
2. 降低 `lr` 5-10x
3. `dtype: bfloat16` (替代 fp16)
4. `norm_adv_by_std_in_grpo: False` (避免除以接近 0 的 std)
5. `clip_grad: 1.0` (确保 grad clip 生效)

---

## 场景 7: GPU OOM (Out of Memory)

**日志表现**:
```
CUDA out of memory. Tried to allocate XXX MiB
训练直接中断
```

**常见 OOM 阶段和对应参数**:

| OOM 阶段 | 识别 | 核心参数 |
|---|---|---|
| Rollout 生成时 | `timing_s/gen` 报错 | `rollout.gpu_memory_utilization`↓, `rollout.max_num_batched_tokens`↓, `rollout.max_num_seqs`↓ |
| Actor 训练 (forward) | `timing_s/update_actor` 报错 | `actor.ppo_micro_batch_size_per_gpu`↓, `actor.ppo_max_token_len_per_gpu`↓ |
| Actor 训练 (backward) | forward 过但 backward 报错 | `actor.fsdp_config.param_offload=true`, `actor.fsdp_config.optimizer_offload=true` |
| Ref log_prob 计算 | `timing_s/ref` 报错 | `ref.log_prob_micro_batch_size_per_gpu`↓, `ref.log_prob_max_token_len_per_gpu`↓ |
| Rollout weights sync | `timing_s/update_weights` 后 | `rollout.free_cache_engine=true` |

**通用减显存组合**:
```yaml
rollout:
  gpu_memory_utilization: 0.35
  free_cache_engine: true
  enforce_eager: true          # 关 CUDA graph 省显存

actor:
  ppo_micro_batch_size_per_gpu: 1
  use_dynamic_bsz: true
  ppo_max_token_len_per_gpu: 8192
  fsdp_config:
    param_offload: true
  calculate_entropy: false      # 不计算熵
  use_kl_loss: false            # 不需要 ref forward
  use_torch_compile: false      # 省编译缓存
```

---

## 场景 8: Agent 学不会停下 (Never Submit / Infinite Loop)

**日志表现**:
```
num_turns/mean:           持续增长到 max_assistant_turns
num_turns/max:            触顶
response/aborted_ratio:   上升 (超时或超步数)
critic/score/mean:        很低
response_length/mean:     很大
```

**直接原因**: Agent 没学会在解决问题后调用 finish/submit 工具，一直继续行动直到被环境强制终止。

**修复**:
1. **Compact filtering**: 过滤掉达到 max steps 或 timeout 的 trajectory（DeepSWE 方法）
2. 在 reward 中鼓励 timely submit：正确且 submit → +1，正确但未 submit → 0
3. 检查 finish tool 的定义是否清晰，模型是否能正确理解
4. 减小 `multi_turn.max_assistant_turns`，给更明确的限制信号

---

## 场景 9: 验证集和训练集趋势背离 (Validation Divergence)

**日志表现**:
```
critic/score/mean:        上涨
val-core/.../mean@N:     下降或停滞
```

**直接原因**: 过拟合训练数据分布。

**修复**:
- 增大训练数据多样性
- 检查训练集和验证集是否存在分布差异
- 减少 `ppo_epochs` 到 1
- 启用 KL 约束限制策略偏离

---

## 场景 10: Clip Fraction 异常

### 10a: clipfrac 过高 (> 0.4)

**日志表现**:
```
actor/pg_clipfrac:       持续 > 0.4
actor/ppo_kl:            偏高
actor/grad_norm:         可能波动大
```

**直接原因**: 大量 token 的 ratio 超出 `[1-ε, 1+ε]` 范围被 clip → 要么 clip_ratio 太小，要么 lr 太大导致每次更新跨度过大。

**修复**: 降低 lr 或增大 clip_ratio。

### 10b: clipfrac 极低 (< 0.01)

**日志表现**:
```
actor/pg_clipfrac:       < 0.01
actor/ppo_kl:            接近 0
critic/score/mean:       不涨
```

**直接原因**: 策略几乎没有变化 → 要么 lr 太小，要么 advantage=0。

**修复**: 
- 检查 `actor/grad_norm` 是否 ≈ 0 (比 clipfrac 更直接)
- 增大 `lr`
- 检查数据/reward 是否有问题导致 advantage 全为 0

---

## 场景 11: 多机训练卡住 (Hang / Deadlock)

**日志表现**:
```
所有指标停止更新，但进程没有退出
timing_s/gen 或 timing_s/update_actor 的时长异常长
Ray worker 超时
```

**常见原因**:
- NCCL 通信超时：`nccl_timeout: 600` 不够 → 增大
- Ray worker 注册失败：增大 `ray_wait_register_center_timeout`
- vLLM/SGLang server 启动失败：检查引擎日志
- 某个 worker OOM 静默失败：dmesg 查看

**修复**:
```yaml
actor_rollout_ref.nccl_timeout: 1200
trainer.ray_wait_register_center_timeout: 600
```

---

## 场景 12: GRPO group 内全对或全错 (Zero-Variance Groups)

**日志表现**:
```
critic/score/max == critic/score/min  (同一 step 内)
critic/advantages/max == critic/advantages/min == 0  (如果 norm_adv_by_std_in_grpo=False)
actor/grad_norm: 接近 0
```

**直接原因**: 同一 prompt 的 N 个 rollout 要么全部成功要么全部失败 → group 内方差为 0 → GRPO 计算出 advantage=0 → 该 group 不贡献有效梯度。

**根本原因**:
- 全对：数据太简单，所有 rollout 都正确
- 全错：数据太难，所有 rollout 都失败

**修复**:
- 全对 → 换更难的数据，或降低 temperature 让探索收敛
- 全错 → 换更简单数据，或提高 temperature 增加多样性
- 用 `algorithm.filter_groups.enable=True` + `metric: seq_reward` 过滤掉零方差 group
- 用 `rloo_vectorized` 而非 `grpo`：RLOO 在 group 内 reward 相同时 advantage 也是 0，但至少不会除以零

---

## 场景速查矩阵

| 场景 | reward | ppo_kl | entropy | grad_norm | pg_clipfrac | resp_len | 动作优先级 |
|---|---|---|---|---|---|---|---|
| 1. 完全不学 | →0 | →0 | → | →0 | → | → | 检查数据+reward |
| 2. 先涨后崩 | ↗↘ | ↗ | ↓或↑ | 波动 | ↗ | → | 加固 reward |
| 3. KL 发散 | →或↘ | ↗↗ | ↑ | ↗ spike | ↗ | → | 降 lr + 加 KL |
| 4. 模式坍缩 | → | →0 | ↓↓ | →0 | ↓↓ | std→0 | 提 temp + 提 clip_high |
| 5. 长度失控 | → | → | → | → | → | ↗↗ | 改 loss_agg_mode |
| 6. NaN | NaN | 高前兆 | ↑或↓ | NaN | 高 | → | 回滚 + 降 lr |
| 7. OOM | — | — | — | — | — | — | 降 batch/长度 |
| 8. Agent 不停止 | ↓ | → | → | → | → | ↗↗ | compact filter |
| 9. 验证背离 | ↗(train) | → | → | → | → | → | 加数据多样性 |
| 10a. clipfrac高 | → | ↗ | → | ↗ | ↗↗ | → | 降 lr 或增 clip |
| 10b. clipfrac低 | → | →0 | → | →0 | ↓↓ | → | 提 lr，查 reward |
| 11. 训练卡住 | 静止 | 静止 | 静止 | 静止 | 静止 | 静止 | 查 NCCL/Ray |
| 12. 零方差group | → | → | → | →0 | → | → | filter_groups |

> 图例: `→` 稳定, `↗` 上升, `↘` 下降, `↗↗` 快速上升, `↓↓` 快速下降, `→0` 趋近 0, `—` 不适用
