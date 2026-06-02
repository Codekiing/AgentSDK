# Round 4 VERL 源码级根因分析

生成时间: 2026-06-01 (VERL source-level analysis)
分析范围: Session ae18fdb6-ae30-495b-b1c7-5959d2e445be, run_verl_gsm8k_0.5b_r4

## 核心发现: Train-Val Paradox

```
R3 → R4 唯一变更: num_problems 48→96, seed 45→46
结果:
  train score: 0.553 → 0.597 (+8%)  ← 训练更好
  val-acc:     0.475 → 0.439 (-8%)  ← 验证更差
  entropy:     0.132稳定 → 0.107振荡 (50% steps <0.1)
```

## 因果链 #1: num_problems 48→96 → entropy 振荡

[数据链]
```
R3 entropy (48 problems): 0.157→0.152, rate=0.0004/step, 0% below 0.1
R4 entropy (96 problems): 0.120→0.096, rate=0.0010/step, 50% below 0.1

R4 entropy per step:
  0.120→0.180→0.119→0.115→0.094→0.123→0.110→0.129→0.119
  →0.097→0.111→0.096→0.130→0.099→0.087→0.117→0.089→0.097
  →0.065→0.080→0.114→0.095→0.085→0.096
```

[VERL 源码因果]
```
losses.py:128: policy_loss -= 0.001 * entropy_loss

entropy_coeff=0.001 在48 problems时足够:
  → 每个problem被模型在~2个step中见到 (48/24=2 batches/epoch)
  → 2次接触 × 0.001 entropy push = 0.002 effective exploration per problem

entropy_coeff=0.001 在96 problems时不够:
  → 每个problem被模型在~1个step中见到 (96/48=2 batches/epoch but 2x more unique problems)
  → 1次接触 × 0.001 entropy push = 0.001 effective exploration per problem
  → entropy push减半! → entropy振荡, 不时跌破0.1
```

[因果结论]
  entropy_coeff=0.001 在48 problems时足够维持entropy稳定,
  但在96 problems时"探索压力密度"减半, 不足以阻止entropy波动。
  这不是entropy_coeff配置错误 — 是problems增加后需要相应增加coeff。

## 因果链 #2: entropy 振荡 → val 退化 (Failure Mode #13)

[数据链]
```
R4 val-acc trend:
  Step 0:  0.491 ← 起始 (与R3的0.496相当)
  Step 5:  0.487 ← 稳定
  Step 10: 0.448 ← 开始下降
  Step 15: 0.448
  Step 20: 0.443
  Step 24: 0.439 ← 持续下降 (连续4降!)

train score vs val 背离:
  train score ↑ (0.594→0.597 mean, oscillating)
  val ↓ (0.491→0.439)
```

[VERL 源码因果]
```
core_algos.py:267-331 (GRPO advantage):
  entropy振荡 → 某些step的策略坍缩到低entropy
  → 在这些step, 模型学到的是"窄模式" (narrow pattern)
  → 窄模式在training data上有效 (train score↑)
  → 窄模式在validation data上无效 (val↓)
  → 每次entropy低谷都是一次"过拟合脉冲"

metric_utils.py:554-702 (validation):
  val = bootstrap mean accuracy over val set
  → 反映模型在未见数据上的泛化能力
  → R4的val单调下降 = 过拟合加速
```

[因果结论]
  entropy振荡 → 交替的"坍缩/恢复"周期
  → 坍缩期间学到的窄模式在val上无效
  → val从0.491持续降到0.439
  → 这是 Failure Mode #13 (验证背离) + entropy不稳定变体

## 因果链 #3: score ↑ → 过拟合信号

[数据链]
```
R4 score: mean=0.597, std=0.120
R3 score: mean=0.553, std=0.094

R4 score比R3高0.044 (8% improvement)
BUT val比R3低0.036 (8% degradation)
```

[VERL 源码因果]
```
core_algos.py:306: advantage = (score - group_mean) / group_std

GRPO组内: 模型学会在训练集的特定问题上找到更好的推理路径
→ train score上升
但在验证集上: 这些推理路径不泛化 (overfitting)
→ val下降

Train-val gap: 0.597 - 0.439 = 0.158 (R3: 0.553 - 0.475 = 0.078)
→ gap扩大2x → 过拟合程度加倍
```

## 因果链 #4: grad_norm ↑ → 梯度竞争加剧

[数据链]
```
R4 grad_norm: mean=2.91, std=0.62, max=4.54
R3 grad_norm: mean=2.41, std=0.29, max=2.85

R4 grad_norm比R3高21%, std高2.1x
```

[VERL 源码因果]
```
FSDP engine: ||∇L||₂ = sqrt(sum of squared gradients over all params)

grad_norm升高 + std升高 → 不同step之间的梯度差异更大
→ 某些step: 大grad (来自"惊喜"的batch)
→ 某些step: 正常grad
→ 梯度不一致性增加 → 训练不稳定
```

## 因果链 #5: KL 仍健康 → 策略未发散

[数据链]
```
R4 KL: mean=0.000833, max=0.00377 (均abs值)
R3 KL: mean=0.000440, max=0.00113
```

[VERL 源码因果]
```
core_algos.py:1333: ppo_kl = masked_mean(old_log_prob - log_prob)

KL虽比R3高1.9x, 但仍在健康范围(<0.05)
→ 模型没有策略发散 (#1 Syndrome)
→ 问题不是"模型更新太激进", 而是"更新方向不够泛化"
```

## 综合因果模型

```
R4 退化的完整因果链:

  num_problems: 48 → 96  ← R3推荐的变更
       │
       ├──[DataLoader] 每个problem接触频率减半
       │      │
       │      └──[losses.py:128] entropy_coeff=0.001 不变
       │             │
       │             └── 有效探索压力密度减半
       │                    │
       │                    ├── entropy振荡 (50% steps <0.1)
       │                    │      │
       │                    │      ├── 低entropy step: 模型坍缩到窄模式
       │                    │      │      └── 窄模式在train data上有效
       │                    │      │             └── train score ↑ (0.553→0.597)
       │                    │      │
       │                    │      └── 高entropy step: entropy_coeff推回
       │                    │             └── 但窄模式已被"记住"
       │                    │                    └── val持续退化 (0.491→0.439)
       │                    │
       │                    └── grad_norm波动增大 (std 0.29→0.62)
       │                           └── 训练不稳定
       │
       └── val 4次连续下降 → Failure Mode #13 (验证背离)

对比 R3 为什么没有这个问题:
  R3: 48 problems, entropy_coeff=0.001
    → 每个problem ~2 contacts per epoch
    → entropy push density = 0.001 × 2 = 0.002 per problem
    → entropy 稳定在 0.15→0.15
    → val 稳定 (仅2点, 0.496→0.475)

  R4: 96 problems, entropy_coeff=0.001
    → 每个problem ~1 contact per epoch
    → entropy push density = 0.001 × 1 = 0.001 per problem
    → entropy 振荡 (50% <0.1)
    → val 持续退化 (0.491→0.439, 4连降)
```

## 优化建议

| 优先级 | 目标 | 描述 | VERL 根因 |
|--------|------|------|----------|
| 🔴P0 | rllm-config | entropy_coeff 从 0.001 提高到 0.002 (配合96 problems) | losses.py:128 — 探索压力密度需与problems数匹配 |
| 🟡P1 | rllm-config | num_problems 96→64 (折中: 更多步数但保持entropy稳定) | 48太短, 96不稳, 64可能最佳 |
| 🟡P1 | rllm-config | temperature 0.7→0.85 (提高初始entropy, 减少振荡) | 高temp→高初始entropy→缓冲entropy下降 |

## 建议 Patch

### Patch 1 [P0]: rllm-config — entropy_coeff 与 num_problems 联动

基于 VERL losses.py:128 的机制 + R3/R4 实证:
- entropy_coeff 的有效探索压力 = coeff × contact_frequency
- contact_frequency = batch_size / num_problems
- 当 problems 增加时, coeff 必须相应增加

将 entropy_coeff 默认值从固定0.001改为与 num_problems 联动:
  problems ≤ 48: coeff = 0.001 (R3验证)
  problems 49-96: coeff = 0.002 (需要验证)
  problems 97-128: coeff = 0.003 (需要验证)
