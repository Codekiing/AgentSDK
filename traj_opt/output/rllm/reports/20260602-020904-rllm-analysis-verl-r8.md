# Round 8 VERL 源码级根因分析

生成时间: 2026-06-02
分析范围: run_verl_gsm8k_0.5b_r8, 80 train steps
VERL 机制: GRPO + use_kl_loss(True) + FSDP(4 GPU) + entropy_coeff(0.003)

## 核心发现: 平滑 Entropy 坍缩 ≠ 有害

R8 首次实现 **entropy 下降但 val 上升**，颠覆了之前"entropy 崩塌 → val 退化"的理论。

```
R3: entropy 稳定(0.16→0.15),  val=0.475 (改善, 但仅13步)
R4: entropy 振荡(0.12→0.10),  val=0.491→0.439 (退化!)
R8: entropy 平滑(0.14→0.07),  val=0.470→0.522 (改善!)  ← 反例!
```

## 因果链 #1: batch_size=32 → 平滑梯度 → 平滑 Entropy 坍缩

[数据链]
```
10-step windows:
  Steps  1-10: score=0.566  entropy=0.129  gn=0.96
  Steps 11-20: score=0.636  entropy=0.117  gn=1.00
  Steps 21-30: score=0.685  entropy=0.102  gn=0.92
  Steps 31-40: score=0.719  entropy=0.094  gn=0.93
  Steps 41-50: score=0.757  entropy=0.084  gn=0.89
  Steps 51-60: score=0.774  entropy=0.077  gn=0.86
  Steps 61-70: score=0.799  entropy=0.073  gn=0.88
  Steps 71-80: score=0.809  entropy=0.070  gn=0.92

entropy 下降速率: (0.129-0.070)/8 windows = 0.0074/window → smooth!
score 上升速率: (0.809-0.566)/8 windows = 0.030/window → consistent!
```

[VERL 源码因果]
```
core_algos.py:1138-1199 (agg_loss):
  batch_size=32 → 32 problems per step
  → gradient = (1/32) × Σᵢ ∇Lᵢ  (32个problem的平均)
  → gradient_noise = σ/√32 << σ/√2 (R4)
  → 梯度方向更一致 → 每次update的方向更稳定
  → entropy沿一致的方向平滑下降，而非震荡
```

[R4 vs R8 对比]
```
R4 (batch=2):  gradient per step = (1/2)Σ → high variance
  → step1: 拉向mode A, step2: 拉向mode B, step3: 拉回mode A
  → entropy 震荡 (alternating between high and low)
  → val 退化 (学到矛盾的窄模式)

R8 (batch=32): gradient per step = (1/32)Σ → low variance
  → 每个step的梯度是32个problem的共识方向
  → entropy 沿共识方向平滑下降
  → val 改善 (收敛到一致的有效模式)
```

## 因果链 #2: use_kl_loss=True → 策略有界

[数据链]
```
KL observed: mean=0.000090, max=0.000329
KL loss: use_kl_loss=True, kl_loss_coef=0.01, kl_loss_type=low_var_kl
```

[VERL 源码因果]
```
losses.py:132-142:
  kld = kl_penalty(logprob, ref_logprob, kl_penalty='low_var_kl')
  kl_loss = agg_loss(kld, response_mask)
  policy_loss += 0.01 * kl_loss  ← KL penalty

low_var_kl 计算的是 per-token KL 的低方差估计:
  → 对高方差token的KL赋予更低的权重
  → 总KL值偏低 (0.00009 vs standard KL通常~0.001+)
  → 但仍在起作用: 策略不会偏离ref太远
```

## 因果链 #3: val 持续上升 — 首次! 

[数据链]
```
val-acc trend (17 points over 80 steps):
  0.470→0.485→0.487→0.506→0.491→0.493→0.504→0.510
  →0.504→0.511→0.529→0.528→0.533→0.521→0.500→0.502→0.522
  
  First 40 steps:  avg=0.494
  Last 40 steps:   avg=0.515  (+0.021)
  
  Best: 0.533 at step 60
  Final: 0.522 at step 80
```

[VERL 源码因果]
```
ray_trainer.py:1392-1396 (_validate):
  每 test_freq=5 步验证 → 17个数据点
  → 这是最密集的验证 (R4仅6点, R3仅2点)
  → 趋势判断可靠

metric_utils.py:554-702:
  val = bootstrap @1 → greedy decode准确率
  → 0.522 = 模型在未见数据上的泛化能力
```

## 因果链 #4: 80步训练 — 充足的训练量

[数据链]
```
train_batch_size=32, ppo_mini_batch_size=16, epochs=10
80 steps × 32 problems/step = 2560 problem-exposures
对比:
  R3: 13 × 2 = 26 exposures (val=0.475)
  R4: 25 × 2 = 50 exposures (val=0.439)
  R8: 80 × 32 = 2560 exposures (val=0.522)
```

## R8 配置演进总结

| 配置 | R3 | R4 | R8 | 效果 |
|------|-----|-----|-----|------|
| lr | 3e-6 | 3e-6 | **2e-6** | 微降, 配合大batch |
| entropy_coeff | 0.001 | 0.001 | **0.003** | 3x, 联动规则生效 |
| use_kl_loss | False | False | **True** | 首次启用! |
| kl_loss_type | — | — | **low_var_kl** | 低方差KL |
| batch_size | 2 | 2 | **32** | 16x! 稳定梯度 |
| epochs | 2 | 2 | **10** | 5x! 充分训练 |
| GPU | 2 | 2 | **4** | 修复GPU利用 |
| FSDP | ? | ? | **fsdp_size=-1** | 自动分片 |
| steps | 13 | 25 | **80** | 6x R3 |
| val-acc | 0.475 | 0.439 | **0.522** | 🏆 最高! |

## 优化建议

R8 已是表现最好的轮次，建议微调而非大改:

| 优先级 | 建议 | 理由 |
|--------|------|------|
| 🟢P2 | entropy_coeff 0.003→0.005 | entropy仍降到0.07, 可能需要更高coeff配合大batch |
| 🟢P2 | 增加max_response_length 512→1024 | 给模型更多推理空间, 可能突破val天花板 |
| ℹ️P3 | 验证 low_var_kl 是否有效 | KL=0.00009极低, 确认KL loss在被正确计算 |

## Patch

### Patch 1 [P2]: rllm-config — 大batch模式识别

R8证明了大batch(≥16)配合高entropy_coeff的有效性。将此模式编码:
- 当 batch_size >= 16 时: entropy_coeff 可维持当前联动规则值
- 大batch本质提供了"平滑收敛"的额外稳定性
