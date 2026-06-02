---
id: traj-20260602-081552-long-training-strategy
target_section: tuning
action: append
description: [P1] 新增长训练调参策略 — 当 epochs≥15 时的参数联动规则和 temperature 调整
status: accepted
source: trajectory-analysis
source_sessions: ["743765dc-6160-4b38-9ebc-fb46ff27a8ef"]
priority: P1
---

## 长训练调参策略 (num_epochs >= 15)

### 轨迹证据 (Round 9, session 743765dc-6160-4b38-9ebc-fb46ff27a8ef)

- R9 在 epochs=10 时, entropy 从 0.121 线性下降到 0.084 (=epoch 9)
- 下降速率: 0.0041/epoch。若 epochs=20 线性外推: entropy→0.039 (危险区, <0.1)
- 对比 R4: epochs=2, entropy 振荡 (短训练不会有熵触底风险)
- 结论: 长训练需要**主动**的 entropy 调控, 不是被动等待

### 长训练参数联动表

| num_epochs | temperature 推荐 | entropy_coeff 推荐 | num_generations | 说明 |
|-----------|-----------------|-------------------|-----------------|------|
| 1-5 | 0.3-0.5 | 0.001 | 4-6 | 短训练, 低风险 |
| 5-10 | 0.5-0.7 | 0.001-0.003 | 6-8 | 中训练 |
| 10-15 | 0.7-0.85 | 0.003-0.005 | 8-10 | 长训练, 需控制熵 |
| 15-20 | 0.85-1.0 | 0.005-0.008 | 10-12 | 超长训练, 积极防坍缩 |

### 新增问题诊断

- entropy 连续 3 epoch 下降且当前值 < 0.08 → 建议增大 entropy_coeff 1.5x
- reward 连续 2 epoch 不涨 → 建议增大 temperature + 增大 num_generations
- 训练结束时 last_5_avg_reward > global_avg × 1.1 → 训练提前终止, 下次 +5 epochs

### 有效 seed 缓存

R9 证明 seed=48 在 256 problems + batch=32 下产生了良好的训练动态 (reward=0.8003)。
当沿用相同配置时, 优先复用已验证的 seed。
