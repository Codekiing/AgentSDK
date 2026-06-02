---
id: traj-20260601-012036-param-ranges
target_section: param-ranges
action: append
description: 扩展0.5B模型lr安全范围至1e-6~1e-5（R4验证lr=5e-6有效）
status: proposed
source: trajectory-analysis
source_sessions: ["4d86a9e2-7985-4779-bdae-5b8e01b3edc8"]
---

### 0.5B 模型 lr 安全范围更新 (R4 验证)

Round 1 第4轮实验验证: lr=5e-6 在稳定配置下安全且高效。
- R1-R3 (lr=1e-6): 最佳 reward=0.1964
- R4 (lr=5e-6): reward=0.5200, val-acc=0.488

在 ppo_mini_batch_size 正确计算和 max_response_length 安全的前提下，lr 安全上限可扩展至 1e-5。

