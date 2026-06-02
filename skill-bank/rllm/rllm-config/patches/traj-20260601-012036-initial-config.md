---
id: traj-20260601-012036-initial-config
target_section: initial-config
action: append
description: 更新0.5B GSM8K推荐num_problems=256（扩大样本量提升泛化）
status: proposed
source: trajectory-analysis
source_sessions: ["4d86a9e2-7985-4779-bdae-5b8e01b3edc8"]
---

### 训练样本数调整建议

0.5B GSM8K 推荐 num_problems=256:
- R1 (64样本): 尽管有ppo_mini_batch问题，仍完成26步训练
- R3 (32样本): 82步稳定但reward上限仅0.1964
- R4 (32样本, lr提升): reward跃升至0.5200

256样本可提供更丰富的训练信号，在lr优化后获得更好泛化效果。0.5B模型训练时间短，256样本成本可控。

