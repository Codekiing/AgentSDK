---
id: traj-20260518-121744-anomaly-detection
target_section: anomaly-detection
action: append
description: "GRPO loss=0 检测：连续 N 步 loss 为 0 且 num_generations >= 3 时标记为严重异常"
status: proposed
source: trajectory-analysis
source_sessions: ["de0bd17e-d38c-4a96-9c1b-ab5bd4028ed3", "b16e9f00-3909-4d98-86d8-fff1ea282539"]
---

### GRPO loss=0 异常检测

| 异常 | 检测方式 | 处理 |
|---|---|---|
| GRPO loss 持续为 0 | 连续 >= 4 步 loss=0.0 且 num_generations >= 3 | **严重异常**：即使 reward 有波动，loss=0 意味着策略梯度未生效。可能原因：(1) reward 传递方式有误，GRPOTrainer 未正确接收 per-completion reward；(2) rollout_func 返回的 logprobs 不正确；(3) env_mask 导致有效 token 过少。建议停止训练，检查 train.py 中 math_reward_fn 的返回值格式是否与 TRL GRPOTrainer 的期望一致。 |
| Loss=0 但 reward 上升 | loss=0 全程 + epoch 间 reward 有提升 | **警告**：reward 提升可能来自 base model 在不同 prompt 上的表现差异，而非 GRPO 学习效果。需要对比 base model 零样本 reward 和训练后 reward 来确认是否有真实提升。 |
