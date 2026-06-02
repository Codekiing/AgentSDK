---
id: traj-20260502-213850-problem-count-refine
target_section: param-ranges
action: append
description: "收紧 0.5B+mixed num_problems 最优范围: 36-42 (48 后期 forgetting)"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### num_problems 最优范围精细化（仅限 0.5B + 合成 mixed 数据）

以下经验只适用于 `dataset_path` 为空、使用内置合成 mixed 数据的 0.5B 训练；不得用于 DeepScaler 或其他外部数据集。

| num_problems | 结果 | 证据 |
|-------------|------|------|
| 32 | 无 forgetting 但 loss=0 (无学习) | 历史合成数据 run：avg≈0.77, loss=0 全程 |
| 48 | avg≈0.85 但后期格式退化 | 历史合成数据 run：后 20% reward 明显下降 |
| 40 (推断) | 平衡点 | 32 太简单, 48 后期崩溃 |

更新推荐（仅合成 mixed）:
- 首轮训练: num_problems=40 (安全起点)
- avg_reward >= 0.85 且无后期 forgetting: 可尝试 44
- 出现后期 forgetting: 减少到 36
- 禁止 0.5B+mixed 合成数据使用 num_problems > 48
