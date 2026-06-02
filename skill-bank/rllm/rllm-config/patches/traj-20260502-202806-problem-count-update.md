---
id: traj-20260502-202806-problem-count-update
target_section: param-ranges
action: append
description: "更新 0.5B 模型 num_problems 安全范围: 32-48 for mixed"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### num_problems 精细化范围（仅限 0.5B + 合成数据）

基于历史合成数据训练更新推荐范围；只在 `dataset_path` 为空时使用:

| difficulty | 推荐范围 | 依据 |
|-----------|---------|------|
| mixed (20% hard) | 40-48 | 32 太简单 (loss=0), 64 forgetting |
| mixed-hard (50% hard) | 24-32 | hard 比例增加后需减少总量 |
| hard | 16-24 | 64 完全超出能力 |

默认推荐配置（仅 0.5B + 合成正式训练）:
- num_problems=48, difficulty=mixed, lr=5e-6, epochs=1
- 预期: 比 32 problems 更有挑战性，但不会 forgetting

外部数据集规则:
- DeepScaler/自定义 Dataset 不应用这些范围。
- 外部数据集的 num_problems 只依据当前 run 的耗时、OOM、reward variance、plateau、calculator_error 等实际分析结果调整。
