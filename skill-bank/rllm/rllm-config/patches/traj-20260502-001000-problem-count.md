---
id: traj-20260502-001000-problem-count
target_section: param-ranges
action: append
description: "0.5B + mixed 难度时 num_problems 上限从 64 降到 32，防止 forgetting"
status: proposed
source: trajectory-analysis
source_sessions: ["run_1777651915"]
---

当使用内置合成数据且 difficulty=mixed 时，0.5B 模型的 num_problems 安全上限进一步收紧:

| 参数 | 原上限 | 新上限 | 条件 | 依据 |
|------|--------|--------|------|------|
| num_problems | 64 | 32 | dataset_path 为空、difficulty=mixed 且 model=0.5B | 合成 mixed 数据历史 run 在 64 problems 下出现 forgetting |

推荐配置（仅 0.5B + 合成 mixed）:
- num_problems=32, lr=5e-6, epochs=1, batch=2, generations=4
- 预期: 32 步训练，reward 稳定不崩溃

替代方案（仅合成数据）: 保持 64 problems 但切换 difficulty=simple
- 适用于需要更多合成训练数据但不需要 hard 题目的场景

外部数据集规则:
- 当 dataset_path 非空时，禁止应用本节 num_problems 上限和 simple/mixed/hard 切换建议。
