---
id: 002-deep-analysis-escalation
target_section: suggestions
action: replace
description: >-
  rllm-analyze 只负责常规分析。当检测到调参无法解决问题时，自动调用 rllm-analyze-deep
  做 15 层分层诊断。rllm-analyze-deep 全自动执行，输出 deep_analysis.json。
  编排者不需要知道升级逻辑，只看到 rllm-analyze 返回的最终结论。
status: active
source: "2026-05-27 链路重构: 将深层诊断从编排层移至 rllm-analyze 内部"
---

### 深层诊断自动升级

`rllm-analyze` 只负责常规训练结果分析和调参建议。当出现以下任一条件时，rllm-analyze 不得输出纯参数建议，必须自动调用 `Skill("rllm-analyze-deep", args="<run_id>")` 做深层诊断：

| 条件 | 现象 | rllm-analyze-deep 动作 |
|------|------|----------------------|
| E1 — 连续 plateau | 连续 2 轮 reward 无改善（提升 < 5%） | 全自动分层排查：指标→轨迹→reward→数据 |
| E2 — 行为指标异常 | finish_rate / answer_coverage / tool_usage_rate 持续异常且调参无效 | 进入轨迹与工程异常层排查 |
| E3 — 学习信号消失 | reward_variance 长期接近 0 且 temperature/num_generations 调整无效 | 检查 reward 区分度、数据难度 |
| E4 — 疑似 reward hacking | train reward 上升但实际正确率不涨 | 进入 reward 设计与验证一致性排查 |
| E5 — 零分/满分过多 | pass@k 接近 0、同组 reward 全 0、或 reward 全 1 | 进入数据难度与 curriculum 排查 |

#### 升级执行步骤

1. rllm-analyze 完成常规分析，写入 analysis.json
2. 检查 E1-E5，如果任意触发：
   - 调用 `Skill("rllm-analyze-deep", args="<run_id>")`
   - rllm-analyze-deep 自动读取所有数据，按 15 层诊断体系排查
   - rllm-analyze-deep 输出 deep_analysis.json
3. 读取 deep_analysis.json，将其结论合并到 analysis.json 的 suggestions 中
4. 在分析报告中标注 "已通过 rllm-analyze-deep 做深层诊断"

#### 升级后的输出

analysis.json 增加字段：
```json
{
  "deep_analysis_triggered": true,
  "deep_analysis": {
    "root_cause_layer": "trajectory_engineering",
    "action": "tune" | "engineering_fix" | "data_or_reward_fix",
    "summary": "<一句话根因>"
  }
}
```

- `action = "tune"`: 返回调参建议，编排者正常回到 Phase 2
- `action = "engineering_fix"`: 输出代码修复方案，编排者自动应用后继续
- `action = "data_or_reward_fix"`: 输出数据/reward 修改方案，编排者自动应用后继续

#### 禁止行为

- 触发 E1-E5 后仍然只给纯参数建议而不调用 rllm-analyze-deep
- 调用 rllm-analyze-deep 后不将其结论写入 analysis.json
- 跳过 rllm-analyze-deep 直接输出"建议参考专家文档"
