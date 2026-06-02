---
id: "006-fully-automatic"
target_section: "execution-modes"
action: replace
description: >-
  移除所有暂停等用户行为，改为全自动自循环。统一路由描述，修正 fix_preset=diagnose 走向。
  rllm-analyze 内部会自动升级到 rllm-analyze-deep，编排者无需关心诊断层级。
source: "2026-05-27 全自动重构: 移除暂停、统一路由、修复 diagnose 死循环"
created: "2026-05-27"

depends_on:
  - "005-stateful-orchestration"
conflicts_with: []
status: active
superseded_by: ""
---

## 执行模式

**全自动执行，绝不暂停等待用户。** 编排者在每个 Phase 完成后直接进入下一个 Phase。

即使以下情况也不暂停，自动处理：
- 训练出错 → 自动 error recovery（见下方）
- 连续 N 轮 reward 无改善 → rllm-analyze 自动升级到 rllm-analyze-deep，全自动深层诊断
- 达到停止条件 → 进入 Phase 6 输出最终报告

禁止输出"是否继续?"、"请确认"、"建议下一步"等等待用户回复的内容。
