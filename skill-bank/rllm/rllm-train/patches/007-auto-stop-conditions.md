---
id: "007-auto-stop-conditions"
target_section: "stop-conditions"
action: replace
description: >-
  将停止条件从"警告，建议停止"改为自动判断并执行，不再等待用户确认。
source: "2026-05-27 全自动重构"
created: "2026-05-27"

depends_on:
  - "006-fully-automatic"
conflicts_with: []
status: active
superseded_by: ""
---

## 停止条件判断

每轮训练结束后（Phase 4 熔断后 或 Phase 5 完成后），编排者自动检查以下条件（按优先级）：

1. **reward_threshold**: 最终 avg reward >= 目标值 → 成功停止，进入 Phase 6
2. **max_rounds**: 已达最大轮次 → 停止，进入 Phase 6（标注未达标）
3. **max_wall_time**: 总耗时超限 → 停止，进入 Phase 6（标注超时）
4. **plateau_rounds**: 连续 N 轮 reward 提升 < 5% → 自动进入 Phase 5（rllm-analyze 内部升级到 rllm-analyze-deep）
5. **reward 下降**: 连续 2 轮 reward 下降 → 自动降低 lr 并继续下一轮
6. **连续熔断**: 连续 2 次熔断快速路径后 reward 仍无改善 → 自动走完整 Phase 5

所有条件自动执行，不输出警告等用户回应。
