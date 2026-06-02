---
id: "006-step-event-echo"
target_section: "monitoring-methods"
action: append
description: >-
  要求 rllm-monitor 对 Monitor 工具收到的每个 [MONITOR_STEP] 事件立即逐条回显，
  避免只启动 Monitor 但不自动向用户输出每步状态。
source: "2026-05-27 用户反馈 monitor 不会每个 step 自动输出"
created: "2026-05-27"

depends_on:
  - "003-active-polling-mandate"
  - "005-behavioral-metrics"
conflicts_with: []

status: active
superseded_by: ""
---

### Monitor 事件逐步回显规则

当 Monitor 工具产生 `<task-notification>`，且事件内容包含 `[MONITOR_STEP]` 时，rllm-monitor 必须立即向用户输出每一条 step 摘要，不得等待用户输入、不得只在最终 Training Report 后汇总。

执行要求：
1. **逐条回显**：如果一个 Monitor notification 中包含多条 `[MONITOR_STEP]`，必须按原顺序逐条输出；每条都转换为固定 schema 或原样去掉 `[MONITOR_STEP]` 前缀。
2. **不可静默**：收到 `[MONITOR_STEP]` 后，本轮响应的首要动作就是输出该 step；不要只更新内部状态、不要只记录 task/todo、不要等下一次 tail。
3. **去重但不漏报**：维护 `last_reported_step`。只跳过已经汇报过的 step；如果 notification 批量补发 Step 8-16，而上次只报到 Step 7，则必须输出 Step 8-16 全部。
4. **状态修正**：输出时必须重新计算 `Status` 的长度风险：`Clip >= 0.80` 或 `Len >= max_completion_length * 0.90` 时显示 `WARN(length-limit)`；连续 3 次命中时显示 `STOP(length-limit)`。即使日志原始 Status 是 OK，也要在用户可见输出中修正。
5. **终止事件**：收到 `Training Report` 时，停止 Monitor task，读取日志尾部/最终指标并汇报完成摘要；不要继续等待用户说“继续”。
6. **后台训练完成事件**：收到训练后台 task completed 但 Monitor 尚未输出 Training Report 时，立即读取 training_log.txt，补报所有未汇报的 `[MONITOR_STEP]`，再输出最终摘要。

错误模式（禁止）：
- 只调用 Monitor 工具，然后发送“我会在事件到达时汇报”后停止，但收到事件时不自动回显。
- 把多个 step 合并成一句“训练在进行”，导致用户看不到每步指标。
- 收到后台训练完成通知后不读取日志、不停止 Monitor、不输出最终摘要。
