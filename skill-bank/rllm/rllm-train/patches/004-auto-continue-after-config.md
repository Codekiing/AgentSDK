---
id: "004-auto-continue-after-config"
target_section: "phase1-5"
action: append
description: >-
  修复 rllm-train 在 rllm-config 生成新配置后停住、不能自动继续 rllm-run/rllm-monitor 的问题。
source: "2026-05-27 用户反馈改完配置之后不能自动继续启动下一轮训练"
created: "2026-05-27"

depends_on:
  - "003-circuit-breaker"
conflicts_with: []

status: active
superseded_by: ""
---

### Phase 自动续跑规则

rllm-train 是全自动编排 skill。每个子 skill 完成后，编排者必须在下一次可执行回合自动进入后续 Phase，不得等待用户输入“继续”。

#### 子 skill 返回后的强制路由

| 刚完成的 Phase | 检测到的完成标志 | 下一步必须自动执行 |
|---|---|---|
| Phase 1 `rllm-clarify` | 输出训练需求摘要 | 调用 `Skill("rllm-config", args="初始配置 | <需求摘要>")` |
| Phase 2 `rllm-config` | 输出/生成 `rllm_train/output/runs/<run_id>/config.json` | 提取 `<run_id>`，调用 `Skill("rllm-run", args="<run_id>")` |
| Phase 3 `rllm-run` | 输出后台 task id 与 training_log.txt | 调用 `Skill("rllm-monitor", args="<run_id>")` |
| Phase 4 `rllm-monitor` 正常完成 | 无 `CIRCUIT_BREAK` 且 Training Report/进程退出 | 调用 `Skill("rllm-analyze", args="<run_id>")` |
| Phase 4 `rllm-monitor` 熔断 | 输出 `CIRCUIT_BREAK` | 按 fix_preset 调用 `Skill("rllm-config", args="调参 | run_id=<run_id> | fix_preset=<fix_preset>")`，随后继续 Phase 3 |
| Phase 5 `rllm-analyze` 未达标 | `analysis.json` 显示未达标且未触发停止条件 | 调用 `Skill("rllm-config", args="调参 | run_id=<run_id> | <调参建议或 analysis.json 路径>")`，随后继续 Phase 3 |
| Phase 5 `rllm-analyze` 达标/停止 | 达到停止条件 | 进入 Phase 6 最终报告 |

#### 执行约束

1. **不得在 Phase 中间给最终答复**：如果下一 Phase 仍需执行，不要输出“配置已生成，是否继续？”或“下一步建议启动训练”。必须直接调用下一个子 skill。
2. **Skill 调用规则仍然有效**：调用 `Skill(...)` 后本轮必须结束；自动续跑发生在子 skill 完成后的下一轮，而不是同一轮连续调用多个 Skill。
3. **run_id 提取失败时才暂停**：只有无法从子 skill 输出或 config 路径中确定 run_id 时，才向用户报告阻塞。
4. **调参后必须启动训练**：任何以 `调参 | run_id=...` 调用 rllm-config 生成的新配置，都必须自动进入 Phase 3 `rllm-run`，不能停在配置摘要。
5. **监控不可省略**：每次 rllm-run 成功后，必须自动进入 rllm-monitor；不能直接进入分析或等待后台任务完成。

错误模式（禁止）：
- rllm-config 生成新配置后回复“配置已生成”，然后等待用户说“继续”。
- 熔断后生成 tuned config，但没有自动调用 rllm-run。
- rllm-run 启动后台训练后没有自动调用 rllm-monitor。
