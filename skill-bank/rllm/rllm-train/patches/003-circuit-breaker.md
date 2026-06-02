---
id: "003-circuit-breaker"
target_section: "phase1-5"
action: append
description: "编排者熔断路由逻辑。熔断检测和 analysis.json 写入由 rllm-monitor (004-circuit-breaker) 负责，编排者只读信号做路由。"
source: "2026-05-22 熔断优化，分析逻辑下沉到 monitor"
created: "2026-05-22"

depends_on:
  - "001-phase4-abort"
  - "rllm-monitor:004-circuit-breaker"
conflicts_with: []

status: active
superseded_by: ""
---

### 熔断快速路径补充说明

base.md 已包含 Phase 4 后的路由决策核心逻辑。以下补充边界情况的处理规则。

#### CIRCUIT_BREAK 信号格式

Monitor 输出格式（编排者只读这个，不做任何分析）:

```
=== CIRCUIT_BREAK ===
circuit_break: true
abort_reason: <check 编号>: <描述>
fix_preset: <预设方案>
analysis_json: rllm_train/output/runs/<run_id>/analysis.json
=== END_CIRCUIT_BREAK ===
```

编排者提取 `fix_preset` 值，传给 Phase 2 的 rllm-config。不需要理解 fix_preset 的含义。

#### fix_preset 到 rllm-config args 的映射

编排者只需要做以下字符串拼接，不需要理解修复逻辑:

| fix_preset 值 | 传给 rllm-config 的 args |
|---|---|
| `lr_half` | `"调参 \| run_id=<id> \| fix_preset=lr_half"` |
| `lr_tenth` | `"调参 \| run_id=<id> \| fix_preset=lr_tenth"` |
| `lr_half_grad_clip` | `"调参 \| run_id=<id> \| fix_preset=lr_half_grad_clip"` |
| `entropy_bonus` | `"调参 \| run_id=<id> \| fix_preset=entropy_bonus"` |
| `increase_max_completion_length` | `"调参 \| run_id=<id> \| fix_preset=increase_max_completion_length"` |
| `retry_same` | `"调参 \| run_id=<id> \| fix_preset=retry_same"` |
| `diagnose` | 不走快速路径，走正常 Phase 5 |

#### 编排者维护 circuit_break_count

每次命中 CIRCUIT_BREAK 后:
1. `training_state.json` 中 `circuit_break_count` +1
2. 如果 `circuit_break_count >= 2` 且最新一轮 reward 无提升 → 重置为 0，走完整 Phase 5
3. 正常完成一轮训练（Phase 5 后 reward 有提升）→ 重置为 0

#### 熔断时的 Heartbeat

熔断触发后，编排者写入 heartbeat 标记熔断状态（仅双 CLI 模式）:

```python
RoundState().write_heartbeat(round_num, run_id, phase="circuit_break",
                             message=f"熔断: {abort_reason}, fix: {fix_preset}")
```
