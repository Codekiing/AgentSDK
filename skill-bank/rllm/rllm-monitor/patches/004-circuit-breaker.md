---
id: "004-circuit-breaker"
target_section: "anomaly-detection"
action: insert_after
description: "熔断机制: 当异常检测命中时，写精简 analysis.json 并返回 circuit_break 信号给编排者。编排者据此跳过 Phase 5 直接调参。"
source: "2026-05-22 熔断优化，将分析和路由下沉到 monitor"
created: "2026-05-22"

depends_on:
  - "001-early-stopping"
conflicts_with: []

status: active
superseded_by: ""
---

### 熔断机制（Circuit Breaker）

当上述异常检测规则触发时，除了报告给用户，还需执行以下结构化输出流程。

#### 日志格式升级

rllm_train/logger.py 已升级，每 step 日志行包含 6 维指标：

```
Step  Trajs  Reward     Loss   Entropy   GradNorm   Rollout    tok/s      ETA
 1/16    64   0.250   0.8234    1.2045     0.3421    12.3s   150.2    2m30s
```

监控过程中，从每行汇总行提取以下指标（`—` 视为缺失）：
- `avg_reward`: Reward 列
- `loss`: Loss 列
- `entropy`: Entropy 列
- `grad_norm`: GradNorm 列
- `tok/s`: tok/s 列

#### 6 维熔断 Checklist

每次日志更新后，除了已有的异常检测规则，额外逐项检查：

- [ ] **C1 — 策略崩溃**: 最近 3 step avg_reward 全为 0，且之前存在连续 ≥2 step avg_reward > 0
- [ ] **C2 — 训练发散**: loss 连续 5 step 上升（严格单调递增），或出现 NaN/Inf
- [ ] **C3 — 梯度爆炸**: grad_norm > 100，或连续 3 step grad_norm 上升幅度 > 50%/step
- [ ] **C4 — 策略坍缩**: entropy 连续 3 step 下降，且最新值 < 0.01
- [ ] **C5 — 吞吐异常**: tok/s 骤降 > 80%（相对前 3 step 平均值），持续 ≥2 step
- [ ] **C6 — 无效启动**: 已完成 ≥3 step，所有 avg_reward 均 = 0，且 loss 波动 < 1%（max - min < 0.01 * avg）

与已有异常规则的关系：本 checklist 是补充，不替代已有的 early-stopping、loss-zero、format-degradation 等规则。如果已有规则先触发，按已有规则处理。

#### 熔断触发后的操作

当 C1-C6 任一命中时，执行以下步骤：

**Step 1: 中止训练进程**

```bash
# 使用 TaskStop 停止训练后台任务
```

**Step 2: 写精简 analysis.json**

在 `rllm_train/output/runs/<run_id>/` 下创建 `analysis.json`：

```json
{
  "completed": false,
  "abort_reason": "<命中的 check 编号>: <一句话描述>",
  "abort_step": "<当前 step>/<总 step>",
  "metrics_snapshot": {
    "reward_trend": [0.3, 0.35, 0.0, 0.0, 0.0],
    "loss_trend": [0.8, 0.7, 0.9, 1.2, 1.5],
    "entropy_trend": [1.2, 1.1, 0.8, 0.3, 0.01],
    "grad_norm_trend": [0.3, 0.4, 0.6, 1.2, 2.5],
    "tok_s_trend": [150, 148, 145, 30, 10]
  },
  "fix_preset": "<预设修复方案>",
  "skip_full_analysis": true
}
```

`metrics_snapshot` 中的 trend 数组取最近 5-10 step 的值（有多少填多少）。

**Step 3: 输出结构化信号**

向编排者输出以下格式的信号（必须是监控结束前的最后一行输出）：

```
=== CIRCUIT_BREAK ===
circuit_break: true
abort_reason: <C1-C6 编号>: <描述>
fix_preset: <预设方案>
analysis_json: rllm_train/output/runs/<run_id>/analysis.json
=== END_CIRCUIT_BREAK ===
```

#### fix_preset 映射表

| 命中 Check | fix_preset | 含义 |
|---|---|---|
| C1 策略崩溃 | `lr_half` | learning_rate ÷ 2 |
| C2 训练发散 | `lr_half` | learning_rate ÷ 2 |
| C2 (NaN) | `lr_tenth` | learning_rate ÷ 10 |
| C3 梯度爆炸 | `lr_half_grad_clip` | lr ÷ 2 + max_grad_norm=1.0 |
| C4 策略坍缩 | `entropy_bonus` | lr ÷ 2 + temperature + 0.2 |
| C5 吞吐异常 | `retry_same` | 配置不变，重试一次 |
| C6 无效启动 | `diagnose` | 不跳过 Phase 5，走完整分析 |

#### 未命中熔断时

正常完成训练时，输出标准完成报告（见 completion section），**不输出 CIRCUIT_BREAK 信号**。编排者通过"没有 CIRCUIT_BREAK 信号"判断训练正常完成，继续走 Phase 5。
