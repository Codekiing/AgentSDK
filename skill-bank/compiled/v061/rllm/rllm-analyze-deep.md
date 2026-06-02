---
name: rllm-analyze-deep
description: Deep 15-layer training analysis for rllm/VERL GRPO training. Run by deep_analyzer.py.
metadata:
  version: "1.1.0"
---

# rllm-analyze-deep — 深度训练分析

## 执行方式

**必须使用** `deep_analyzer.py` 脚本。禁止手写内联分析。

```bash
python skill-bank/rllm/rllm-analyze-deep/deep_analyzer.py <run_dir> --target <reward_threshold>
```

## 输入

| 模式 | args |
|------|------|
| 正常 | `<run_id>` |
| 熔断 | `<run_id> | circuit_break=true | fix_preset=<x> | abort_reason=<reason>` |

## 输出

`rllm_train/output/runs/<run_id>/deep_analysis.json`

```json
{
  "reward_avg": 0.32,
  "n_steps": 32,
  "layers": [
    {"id": 1, "name": "Reward/Score Trends", "status": "WARN", "alerts": ["DIFFICULTY_DRIVEN_OSCILLATION"]},
    {"id": 2, "name": "KL / Reward Gap", "status": "ALERT", "alerts": ["POLICY_NOT_UPDATING"]},
    ...
    {"id": 15, "name": "Synthesis", "status": "ALERT", "samples": [<tuning_suggestions>]}
  ],
  "tuning_suggestions": [
    {"priority": 1, "param": "learning_rate", "direction": "INCREASE", "reason": "..."}
  ]
}
```

## 诊断覆盖

`deep_analyzer.py` 强制执行 15 层诊断，不可跳过：

L1 Reward 趋势 | L2 KL 压制 | L3 GRPO 方差 | L4 PPO 更新强度 | L5 Critic
L6 Entropy 探索 | L7 截断/空响应 | L8 吞吐 | L9 VRAM/OOM | L10 Validation
L11 Reward 函数 | L12 轨迹样本 | L13 数据难度 | L14 交叉相关性 | L15 综合建议

每层的 findings/alerts/samples 都写入 `deep_analysis.json`。
编排者读取 `tuning_suggestions` 传给 rllm-config，不做二次分析。

## 熔断模式

当 `circuit_break=true` 时，脚本会：
1. 验证 fix_preset 是否对症（检查实际 temp/gen/lr 配置）
2. 差异化诊断 S2 根因（温度 vs lr vs ppo_epochs）
3. 必要时覆盖 fix_preset 建议
4. 同样输出完整的 deep_analysis.json
