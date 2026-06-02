---
id: traj-20260602-081552-extend-training
target_section: param-ranges
action: replace
description: [P0] 延长 num_epochs 支持到 20, 配套上调 entropy_coeff 联动规则 — R9 证据显示训练结束时 reward 仍在上升
status: accepted
source: trajectory-analysis
source_sessions: ["743765dc-6160-4b38-9ebc-fb46ff27a8ef"]
priority: P0
---

## param-ranges 更新: num_epochs & entropy_coeff

### 轨迹证据 (Round 9)

- **session**: 743765dc-6160-4b38-9ebc-fb46ff27a8ef
- **配置**: 256 problems, batch=32, epochs=10, entropy_coeff=0.003, temp=0.7
- **epoch trend** (reward 每 epoch 上升, 未 plateau):
  - E0: 0.512, E1: 0.612, E2: 0.675, E3: 0.692, E4: 0.730, E5: 0.751, E6: 0.755, E7: 0.778
- **last_5_avg**: 0.806 vs global_avg: 0.707 (14% gap → 训练结束过早)
- **entropy**: 0.121→0.084, 下降速率 0.0041/epoch, 0.00046/step

### 更新 num_epochs 上限

| 参数 | 旧 max | 新 max | 理由 |
|------|--------|--------|------|
| num_epochs | 20 (理论) | 20 (已验证上限) | 0.5B + 256 problems需≥15 epochs |

### 新增联动规则: entropy_coeff ∝ num_epochs

当 num_epochs >= 15 时, entropy_coeff 不能维持原值, 需按训练长度缩放:

```
effective_coeff = base_coeff × (num_epochs / 10)
上限: 0.01
```

**R10 推荐配置**: num_epochs=20, entropy_coeff=0.005

### 更新 num_generations 上限

| 参数 | 旧 max | 新 max | 理由 |
|------|--------|--------|------|
| num_generations | 8 | 12 | 0.5B 小模型变异性低, 需更多采样 → 更准确 advantage |

R9 证据: advantage_abs_avg=0.0152 (极低, 说明 group 内 variability 不足)
