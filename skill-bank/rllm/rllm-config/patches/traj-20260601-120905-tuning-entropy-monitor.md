---
id: traj-20260601-120905-tuning-entropy-monitor
target_section: tuning
action: append
description: "新增entropy监控规则: entropy_coeff=0时WARN; entropy<0.15时WARN; entropy<0.1时紧急停止建议"
status: proposed
source: trajectory-analysis
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

## Entropy崩塌监控规则

### 配置验证阶段
- 如果 `entropy_coeff` 未配置或等于 0:
  → **WARN**: "entropy_coeff=0, entropy可能在训练中崩塌. 建议启用entropy_coeff≥0.001."
  → 引用: Round 2 entropy_coeff=0 → entropy 0.289→0.092

### 训练监控阈值
- `actor/entropy` < 0.15:
  → **WARN**: "Entropy低于0.15警戒线, 探索空间收窄. 检查temperature和entropy_coeff配置."
- `actor/entropy` < 0.10:
  → **EMERGENCY**: "Entropy低于0.10危险阈值, 策略正在坍缩为确定性输出. 建议: 立即停止训练, 增大entropy_coeff(>0.001), 提升temperature(>0.85)."
- `actor/entropy` < 0.05:
  → **CRITICAL**: "Entropy极低, 模型已坍缩. 停止训练, 回滚checkpoint, 大幅调整配置."

### 参考基线
- Round 1 (健康): entropy 0.279→0.218 (48步, Δ=-0.06, 下降率0.0013/step)
- Round 2 (崩塌): entropy 0.289→0.092 (128步, Δ=-0.20, 下降率0.0016/step)
- 崩塌临界值: entropy < 0.10 (诊断参考框架)
