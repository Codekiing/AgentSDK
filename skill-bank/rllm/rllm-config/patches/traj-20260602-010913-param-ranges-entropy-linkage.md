---
id: traj-20260602-010913-param-ranges-entropy-linkage
target_section: param-ranges
action: append
description: "entropy_coeff 必须与 num_problems 联动: problems增加→coeff相应增加. R3(48p,coeff=0.001→稳定) vs R4(96p,coeff=0.001→振荡). VERL losses.py:128有效探索压力密度=coeff×contact_frequency."
status: proposed
source: trajectory-analysis
source_sessions: ["ae18fdb6-ae30-495b-b1c7-5959d2e445be"]
---

## entropy_coeff 与 num_problems 联动规则 [VERL源码验证]

### 机制 (losses.py:128)
```python
policy_loss -= entropy_coeff * entropy_loss  # 每次step的探索压力
```

### 关键发现: 探索压力密度
entropy_coeff的绝对值不是关键 — **有效探索压力密度**才是:
```
effective_pressure = entropy_coeff × contact_frequency
contact_frequency = batch_size / num_problems
```

### 实证证据 (R3 vs R4, 0.5B模型)
| 轮次 | problems | coeff | contact_freq | effective_pressure | entropy | val-acc |
|------|----------|-------|-------------|-------------------|---------|---------|
| R3 | 48 | 0.001 | 2/48=0.042 | 4.2e-5 | ✅稳定 | 0.475 |
| R4 | 96 | 0.001 | 2/96=0.021 | 2.1e-5 | ⚠️振荡50%<0.1 | 0.439↓ |

### 联动规则
```yaml
# 0.5B 模型 entropy_coeff 推荐值
num_problems <= 48:  entropy_coeff = 0.001   # R3验证有效
num_problems 49-96:  entropy_coeff = 0.002   # 2x R4当前值, 恢复R3的有效密度
num_problems 97-128: entropy_coeff = 0.003   # 3x, 需验证
```

### 设计理由
R4的paradox (train score↑0.553→0.597但val↓0.475→0.439)证实:
- 相同的coeff在更多problems上产生的探索压力不足
- entropy在低值振荡→低entropy step中模型学到了"窄模式"
- 窄模式在train data上有效(train score↑)但在val data上不泛化(val↓)
- 这不是coeff太低的问题 — 是coeff与problems未联动

### 实现
config生成时:
```python
if num_problems <= 48:
    entropy_coeff = 0.001
elif num_problems <= 96:
    entropy_coeff = 0.002
else:
    entropy_coeff = 0.003
# 警告: entropy_coeff > 0.005 可能过度鼓励随机探索
```
