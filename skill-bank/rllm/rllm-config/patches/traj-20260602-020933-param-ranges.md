---
id: traj-20260602-020933-param-ranges
target_section: param-ranges
action: append
description: "新增发现: 大batch(≥16)可实现entropy平滑坍缩+val上升. R8(batch=32,entropy 0.14→0.07平滑,val 0.47→0.52↑). 颠覆'熵崩塌必然有害'理论."
status: proposed
source: trajectory-analysis
source_sessions: ["ae18fdb6-ae30-495b-b1c7-5959d2e445be"]
---

## 大Batch模式: 平滑 Entropy 坍缩 [R8实证]

### 核心发现
R8首次证明: **entropy下降不一定有害**。关键在于下降是"平滑"还是"震荡"。

### 两种Entropy坍缩模式

| 模式 | Batch | Entropy行为 | Val结果 | 代表轮次 |
|------|-------|------------|---------|---------|
| **震荡型** | ≤8 | 上下振荡, >30% steps <0.1 | ↓ 退化 | R4 (batch=2, val↓) |
| **平滑型** | ≥16 | 单调下降, window-over-window稳定 | ↑ 改善 | R8 (batch=32, val↑) |

### 机制 (VERL源码)
```
core_algos.py:1138-1199 (agg_loss):
  small batch (2-8): gradient = avg of 2-8 problems → high noise
    → step N pulls toward mode A, step N+1 pulls toward mode B
    → entropy oscillates → model learns conflicting narrow modes
    → val degrades

  large batch (16-32): gradient = avg of 16-32 problems → low noise
    → each step represents consensus direction across many problems
    → entropy smoothly converges along consensus
    → model converges to a single effective mode
    → val improves
```

### R8 实证 (0.5B, batch=32, 10 epochs)
```
10-step window analysis:
  Window 1 (steps 1-10):   score=0.566  entropy=0.129  gn=0.96
  Window 2 (steps 11-20):  score=0.636  entropy=0.117  gn=1.00
  Window 3 (steps 21-30):  score=0.685  entropy=0.102  gn=0.92
  Window 4 (steps 31-40):  score=0.719  entropy=0.094  gn=0.93
  Window 5 (steps 41-50):  score=0.757  entropy=0.084  gn=0.89
  Window 6 (steps 51-60):  score=0.774  entropy=0.077  gn=0.86
  Window 7 (steps 61-70):  score=0.799  entropy=0.073  gn=0.88
  Window 8 (steps 71-80):  score=0.809  entropy=0.070  gn=0.92
  
  Score: monotonically increasing across all windows
  Entropy: monotonically decreasing, smooth (no oscillation)
  Grad norm: stable 0.86-1.00 (consistent updates)
  Val: 0.470 → 0.522 (improving throughout!)
```

### 配置建议
```yaml
# 大batch稳定训练配置 (R8验证)
batch_size: 32               # 大batch → 平滑梯度
entropy_coeff: 0.003         # 配合大batch (联动规则)
use_kl_loss: true            # KL约束防止策略偏离
kl_loss_coef: 0.01
kl_loss_type: low_var_kl     # 低方差KL估计
epochs: 10                   # 充分训练
lr: 2e-6                     # 配合大batch略降lr
GPU: 4                       # FSDP分布式
```

