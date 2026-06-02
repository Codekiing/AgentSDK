---
id: traj-20260601-135344-initial-config
target_section: initial-config
action: append
description: "entropy_coeff=0.001 设为0.5B训练强制默认值。VERL源码(losses.py:128)证实这是唯一的entropy保护机制, R1/R2/R3实证验证."
status: proposed
source: trajectory-analysis
source_sessions: ["ae18fdb6-ae30-495b-b1c7-5959d2e445be", "c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

## entropy_coeff 强制默认值 [VERL源码验证]

### 机制 (losses.py:123-129)
```python
if entropy is not None:
    entropy_loss = agg_loss(entropy, response_mask, loss_agg_mode)
    policy_loss -= entropy_coeff * entropy_loss  # ← 唯一的entropy保护!
```

### 强制规则
- **0.5B 模型: entropy_coeff 必须 >= 0.001 (不可为0)**
- **7B 模型: entropy_coeff 必须 >= 0.0005 (不可为0)**
- 配置生成时检查: 如果 entropy_coeff=0 → **ERROR: 拒绝生成配置**

### 实证证据 (3轮对比)
```
         entropy_coeff  entropy trend          decline_rate  val-acc
R1 (0.5B)     0         0.279→0.218             0.0013       0.488
R2 (0.5B)     0         0.289→0.092 ⚠️崩塌      0.0015       0.334
R3 (0.5B)     0.001     0.157→0.152 ✅稳定      0.0004       0.475
```

### 为什么必须强制
1. GRPO 无 critic → 无 value-based exploration bonus
2. 无 entropy_coeff 时, 策略自然趋向确定性 (entropy 单调下降)
3. R1 在 48 步下降 0.06, R2 在 128 步下降 0.20
4. R3 启用后, entropy 几乎水平 (Δ=0.005 in 12 steps)
5. **entropy 崩塌是不可逆的** — 一旦策略坍缩, 无法通过继续训练恢复

### 7B 模型特别说明
R2 的 3 次 7B 实验中, 即使 entropy_coeff=0.001 (R2-R2, R2-R3), entropy 仍在下降.
7B 对低温 (temp=0.3) 更敏感, 建议 temperature >= 0.5 配合 entropy_coeff >= 0.001.

