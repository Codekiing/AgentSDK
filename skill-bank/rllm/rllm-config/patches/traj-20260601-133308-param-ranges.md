---
id: traj-20260601-133308-param-ranges
target_section: param-ranges
action: append
description: 0.5B模型num_problems推荐增加到96 (R3仅13步, 训练不充分), 预期~24步/epoch
status: proposed
source: trajectory-analysis
source_sessions: ["ae18fdb6-ae30-495b-b1c7-5959d2e445be"]
---

## 0.5B num_problems 迭代建议

### 当前状态
R3实证: 48 problems + batch=2 + epochs=2 → **仅13步训练**
- 配置质量很高 (entropy稳定, val不再退化)
- 但训练步数不足, 限制了模型充分学习

### 迭代建议
```yaml
# Round 4 推荐
num_problems: 96     # 2x R3 (仍在安全上限128内)
batch_size: 2        # 保持 (R3验证batch=2稳定)
num_epochs: 2        # 保持
# 预期: ~24步/epoch × 2 = ~48步 (匹配R1的48步)
```

### 备选方案
```yaml
# 如果problems增加不可行, 增大batch
num_problems: 48     # 保持R3
batch_size: 4        # 2x R3
# 预期: ~24步 (但per-step梯度可能不够精细)
```

### 安全边界
- num_problems上限: 128 (基于R2教训: 512有害)
- batch_size上限: 4 (基于R2教训: 8过大)
- 优先增加problems而非batch (更多数据多样性 > 更大梯度batch)

