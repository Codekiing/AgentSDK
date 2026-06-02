---
id: traj-20260601-135344-param-ranges
target_section: param-ranges
action: append
description: 0.5B num_problems 从48增加到96. R3配置健康但仅13步(DataLoader OOM), 需更多步数以充分利用entropy_coeff的稳定性.
status: proposed
source: trajectory-analysis
source_sessions: ["ae18fdb6-ae30-495b-b1c7-5959d2e445be"]
---

## 0.5B num_problems 迭代: 48 → 96

### R3 验证结果
- 48 problems + entropy_coeff=0.001: entropy 稳定 (rate=0.0004)
- 但仅 13 步 (DataLoader OOM 中断) — 训练不充分

### 下一轮推荐
```yaml
num_problems: 96     # 2x R3, 仍在上限128内
batch_size: 2        # 保持 (R3验证稳定)
num_epochs: 2        # 保持
entropy_coeff: 0.001 # 保持 (强制, 已验证)
lr: 3e-6             # 保持 (已验证)
# 预期: 96/2×2 = ~96 batches total, ~48步/epoch
```

### 安全边界 (基于R2教训)
- 上限: 128 (R2: 512 → val崩塌)
- 推荐: 96 (2x R3, 提供充足步数)
- 下限: 48 (R3验证, 但步数不足)

