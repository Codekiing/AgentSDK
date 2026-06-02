---
id: traj-20260601-040729-param-ranges
target_section: param-ranges
action: append
description: "0.5B模型num_problems安全上限设为64（Round2:512导致PPO_KL=0）"
status: proposed
source: trajectory-analysis
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

### 0.5B num_problems 安全上限收紧 (Round 2 验证)

Round 2 num_problems=512 导致严重退化:
- PPO_KL=0 (policy未更新)
- val-acc=0.3343 (vs Round 1 0.488)
- Reward趋势下降

**0.5B 模型 num_problems 安全范围: 32 ~ 64**
推荐: 32-64问题 + 2-3 epochs深度训练
