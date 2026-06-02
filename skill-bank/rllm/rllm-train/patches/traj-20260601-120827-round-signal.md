---
id: traj-20260601-120827-round-signal
target_section: round-signal
action: append
description: 训练结束/失败时强制调用RoundState.write_training_complete(), 确保CLI-2可自动发现训练完成状态
status: proposed
source: trajectory-analysis
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

## 训练状态写入 (强制步骤)

训练结束时必须写入RoundState状态文件, 确保CLI-2可以自动发现并开始优化流程.

### 实现代码
```python
from traj_opt.round_state import RoundState

# 训练完成时
rs = RoundState()
rs.write_training_complete(
    round_num=round_num,        # 从上下文或参数提取
    run_id=run_id,              # 当前训练的run_id
    reward=final_val_acc,       # 使用验证集准确率 (如val-aux/gsm8k/reward/mean@1)
    session_id=session_id,      # 从traj_opt hooks获取
    run_ids=[run_id],           # 本轮所有run (如有多个尝试)
    success=True,
    task_id=f"gsm8k_qwen0.5b_round{round_num}",
)

# 训练失败时
rs.write_training_failed(
    round_num=round_num,
    run_id=run_id,
    error=str(error),
    session_id=session_id,
)
```

### 强制性声明
- 此步骤为训练流程的最后一步, **不可跳过**
- 如未执行, CLI-2将无法自动发现训练完成 → 双CLI协调断裂
- Round 2为第二起已知遗漏案例 (Round 1后同样缺失)

