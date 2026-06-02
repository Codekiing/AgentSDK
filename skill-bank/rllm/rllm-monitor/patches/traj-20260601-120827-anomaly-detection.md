---
id: traj-20260601-120827-anomaly-detection
target_section: anomaly-detection
action: append
description: "新增PPO_KL=0静默告警: KL持续为0超过10步时WARN, 配合entropy崩塌时升级为ALERT"
status: proposed
source: trajectory-analysis
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

## PPO_KL=0 分层告警

### 检测规则
```python
kl_values = []  # 最近N步的KL值

def check_kl_zero(kl_values, entropy_current, threshold_steps=10):
    if len(kl_values) < threshold_steps:
        return None  # 数据不足
    recent = kl_values[-threshold_steps:]
    if all(abs(kl) < 1e-6 for kl in recent):
        if entropy_current is None:
            return ("WARN", "PPO_KL持续为0超过10步. 检查ref_path和use_kl_loss配置.")
        elif entropy_current < 0.10:
            return ("ALERT", f"PPO_KL=0 + entropy={entropy_current:.3f}(<0.10). 高风险模式坍缩! 建议立即增大entropy_coeff.")
        elif entropy_current < 0.15:
            return ("WARN", f"PPO_KL=0 + entropy={entropy_current:.3f}(<0.15). 建议修复KL约束 + 增大entropy_coeff.")
        else:
            return ("INFO", "PPO_KL=0但entropy正常. KL=0不一定致命(Round 1实证: val=0.488).")
```

### 严重等级
- KL=0 alone: **INFO** (Round 1证明可正常工作)
- KL=0 + entropy < 0.15: **WARN** (无KL约束 + 探索枯竭)
- KL=0 + entropy < 0.10: **ALERT** (高风险模式坍缩)

### 参考
- 所有5个run (Round 1×1 + Round 2×4)的KL均≈0
- 诊断参考框架: ppo_kl健康范围 0.001-0.05

