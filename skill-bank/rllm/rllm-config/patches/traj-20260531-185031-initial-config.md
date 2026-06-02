---
id: traj-20260531-185031-initial-config
target_section: initial-config
action: append
description: 0.5B GSM8K推荐初始配置：基于4轮调参最优实践(R2-R3混合)
status: proposed
source: trajectory-analysis
source_sessions: ["run_1780247306", "run_1780248991", "run_1780249675", "run_1780250271"]
---

### 0.5B GSM8K 推荐初始配置

基于 4 轮 GSM8K GRPO 训练的最优配置基线:

```python
# 0.5B GSM8K 推荐起手配置
gsm8k_0_5b_default = {
    "num_problems": 64,              # R1 64题稳定, R2 32题也够
    "num_epochs": 1,                 # 先单epoch验证稳定性
    "learning_rate": 1e-6,           # R2-4验证稳定; 5e-6会导致崩溃
    "temperature": 0.9,              # 0.85-1.0之间; 0.7太低导致entropy崩溃
    "num_generations": 8,            # R3-4验证: gen=8稳定advantage方差
    "batch_size": 1,                 # 0.5B单卡宽松, 但保持保守
    "max_completion_length": 1536,   # R3-4验证: 截断<14%; 最少1024
    "gradient_accumulation_steps": 4, # 等效batch=4
    "max_agent_steps": 4,            # GSM8K需要多步推理
    "max_grad_norm": 1.0,            # 梯度裁剪防止不稳定
}
```

#### 调参路线图

```
Round 1: 用上面默认配置建立基线
  -> 如果截断率>30%: max_completion_length += 512
  -> 如果Step后期崩溃: lr /= 2
  -> 如果pg_loss~0: num_generations += 2

Round 2+: 根据分析报告调整
  -> 如果Val平台+Train上升: 天花板, 换模型或加tool
  -> 如果clipfrac=0: lr *= 2, 或增加ppo_epochs
  -> 如果稳定: epochs += 1, lr /= 2
```

