---
id: traj-20260601-133308-initial-config
target_section: initial-config
action: append
description: "固化R3验证成功的0.5B配置基线: lr=3e-6, entropy_coeff=0.001, batch=2, problems=48, temp=0.7 (R2优化生效实证)"
status: proposed
source: trajectory-analysis
source_sessions: ["ae18fdb6-ae30-495b-b1c7-5959d2e445be", "c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

## R3 验证安全基线 (2026-06-01)

基于Round 3实证 (session: ae18fdb6), 以下配置被验证为0.5B模型的安全有效基线:

```yaml
# 验证安全基线 (R3: val-acc=0.475, entropy稳定)
model: Qwen2.5-0.5B-Instruct
learning_rate: 3e-6          # R2优化推荐, R3验证有效
num_problems: 48             # R2优化推荐上限64, R3用48安全
num_epochs: 2                # 多epoch精炼
batch_size: 2                # 保守, per-step stability优先
num_generations: 8           # GRPO group size
temperature: 0.7             # 配合entropy_coeff=0.001
entropy_coeff: 0.001         # R2关键发现: 必须启用
max_response_length: 1536
max_prompt_length: 512
gradient_checkpointing: false
backend: verl
```

### 实证指标
| 指标 | R1 (baseline) | R2 (degraded) | R3 (this config) |
|------|--------------|---------------|------------------|
| Val-acc | 0.488 | 0.334 | **0.475** |
| Entropy stability | 0.279→0.218 | 0.289→0.092 | **0.157→0.152** |
| Entropy decline rate | 0.0013 | 0.0015 | **0.0004** |
| PPO_KL | 0.0 | 0.0 | **0.000433** |
| Clipfrac | 0.0 | 0.0 | **0.0053** |
| Grad norm | 2.46 | 1.60 | **2.41** |
| Train score mean | 0.520 | 0.464 | **0.553** |

### 与R2优化建议的对应
- R2 Patch 1 (0.5B上限64): ✅ problems=48 生效
- R2 Patch 2 (lr=3e-6, entropy_coeff=0.001): ✅ 直接采纳, 效果显著
- R2 Patch 3 (entropy监控): ✅ entropy稳定在0.15附近

### 已知限制
- 训练步数过短 (仅13步): 建议下一轮增加num_problems到96
- 基础设施偶发OOM (DataLoader worker, 非GPU OOM)

