---
id: traj-20260601-120827-initial-config
target_section: initial-config
action: append
description: "推荐默认配置回归验证有效的保守参数: lr=3e-6, num_problems=48, epochs=2, batch=4, gen=8, entropy_coeff=0.001"
status: proposed
source: trajectory-analysis
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

## 0.5B 模型推荐默认配置

基于Round 1和Round 2的实证数据, 0.5B模型的最佳保守配置:

```yaml
# 已验证有效的保守基线 (Round 1: val-acc=0.488)
model: Qwen2.5-0.5B-Instruct
learning_rate: 3e-6        # R1=5e-6有效但entropy下降偏快, 降为3e-6
num_problems: 48            # R1=32有效, R2=512有害, 48为折中
num_epochs: 2               # 多epoch精炼 > 单epoch海量数据
batch_size: 4               # R2=8导致等效batch过大
num_generations: 8          # GRPO最小group size=4, 8提供充足variance
temperature: 0.85           # R1/R2一致, 对0.5B数学推理有效
entropy_coeff: 0.001        # R2未启用导致entropy崩塌, 必须显式启用
max_response_length: 1536   # 足够的推理空间
gradient_checkpointing: false # 0.5B不需要, 增加开销无收益
```

### 关键约束
- **entropy_coeff必须>0**: Round 2实证entropy_coeff=0.0时entropy从0.29崩塌到0.092
- **lr上限**: 5e-6对0.5B偏高 (entropy下降率0.0016/step), 3e-6更安全
- **epochs≥2**: 单epoch = shallow contact → val divergence

