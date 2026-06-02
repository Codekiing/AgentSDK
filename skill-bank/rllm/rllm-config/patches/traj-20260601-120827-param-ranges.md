---
id: traj-20260601-120827-param-ranges
target_section: param-ranges
action: append
description: "新增7B模型参数安全范围: lr=1e-6~3e-6, temperature≥0.5, entropy_coeff≥0.001, num_problems=128-256"
status: proposed
source: trajectory-analysis
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

## 7B 模型参数安全范围

基于Round 2中3次Qwen2.5-7B-Instruct实验 (每次16-96步, 4×A800):

### 学习率
- **安全范围**: 1e-6 ~ 3e-6
- **下限**: 1e-6 (1e-7导致KL=0.001, 模型几乎不更新)
- **上限**: 3e-6 (5e-6导致entropy崩塌到0.038, reward hacking)
- **推荐**: 2e-6 (平衡学习速度和稳定性)

### Temperature
- **下限**: 0.5 (0.3导致7B熵快速塌缩)
- **推荐**: 0.6-0.8 (数学推理场景)
- 注意: 7B模型对低温比0.5B更敏感

### Entropy Coeff
- **必须启用**: entropy_coeff ≥ 0.001
- 7B模型参数空间大, 无约束时更容易坍缩到确定性模式

### Num Problems
- **推荐**: 128-256 (7B容量大于0.5B, 可处理更多问题)
- **上限**: 512 (未验证, 但基于0.5B教训应保守)

### 实验数据
| Run | LR | Temp | Entropy Coeff | Entropy Range | Score | 诊断 |
|-----|-----|------|---------------|---------------|-------|------|
| R2-R1 | 1e-7 | 0.3 | 0.0 | 0.275→0.438 | 0.446 | lr过低 |
| R2-R2 | 1e-6 | 0.3 | 0.001 | 0.289→0.150 | 0.485 | 可工作但慢 |
| R2-R3 | 5e-6 | 0.3 | 0.001 | 0.038→0.072 | 0.961⚠️ | 模式坍缩+reward hacking |

