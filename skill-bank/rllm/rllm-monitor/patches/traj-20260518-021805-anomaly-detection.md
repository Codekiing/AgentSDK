---
id: traj-20260518-021805-anomaly-detection
target_section: anomaly-detection
action: append
description: 7B 模型 GPU 训练的额外异常检测规则（来自 Round 1 轨迹分析）
status: proposed
source: trajectory-analysis
source_sessions: ["98366057-ccbd-43bc-8dc2-022d649d4a3f"]
---

### 7B 模型 GPU 训练异常检测（A100 80GB）

| 异常 | 检测方式 | 处理 |
|---|---|---|
| Response 长度撞限 | avg_response_len == max_completion_length 且 reward=0 | 报告，增大 max_completion_length 或检查模型是否陷入重复生成 |
| Grad norm 过高 | grad_norm > 100 | 报告，建议降低 lr 或添加 max_grad_norm clipping |
| Entropy 快速下降 | 4 步内 entropy 下降 > 70% | 警告过早收敛，建议提高 temperature |
| Reward 震荡 | 连续 4 步 reward 标准差 > 0.15 | 报告训练不稳定，检查 lr 和 batch_size |
| 单卡 OOM | CUDA out of memory + 模型 > 3B | 建议减小 batch_size/num_generations 或使用多卡 FSDP |
| 训练时间异常长 | 每 step rollout > 60s（7B 模型） | 检查是否在 CPU 上 fallback
