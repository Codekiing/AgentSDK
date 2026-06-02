---
id: "005-7b-multi-gpu"
target_section: "param-ranges"
action: append
description: "多 GPU 参数缩放规则、GPU 显存估算、7B 多卡配置指导"
source: "2026-05-22 用户反馈: 7B+4×A100 场景下 rllm-config 无法正确生成配置"
created: "2026-05-22"

depends_on: ["traj-20260518-080303-param-ranges"]
conflicts_with: []

status: active
superseded_by: ""
---

### 多 GPU 参数缩放

当硬件为 N 张 GPU 时（从用户输入或硬件信息中提取），参数按以下规则调整：

1. **batch_size 是 per_device 值**: TRL/Transformers 的 `per_device_train_batch_size` 指单卡 batch。N 卡时有效 batch = batch_size × N
2. **TRL 整除约束不变**: 仍按 per_device 值校验 `(batch_size * gradient_accumulation_steps) % num_generations == 0`
3. **显存可分片**: 多卡时可用 FSDP/DeepSpeed 分片模型，每卡显存负载约为总显存 / N
4. **不决定后端**: rllm-config 只负责生成 TrainingConfig JSON 和参数安全校验。多卡训练的后端选择（rllm_train HF / VERL / DeepSpeed）由 rllm-run 根据实际环境决定

GPU 显存估算（单卡，bf16 + LoRA，不含 FSDP 分片）:

| 模型 | 基础占用 | + batch=1,gen=4,len=512 | + batch=2,gen=4,len=512 | 单卡推荐 |
|------|---------|------------------------|------------------------|---------|
| 0.5B | ~2GB | ~4GB | ~6GB | batch=2, gen=4 |
| 1.5B | ~5GB | ~10GB | ~15GB | batch=2, gen=4 |
| 3B | ~10GB | ~18GB | ~28GB | batch=1, gen=4 |
| 7B | ~14GB | ~30GB | ~50GB | batch=1, gen=4 (80GB卡) |

7B 多卡推荐:

| GPU 配置 | batch_size | num_generations | 有效 batch | 显存/卡 |
|---------|-----------|----------------|-----------|--------|
| 1×A100-80G | 1 | 4 | 1 | ~50GB |
| 2×A100-80G | 2 | 4 | 4 | ~30GB |
| 4×A100-80G | 2 | 4 | 8 | ~20GB |
| 4×A100-40G | 1 | 4 | 4 | ~35GB (临界) |
