# rllm-train 轨迹分析报告 (Round 1)

生成时间: 2026-05-18T02:16:54.560113+00:00
分析范围: Round 1
模型: Qwen2.5-7B (7.6B params)
Run ID: run_1779070223

## 训练执行概览

| 配置参数 | 值 |
|---------|-----|
| Model | Qwen2.5-7B (7.6B) |
| Problems | 16 (easy, seed=42) |
| Epochs | 1 |
| Batch size | 1 |
| Generations | 2 |
| Learning rate | 5e-6 |
| Max agent steps | 2 |
| Max completion length | 192 |
| Gradient accumulation | 8 |

## 训练动态

| Step | Reward | Rollout Time | tok/s | avg_response_len |
|------|--------|-------------|-------|-----------------|
| 1/4  | 0.250  | 13.3s       | 84.2  | 45.6            |
| 2/4  | 0.000  | 38.6s       | 298.5 | 192.0 (hit max) |
| 3/4  | 0.375  | 20.3s       | 171.3 | 102.4           |
| 4/4  | 0.000  | 38.6s       | 321.4 | 192.0 (hit max) |

## 问题发现

### 1. Reward 剧烈震荡 (高严重性) [影响: rllm-config, rllm-train]
**现象**: Reward 在 [0.0, 0.375] 之间剧烈震荡，Step 1=0.25, Step 2=0.0, Step 3=0.375, Step 4=0.0
**证据**: perf_stats.json per_step_rollouts rewards 数据
**分析**: 
- Step 2 和 Step 4 的 avg_response_len 均为 192.0，恰好等于 max_completion_length 上限
- 这意味着模型在这两步生成了大量无效 token（达到长度上限被截断），导致无法产生有效答案
- Step 1 和 Step 3 的 avg_response_len 分别为 45.6 和 102.4，在这些步模型能正常完成任务
- **长度与 reward 强相关**: 长 response = 0 reward, 短 response = 有 reward
**建议**: 增大 max_completion_length 或添加 length penalty；同时降低 temperature 减少 exploration noise

### 2. Grad norm 异常高 (中等严重性) [影响: rllm-config]
**现象**: Grad norm 范围 0~145，平均 71.25，极不稳定
**证据**: Training Report 显示 Grad norm avg 71.25, max 145.0
**分析**:
- 对于 7B 模型，grad norm > 100 通常表示训练不稳定
- batch_size=1 + gradient_accumulation=8 意味着每个 micro-batch 只有 2 条轨迹，梯度方差很大
- lr=5e-6 对 7B 可能偏高
**建议**: 增大 batch_size（需要多卡），或降低 lr 到 1e-6~2e-6，或添加 max_grad_norm clipping

### 3. Entropy 快速下降 (中等严重性) [影响: rllm-config]
**现象**: Entropy 从 0.8963 降到 0.1718（下降 81%）
**证据**: Training Report 显示 Entropy 变化
**分析**:
- 4 步之内 entropy 下降 81%，模型过早收敛到单一策略
- 结合 reward 震荡，说明模型可能在 learning wrong pattern
- temperature=0.7 在 GRPO 中偏低，模型缺乏探索多样性
**建议**: 提高 temperature 到 0.9~1.0，或使用 cosine entropy bonus

### 4. 训练规模不足 (低严重性) [影响: rllm-train]
**现象**: 仅 16 个问题、1 epoch、4 步训练
**证据**: config.json num_problems=16, num_epochs=1
**分析**:
- 训练数据太少，模型无法学习稳定的策略
- 但受限于单卡 80GB，batch_size=1 已是 7B 的极限
- 需要多卡分布式训练才能增加规模

## 优化建议

| # | 优先级 | 目标 Skill | Section | Action | 描述 |
|---|--------|-----------|---------|--------|------|
| 1 | HIGH | rllm-train | gpu-config | append | 添加 7B 模型在单卡 A100 上的推荐配置 |
| 2 | HIGH | rllm-train | gpu-config | append | 添加 grad_norm clipping 和 warmup 配置 |
| 3 | MEDIUM | rllm-config | param-ranges | append | 7B 模型的参数安全范围 |
| 4 | MEDIUM | rllm-train | gpu-config | append | 多卡 FSDP 分布式训练建议 |

## 建议的 Patch 内容

### Patch 1: 7B 单卡 GPU 配置
Target: rllm-train (new section: gpu-a100-config)
Action: insert_after gpu-config

### Patch 2: 7B 参数安全范围
Target: rllm-config (section: param-ranges)
Action: append
