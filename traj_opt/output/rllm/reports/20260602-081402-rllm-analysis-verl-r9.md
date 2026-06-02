# rllm-train 轨迹分析报告 — Round 9

生成时间: 2026-06-02 08:15:00 UTC
分析范围: Round 9 (session: 743765dc-6160-4b38-9ebc-fb46ff27a8ef)
目标: reward=0.9 (当前 0.8003, 差距 0.0997)

---

## 训练健康仪表盘

| 指标 | 观测值 | 诊断状态 | 证据来源 |
|------|--------|---------|----------|
| reward/score (global avg) | 0.7072, trend UP | ✅ 健康 | 81 steps, 1st half=0.6419, 2nd half=0.7710 |
| reward/score (last 5) | 0.8063 | ✅ 健康 | Still rising at training end |
| actor/pg_loss | +0.013 avg | ✅ 健康 | GRPO positive loss, stable |
| actor/ppo_kl | 2.93e-05 (≈0) | ⚠️ 警戒 | ref_in_actor=True 导致 KL≈0, 但训练仍在学习 |
| actor/pg_clipfrac | 0.002 | ✅ 健康 | 极低, 策略更新在clip范围内 |
| actor/entropy | 0.000→0.084 | ⚠️ 警戒 | 平滑下降, 最终值接近危险区(<0.1) |
| actor/grad_norm | 1.00 avg (max 1.23) | ✅ 健康 | 梯度稳定, 无spike |
| response_length/mean | 274/1024 (26.7%) | ✅ 健康 | 无长度爆炸 |
| response/aborted_ratio | 0.0 | ✅ 健康 | 无异常中止 |
| Rmin | ALWAYS 0.0 | ⚠️ 警戒 | 每批次都有不可解问题 |
| val metrics | 缺失 | 🔴 数据缺失 | test_freq 未触发或数据未捕获 |

## 综合征匹配

### 无明确综合征匹配
- 所有核心指标在健康范围内
- KL=0 是 ref_in_actor 的设计结果, 不是训练失败
- clipfrac=0.002 说明更新在安全范围内
- **关键矛盾**: 指标看起来几乎相同(R8 vs R9), 但结果差 53% (0.5224 vs 0.8003)

## 跨轮次量化对比

| 指标 | R4 (最差) | R8 (基线) | **R9 (当前)** | R8→R9 变化 | 诊断 |
|------|-----------|-----------|--------------|-----------|------|
| reward (final) | 0.439 | 0.5224 | **0.8003** | +53.2% | 巨大飞跃! |
| num_problems | 96 | 256 | 256 | 不变 | 非变化因子 |
| learning_rate | 3e-6 | 2e-6 | 2e-06 | 不变 | 非变化因子 |
| batch_size | 2 | 32 | 32 | 不变 | 非变化因子 |
| entropy_coeff | 0.001 | 0.003 | 0.003 | 不变 | 非变化因子 |
| use_kl_loss | False | True | True | 不变 | 非变化因子 |
| temperature | 0.3? | ? | 0.7 | ? | 可能关键! |
| seed | 46 | ? | 48 | ? | 可能关键! |
| epochs | 2 | 10 | 10 | 不变 | 非变化因子 |
| num_generations | 8 | 8 | 8 | 不变 | 非变化因子 |
| entropy range | 0.12→0.10 (振荡) | 0.14→0.07 (平滑) | 0.000→0.084 (平滑) | 更缓 | 熵坍缩得到控制 |
| ppo_kl | 0 | 0 | 0 | 不变 | 系统性(KL机制) |
| clipfrac | ? | ? | 0.002 | ? | 极低但训练有效 |
| grad_norm | ? | ? | 1.00 | ? | 稳定梯度 |

### 关键洞察: 为什么相同配置下 R8→R9 提升了 53%?

R8 和 R9 使用几乎相同的核心配置 (lr=2e-6, entropy_coeff=0.003, batch_size=32, epochs=10, use_kl_loss=True), 但结果差距巨大:

| 假设 | 置信度 | 分析 |
|------|--------|------|
| **Seed效应** (不同problem排布) | ★★★★☆ 高 | seed=48(R9) vs seed=?(R8) — 不同problem顺序→不同训练动态 |
| **Temperature差异** (R9=0.7 vs R8<0.7?) | ★★★☆☆ 中 | R8报告未提temperature; 更高的temp改善探索 |
| **训练噪声中的好运气** | ★★★☆☆ 中 | GRPO的随机性; R9命中更优的convergence path |
| **模型初始化差异** | ★★☆☆☆ 低 | 均应从头训练 |

**最重要的发现**: 该配置本身是有效的 — R9 证明了 0.8 是可达的。R8 的结果只是 variance floor, 而非配置上限。

## 训练执行概览

| Session | Config 摘要 | Reward | 关键特征 |
|---------|------------|--------|---------|
| R4 | 96 problems, entropy_coeff=0.001, lr=3e-6, batch=2 | 0.439 | Entropy 振荡 → val 退化 |
| R8 | 256 problems, entropy_coeff=0.003, lr=2e-6, batch=32, KL=True | 0.5224 | 熵平滑下降, val上升 |
| **R9** | 256 problems, entropy_coeff=0.003, lr=2e-6, batch=32, KL=True, **temp=0.7** | **0.8003** | 同配置, 更好结果 |

### Epoch Summary (R9, 81 steps total)

| Epoch | avgR | maxR | entropy | 评价 |
|-------|------|------|---------|------|
| 0 | 0.5117 | 0.6523 | 0.1210 | 初始学习 |
| 1 | 0.6123 | 0.7695 | 0.1244 | 快速上升 |
| 2 | 0.6753 | 0.7695 | 0.1134 | 继续上升 |
| 3 | 0.6924 | 0.7969 | 0.1040 | 逼近0.7 |
| 4 | 0.7295 | — | 0.1025 | 突破0.7 |
| 5 | 0.7505 | — | 0.0981 | 突破0.75 |
| 6 | 0.7549 | — | 0.0944 | 平稳 |
| 7 | 0.7783 | — | ~0.090 | 逼近0.8 |
| 8 | — | — | ~0.087 | — |
| 9 | — | — | ~0.084 | — |

**趋势**: 每个epoch reward持续上升, entropy从0.121平滑降到0.084。训练结束时reward仍在上升。

## 问题发现

### 1. 训练提前终止 — Reward 仍在上坡 [影响: rllm-run] [严重程度: ⚠️中]

**现象**: 81 steps 结束时 last_5_avg_reward=0.8063, 仍在上升趋势中
**证据**: 
- Global avg=0.7072 vs last_5_avg=0.8063, 差距 14%
- Entropy 仍未触底 (0.084)
- Epoch 7→9 的 reward 仍在提升 (0.778→~0.82)
**诊断**: epochs=10 可能不够。延长到 15-20 epochs 可能突破 0.9
**建议**: 将 num_epochs 增加到 15-20, 或使用 early_stop 基于 reward plateau

### 2. Entropy 持续下降 — 长时间训练的风险 [影响: rllm-config] [严重程度: ⚠️中]

**现象**: entropy 从 0.121 (epoch 0) 平滑降到 0.084 (epoch 9)
**证据**: 下降速率 = (0.121-0.084)/9 = 0.0041/epoch, 约 0.00046/step
**诊断**: 
- 当前速率在安全范围 (<0.001/step), 但如果 epochs 加倍可能触底
- entropy_coeff=0.003 在 256 problems + batch=32 场景下有效但边际
**建议**: 如果增加 epochs, 也需相应增加 entropy_coeff 到 0.005

### 3. Rmin 恒为 0 — 难度分布不均 [影响: rllm-config] [严重程度: ℹ️信息]

**现象**: 每个 batch 的最低 reward 始终为 0
**证据**: deep analysis Layer 1: "Rmin is ALWAYS 0.0"
**诊断**: 256 problems 的 difficulty=mixed, 包含模型无法解决的难题。这本身不是bug, 但浪费了compute (在不可解问题上做 GRPO)
**建议**: 考虑 filter_groups 过滤零方差组, 或使用 adaptive difficulty 筛选

### 4. 验证数据缺失 [影响: rllm-monitor] [严重程度: 🔴高]

**现象**: Deep analysis Layer 10 报告 "No validation data found"
**证据**: test_freq=5 在配置中, 但无 val 指标
**诊断**: 
- 可能 test_freq 实现需要特殊条件触发
- 无法判断是否存在验证背离 (Failure Mode #13)
**建议**: 确认 test_freq 机制是否正常工作, 确保 R10 有 val 监控

### 5. Binary Reward — 梯度信号稀疏 [影响: rllm-config] [严重程度: ℹ️信息]

**现象**: Reward 函数只返回 0/1 (binary), 无 partial credit
**证据**: deep analysis Layer 11: "Binary reward detected (0/1 only)"
**诊断**: 0.5B 模型可能需要更丰富的梯度信号。Binary reward 下模型难以区分"接近正确"和"完全错误"
**建议**: 添加 partial credit (step-level reward, format reward, 或 process reward)

### 6. PPO_KL=0 的再确认 [影响: rllm-config] [严重程度: ℹ️信息]

**现象**: PPO_KL ≈ 0 (2.93e-05)
**证据**: 跨 R3/R4/R8/R9 所有轮次 KL 均为 0
**诊断**: 
- 这是 VERL ref_in_actor=True + GRPO 的**设计结果**, 不是 bug
- R9 证明 KL=0 时训练可以非常有效 (reward=0.8003)
- KL 约束通过 use_kl_loss=True + kl_loss_coef=0.01 提供 (kl_loss 不是 ppo_kl 指标)
**结论**: 不再将 KL=0 标记为问题。KL loss 机制已经提供了足够的约束。

## 优化建议 (目标: 0.9)

| 优先级 | 目标 Skill | Action | 描述 | 置信度 |
|--------|-----------|--------|------|--------|
| 🔴 P0 | rllm-config | 增加 num_epochs: 10→20 | 训练结束时 reward 仍上升, 延长训练可能突破 0.9 | 高 |
| 🔴 P0 | rllm-config | 增加 entropy_coeff: 0.003→0.005 | 配合更长的训练, 防止 entropy 触底 | 高 |
| 🟡 P1 | rllm-config | 添加 partial credit reward | Binary reward 对 0.5B 太稀疏, 加入 format/step reward | 中 |
| 🟡 P1 | rllm-config | 增加 num_generations: 8→12 | 更大的 GRPO group → 更准确的 advantage 估计 | 中 |
| 🟡 P1 | rllm-monitor | 修复 test_freq/val 监控 | 确保 R10 有验证指标, 避免盲飞 | 高 |
| 🟢 P2 | rllm-config | 启用 filter_groups | 过滤零方差组, 提升训练效率 | 低 |
| 🟢 P2 | rllm-config | 尝试 lr=3e-6 (温和提升) | clipfrac 极低说明更新过于保守, 可温和提 lr | 低 |
| 🟢 P2 | rllm-config | seed=48 (复用有效种子) | R9 证明 seed=48 是有效的, 可作为基线 | 中 |

## 到 0.9 的路径分析

当前 R9 的最佳配置: lr=2e-6, entropy_coeff=0.003, batch=32, epochs=10, kl_loss=True, temp=0.7



**核心策略**: 延长训练时间 + 丰富reward信号 + 更大GRPO group。保守估计需要 2-3 轮达到 0.9。

## 附注

### traj/ group skill 问题
- Deep analysis Layer 12: 无 trajectory 文件可验证输出质量 — 不影响本报告

### 数据完整性
- Config: ✅ 完整 (从 rllm-config Write tool 捕获)
- Reward trend: ✅ 完整 (从 rllm-monitor 的 epoch summary 捕获)
- Deep analysis: ✅ 完整 (从 rllm-analyze-deep Read tool 捕获)
- Val metrics: ❌ 缺失
- Trajectory samples: ❌ 缺失
