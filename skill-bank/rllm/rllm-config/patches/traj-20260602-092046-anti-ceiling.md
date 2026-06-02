---
id: traj-20260602-092046-anti-ceiling
target_section: tuning
action: replace
description: [P0] 替换通用调参策略为证据驱动规则 + 反假天花板协议
status: accepted
source: trajectory-analysis
source_sessions: ["743765dc-6160-4b38-9ebc-fb46ff27a8ef", "ae18fdb6-ae30-495b-b1c7-5959d2e445be"]
priority: P0
---

### 调参策略 (证据驱动)

#### 反假天花板协议 (ANTI-CEILING PROTOCOL)

**禁止行为**:
- ❌ 禁止声明 "模型容量上限" (capacity ceiling) 除非满足 ALL 以下条件:
  1. 至少 3 轮训练尝试了实质上不同的配置
  2. 至少尝试了 lr 在有效范围的 2x 变化
  3. 至少尝试了 batch_size 在 4x 范围内的变化
  4. 训练步数足够让 reward plateau (≥50 步无改善)
- ❌ 禁止使用 "上限/ceiling/limit/cap" 等词描述训练结果
- ✅ 使用 "当前最佳配置下的 reward" (current-best reward under tested configs)

**Ceiling Audit (天花板审查)**:
每当新轮次的 reward 超过之前声明的"上限" >20% 时，必须:
1. 列出之前所有的 "上限" 声明及轮次
2. 分析为什么之前的声明是错的
3. 将错误归因到具体原因 (配置不足/训练不充分/分析方法缺陷)
4. 更新 param-ranges 中相应的限制

**历史案例**: 
- R2 声明 "0.5B 上限 64 problems" → R9 在 256 problems 下达到 0.800
  - 归因: 把 batch_size=2 的梯度噪声误诊为 "数据量上限"
  - 真正因素: batch_size=32 使大problem数可行
- R9 报告 "0.8 是可达的" → 暗含新天花板
  - 错误: 未区分 "当前配置下的结果" vs "模型容量上限"

#### 从 R9→R10 失败的调参教训

**R10 配置回退分析**:
- use_kl_loss: True→False (关键回退! R9 证明 KL loss 对 0.5B 有益)
- max_response_length: 1024→1536 (无依据的增大)
- 原因: deep_analysis 的 "STRENGTHEN_UPDATE_SIGNAL" 建议被误解为 "关 KL loss"
- 教训: 调参建议必须包含具体参数变更, 不能仅靠抽象建议

**正确响应 STRENGTHEN_UPDATE_SIGNAL 的方式**:
- ✅ 增大 entropy_coeff (0.003→0.005): 更丰富的 policy gradient
- ✅ 增大 num_generations (8→12): 更准确的 GRPO advantage
- ✅ 增大 epochs (10→20): 更多更新机会
- ❌ 关闭 use_kl_loss: 移除唯一有效的 KL 约束
