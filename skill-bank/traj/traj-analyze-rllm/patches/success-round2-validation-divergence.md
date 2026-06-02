---
id: success-round2-validation-divergence
target_section: domain-knowledge
action: append
description: "新增失败模式#13: 验证背离 + Entropy崩塌速率量化 + Multi-run Intra-round分析方法论"
status: proposed
source: round2-reanalysis-success
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa"]
---

### 失败模式 #13: 验证背离 (Validation Divergence) [新增 — Round 2实证]

| 属性 | 值 |
|------|-----|
| 场景 | 训练 reward 震荡或不变，但验证集指标持续单调下降 |
| 关键指标 | val-acc 连续 ≥3 次检查下降，train score 无明显上升趋势 |
| 直接原因 | 模型在训练集上过拟合或学习到有害模式，该模式在验证集上无效 |
| 首要诊断 | 检查 num_problems 是否过大？epochs 是否完成？entropy 是否崩塌？ |

**指标特征**:
- `critic/score/mean`: 震荡但无明显趋势 (看起来"正常"，但实际有问题)
- `val-aux/*/reward/mean@1`: 持续单调下降 (每次val检查都在退化)
- `actor/entropy`: 通常伴随下降 (但早期可能仍在正常范围)
- train-val gap: 持续扩大

**与已有模式的区别**:
- 不同于 #1 (完全不学): train reward 并不为0, 只是不涨
- 不同于 #9 (验证背离原版): 原版侧重 train↑ val↓, 本模式是 train→ val↓
- **关键特征**: train side 看起来"还好", 但 val side 在默默退化 — 这是最隐蔽的危险模式

**排查步骤**:
1. 检查 val-acc 最近4次检查的趋势 (连续下降≥3次→确认)
2. 检查 entropy 趋势 (是否在崩塌?)
3. 检查 num_problems vs model size (是否数据过多?)
4. 检查 epochs 完成情况 (是否完成配置的epoch数?)
5. 对比训练/验证集的 problem difficulty 分布

**Case Study: Round 2** (session: c29e2662-4417-4a6b-ac04-ecbaa84daeaa):
```
Step 10:  val-acc=0.444, train_score=0.594  ← 起始: val已低于R1(0.486)
Step 50:  val-acc=0.372, train_score=0.471  ← val降16%, train看起来正常
Step 100: val-acc=0.359, train_score=0.476  ← val再降, train仍震荡
Step 128: val-acc=0.334, train_score=0.406  ← val降25%, train才刚显下降
```
诊断: 512 problems × 1 epoch + entropy_coeff=0 → entropy崩塌 + 验证背离
修复: num_problems→48, epochs→2, entropy_coeff=0.001 → 参考Round 1实证(val=0.488)

### Entropy 崩塌速率量化 [新增诊断指标]

当 entropy 呈下降趋势时，计算崩塌速率有助于在早期预警:

```
entropy_decline_rate = (entropy_start - entropy_end) / n_steps
```

临床参考阈值 (0.5B-7B模型, temp=0.3-0.85):
- rate < 0.001/step: ✅ 健康 (如 Round 1: 0.0013, val稳定)
- rate 0.001-0.002/step: ⚠️ 警戒 (如 Round 2: 0.0016, val退化)
- rate > 0.002/step: 🔴 危险 (预期val快速退化)

**注意**: 速率阈值受 temperature/entropy_coeff/model_size 影响，必须以本训练基线为参考。

### Multi-run Intra-round 分析 [新增方法论]

当一轮训练包含多次尝试 (多次 rllm-config + rllm-run):

1. **提取所有 run 的配置和指标** — 不只是最终的 run
2. **按时间排序** 重现训练探索路径: 每次配置变更试图解决什么问题?
3. **关联分析**: 配置变更 → 指标变化的因果关系
4. **中间 run 的教训** 可能比最终 run 更有价值:

Round 2案例 (session: c29e2662):
| Run | 模型 | LR | Entropy范围 | Score | 教训 |
|-----|------|-----|------------|-------|------|
| R2-R1 | 7B | 1e-7 | 0.275→0.438 | 0.446 | 7B需要≥1e-6 |
| R2-R2 | 7B | 1e-6 | 0.289→0.150 | 0.485 | 7B 1e-6可工作 |
| R2-R3 | 7B | 5e-6 | 0.038→0.072 | 0.961⚠️ | 7B上限~3e-6, 5e-6=崩塌 |
| R2-final | 0.5B | 5e-6 | 0.289→0.092 | 0.464 | 同lr对0.5B也偏高 |

→ 这些信息**只能从多run分析中获取**, 仅看final run会丢失7B vs 0.5B的关键对比数据。
