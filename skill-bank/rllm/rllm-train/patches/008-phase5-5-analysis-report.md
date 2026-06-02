---
id: "008-phase5-5-analysis-report"
target_section: "phase1-5"
action: insert_after
description: "Add mandatory Phase 5.5 detailed analysis report: round-over-round comparison, metric diagnosis, problem identification, next measures with reasons"
source: "2026-05-28 VERL backend integration feedback"
created: "2026-05-28"

depends_on:
  - "verl-backend-routing"
conflicts_with: []
status: active
superseded_by: ""
---

### Phase 5.5: 训练总结与详细分析报告（编排者自己执行）

rllm-analyze-deep 完成后，编排者必须输出详细分析报告。**禁止**只报 avg/max reward 后就进入下一阶段。

#### 报告必须包含以下 7 个部分：

**1. 本轮配置**

列出本轮关键参数，标注与上一轮的变更及原因。

**2. 训练效果总览**

```
指标对比:
  Reward avg:     R(N-1) → R(N)  (±X%, 趋势: 上升/下降/持平)
  Reward max:     R(N-1) → R(N)
  Grad norm:      R(N-1) → R(N)  (信号强度: 增强/减弱/稳定)
  Entropy:        start → end    (探索: 扩张/收缩/稳定)
  Response len:   start → end    (简洁度变化)
  Step time:      avg Xs/step    (训练速度)
  Clipfrac:       max X          (更新裁剪率)
  PPO KL:         max X          (策略偏离度)
```

**3. Epoch 级分析**（多 epoch 时必须输出）

```
Epoch 1 avg: X.XXX | Epoch 2 avg: X.XXX | ...
Epoch 间趋势: 上升/下降/持平 (±X%)
跨 epoch 学习是否有效: 是/否，原因: ...
```

**4. 问题诊断**

基于 16 层诊断系统，列出发现的问题和未确认的风险：
- [问题1] 指标证据 + 严重程度 + 根因分析
- [问题2] ...
- [风险1] 需持续观察的指标

**5. 改进措施**

按优先级列出参数变更建议，每条包含：
```
优先级 N: <参数名> <旧值>→<新值>
  原因: <基于哪项指标/问题的推理>
  预期: <预期效果>
  风险: <可能的副作用>
```

**6. 本轮结论**

```
目标:        avg reward >= <target>
当前:        avg reward = <current> (±X% vs 上轮)
是否达标:    ✓/✗
进度评估:    <一句话，判断是否在正确轨道上>
是否继续:    <继续下一轮/停止-已达目标/停止-达到最大轮次/停止-连续无提升>
```

**7. 训练历程汇总**

```
轮次  配置关键参数          Reward avg   Reward max   时间    状态
R1    lr=X, ep=X, gen=X    0.XXX        0.XXX        Xm     基线
R2    ep=X→Y, temp=X→Y    0.XXX        0.XXX        Xm     +X%
R3    ...                  0.XXX        0.XXX        Xm     ±X%
```

#### 互轮比较规则

- **必须对比当前轮与所有历史轮次**，不能只看本轮的绝对值
- 如果 reward 提升 < 5%：标注"边际提升"，重点分析是否需要换方向
- 如果 grad_norm 连续 2 轮下降 > 30%：标注"更新信号衰减"，建议降低 lr 或增加 rollout.n
- 如果 entropy 连续 2 轮上升但 reward 不涨：标注"无效探索"，建议降低 temperature
- 如果 response_length 持续下降 > 50%：标注"输出退化风险"，检查是否过度压缩

#### VERL 特有分析

当 backend=verl 时，额外分析：
- `critic/score/max` 趋势：GRPO 组内满分样本是否增加
- `critic/score/min` vs `critic/score/max` 差距：题目难度差异是否在缩小
- `response_length/clip_ratio`：截断是否严重
- `perf/mfu/actor_infer`：MFU 是否正常（>0.1 为合理）
<!-- /section:phase5-5 -->
