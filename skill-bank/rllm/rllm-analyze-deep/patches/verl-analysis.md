---
id: "verl-analysis"
target_section: "intro"
action: append
description: "Handle VERL-specific analysis: parse training_log.txt for VERL Tracking metrics (instead of perf_stats.json), adapt 16-layer diagnosis for VERL metric names"
source: "2026-05-28 VERL backend integration"
created: "2026-05-28"

depends_on: []
conflicts_with: []
status: active
superseded_by: ""
---

### VERL 分析适配 (backend=verl)

当 run 使用 VERL 后端时（config.json 中 `"backend": "verl"`），分析输入格式有所不同：

#### 指标数据源差异

| 数据 | TRL 模式 | VERL 模式 |
|---|---|---|
| 训练指标 | perf_stats.json / StepRecord | training_log.txt (VERL Tracking 输出) |
| Reward 历史 | StepRecord.avg_reward | agent_grpo/critic/score/mean 时间序列 |
| Loss/梯度 | StepRecord.loss / StepRecord.grad_norm | agent_grpo/actor/grad_norm, agent_grpo/actor/pg_loss |
| KL 指标 | StepRecord.kl | agent_grpo/actor/ppo_kl |
| 吞吐量 | tokens_per_second | agent_grpo/perf/throughput |
| 完成标志 | Training Report | "Training completed" 或进程退出 |
| 验证指标 | 不支持 (TRL 无原生验证) | val-core/*/reward/mean (VERL 原生支持) |

#### VERL 指标提取

```bash
python -c "
from rllm_train.verl_analysis import extract_verl_metrics
metrics = extract_verl_metrics('rllm_train/output/runs/<run_id>/training_log.txt')
print(f'Steps: {metrics[\"total_steps\"]}')
print(f'Final reward: {metrics[\"summary\"][\"final_reward\"]}')
print(f'Max reward: {metrics[\"summary\"][\"max_reward\"]}')
"
```

提取的关键字段：
- rewards: [step, score_mean] 数组
- losses: [step, pg_loss] 数组
- grad_norms: [step, grad_norm] 数组
- entropies: [step, entropy] 数组
- val_scores: [step, val_score_mean] 数组
- completed: bool
- total_steps: int

#### 诊断流程适配

本 skill 的 16 层诊断系统优先使用以下 VERL 指标：
1. critic/score/mean（对应 TRL 的 training reward）
2. critic/rewards/mean（KL 扣减后奖励）
3. actor/ppo_kl（策略更新强度）
4. actor/pg_clipfrac（PPO clip 比例）
5. actor/grad_norm（梯度信号强度）
6. val-core/*/（验证集指标，VERL 原生支持）

TRL 专有指标（如 finish_rate, tool_usage_rate, calculator_error）在 VERL 模式下标记为 N/A，不影响分析结果。
