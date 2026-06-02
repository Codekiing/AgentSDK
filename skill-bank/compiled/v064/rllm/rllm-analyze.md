---
description: Analyze rllm_train training results including reward effectiveness, training
  speed, and performance bottlenecks. Generates specific hyperparameter tuning recommendations
  for the next training round.
metadata:
  categories:
  - machine-learning
  - analysis
  version: 1.0.0
name: rllm-analyze
---


# rllm-analyze — 训练结果分析与调参建议

你是 rllm_train 训练分析专家。你的任务是分析训练结果，诊断问题，并生成具体的调参建议。

## 分析输入

训练输出目录: `rllm_train/output/runs/<run_id>/`

需要读取的文件：
1. `config.json` — 本轮训练配置
2. `training_log.txt` — 训练日志（reward/loss 趋势）
3. `perf_stats.json` — 性能统计（时间分解、吞吐量）
4. `trajectories/` 目录下的 JSONL 文件 — agent 行为轨迹

## 数据表面化准则

分析时必须用 Read 工具逐一完整读取以下文件，即使对话上下文中已有部分内容:

1. `rllm_train/output/runs/<run_id>/config.json` — 训练配置
2. `rllm_train/output/runs/<run_id>/training_log.txt` — 完整训练日志
3. `rllm_train/output/runs/<run_id>/perf_stats.json` — 性能统计
4. `rllm_train/output/runs/<run_id>/trajectories/*.jsonl` — 训练轨迹（至少读取前 3 个文件）

禁止仅依赖对话上下文中已有的信息做分析。每个文件必须通过 Read 工具显式读取，确保 hooks 捕获到完整的分析输入数据。

原因: trajectory 系统通过 hooks 捕获工具调用的 response 来记录训练数据。如果分析阶段不重新 Read 这些文件，轨迹中将缺少完整的分析输入，后续的 traj-analyze-rllm 无法从轨迹数据中提取训练详情。

## 分析维度

### 一、训练效果分析

从 training_log.txt 和 perf_stats.json 中提取：

**Reward 分析**:
- 初始 reward（第 1 步）
- 最终 reward（最后一步）
- 最高 reward
- reward 趋势（上升/下降/震荡/plateau）
- 是否达到目标阈值

**Loss 分析**:
- loss 趋势（应该下降）
- loss 是否收敛
- loss 与 reward 的相关性

**Agent 行为分析**（从 trajectory JSONL）:

从 `trajectories/` 目录下所有 JSONL 文件中提取 8 个行为指标，跨 step 聚合趋势。这是诊断 loss=0 和 reward plateau 的核心数据源。

**提取方式**: 读取每个 step JSONL 文件，计算 finish_rate / finish_format_rate / tool_usage_rate / truncation_rate / answer_coverage / reward_variance / avg_agent_steps / completion_length_mean（与 rllm-monitor 相同计算逻辑），然后跨 step 聚合趋势。

**8 个行为指标**:

| 指标 | 健康趋势 | 异常趋势 | 诊断含义 |
|------|---------|---------|---------|
| finish_rate | 稳定 > 50% | 下降到 < 20% | 模型格式退化，忘了如何提交答案 |
| finish_format_rate | > 80% | 下降到 < 50% | 调了 finish 但参数格式错（`finish("text")` 而非 `finish(42)`） |
| tool_usage_rate | 稳定 > 60% | 下降到 < 30% | 模型退化为纯文本生成 |
| truncation_rate | < 20% | 上升到 > 50% | max_completion_length 不够，推理链被截断 |
| answer_coverage | > 60% | 下降到 < 30% | 输出格式崩坏，无法解析答案 |
| reward_variance | > 0.1 | 趋近 0 | GRPO 学习信号消失，loss=0 的根因 |
| avg_agent_steps | 1.5-2.5 | 上升到 > 2.8 | 模型空转，无法在有限步内完成 |
| completion_length_mean | < max_completion_length×0.7 | 上升到 > max_completion_length×0.9 | 即将被截断的前兆 |

**Loss=0 根因诊断**: 当 training_log 显示 loss=0 时，不要直接跳到"增大 lr"。先检查行为指标：

```
loss=0 → 检查:
  reward_variance < 0.02 → GRPO 无对比信号 → 增大 temperature
  finish_rate < 30% → 模型不会调 finish → 调整 system prompt
  finish_format_rate < 50% 但 finish_rate 正常 → 模型调了 finish 但参数格式错 → reward 函数无法解析
  answer_coverage < 50% → 输出格式崩坏 → 检查 max_completion_length
  truncation_rate > 50% → 输出被截断 → 增大 max_completion_length
  completion_length_mean 持续增长 → 预警截断即将发生
  全部正常但 reward < 0.3 → 模型能力不足 → 降低 difficulty
```

**错误模式分类**（从 `chat_completions` 中统计）:

| 错误类型 | 识别方式 | 影响 |
|----------|---------|------|
| 未调 finish | chat_completions 中无 finish 调用 | 不会提交答案 |
| 只调 calculate 不提交 | 有 calculate 无 finish | 推理了但忘了最后一步 |
| finish 参数格式错 | 有 finish 但参数非数字（如 `finish("the answer is 42")`） | reward 函数无法解析 |
| 答案格式错误 | 有 finish(数字) 但与 expected_answer 不符 | 模型推理有误 |
| 完全无工具调用 | 无 calculate/finish | 退化为纯文本 |
| 输出被截断 | num_steps == max_agent_steps 且 reward=0 | 推理链不完整 |

### 二、性能分析

从 perf_stats.json 中提取：

**时间分解**:
- LLM 推理时间占比
- 环境执行时间占比
- Logprob 计算时间占比
- GRPO 训练时间占比
- 其他开销

**吞吐量**:
- tokens/sec
- steps/min
- 每步平均耗时

**瓶颈识别**:
- 哪个阶段占比最高？
- 是否有异常慢的步骤？

### 三、对比分析（多轮训练时）

如果存在历史训练记录，对比：
- 本轮 vs 上一轮的 reward 变化
- 配置变更是否带来预期效果
- 性能是否有退化

### 深层诊断自动升级

`rllm-analyze` 只负责常规训练结果分析和调参建议。当出现以下任一条件时，rllm-analyze 不得输出纯参数建议，必须自动调用 `Skill("rllm-analyze-deep", args="<run_id>")` 做深层诊断：

| 条件 | 现象 | rllm-analyze-deep 动作 |
|------|------|----------------------|
| E1 — 连续 plateau | 连续 2 轮 reward 无改善（提升 < 5%） | 全自动分层排查：指标→轨迹→reward→数据 |
| E2 — 行为指标异常 | finish_rate / answer_coverage / tool_usage_rate 持续异常且调参无效 | 进入轨迹与工程异常层排查 |
| E3 — 学习信号消失 | reward_variance 长期接近 0 且 temperature/num_generations 调整无效 | 检查 reward 区分度、数据难度 |
| E4 — 疑似 reward hacking | train reward 上升但实际正确率不涨 | 进入 reward 设计与验证一致性排查 |
| E5 — 零分/满分过多 | pass@k 接近 0、同组 reward 全 0、或 reward 全 1 | 进入数据难度与 curriculum 排查 |

#### 升级执行步骤

1. rllm-analyze 完成常规分析，写入 analysis.json
2. 检查 E1-E5，如果任意触发：
   - 调用 `Skill("rllm-analyze-deep", args="<run_id>")`
   - rllm-analyze-deep 自动读取所有数据，按 15 层诊断体系排查
   - rllm-analyze-deep 输出 deep_analysis.json
3. 读取 deep_analysis.json，将其结论合并到 analysis.json 的 suggestions 中
4. 在分析报告中标注 "已通过 rllm-analyze-deep 做深层诊断"

#### 升级后的输出

analysis.json 增加字段：
```json
{
  "deep_analysis_triggered": true,
  "deep_analysis": {
    "root_cause_layer": "trajectory_engineering",
    "action": "tune" | "engineering_fix" | "data_or_reward_fix",
    "summary": "<一句话根因>"
  }
}
```

- `action = "tune"`: 返回调参建议，编排者正常回到 Phase 2
- `action = "engineering_fix"`: 输出代码修复方案，编排者自动应用后继续
- `action = "data_or_reward_fix"`: 输出数据/reward 修改方案，编排者自动应用后继续

#### 禁止行为

- 触发 E1-E5 后仍然只给纯参数建议而不调用 rllm-analyze-deep
- 调用 rllm-analyze-deep 后不将其结论写入 analysis.json
- 跳过 rllm-analyze-deep 直接输出"建议参考专家文档"

## 输出格式

### 分析报告

```
训练分析报告 [Run: <run_id>]
================================

效果分析:
  Reward:    0.25 → 0.45 (↑80%)  目标: 0.80  未达标
  Loss:      2.5 → 1.8 (↓28%)    趋势: 持续下降

行为分析:
  工具使用:    calculate 78%, finish 45%, 无工具 22%
  错误分布:    未调 finish 33%, 无工具 22%, 计算错误 15%, 格式错误 12%, 答案错误 18%
  主要错误:    未调 finish — 模型生成了推理但未提交答案
  趋势:       finish_rate 从 60%→20%，后期格式退化
  学习信号:    reward_variance 从 0.15→0.01，后期 GRPO 无梯度

性能分析:
  总耗时:    3m12s
  LLM 推理:  72% (2m18s)  ← 主要瓶颈
  环境执行:  8% (15s)
  Logprob:   12% (23s)
  GRPO 训练: 8% (15s)
  吞吐量:    11.8 tok/s

调参建议:
  1. [高] num_epochs: 2 → 4     (reward 仍在上升，增加训练量)
  2. [高] num_problems: 64 → 128 (更多训练数据)
  3. [中] temperature: 0.7 → 0.6 (减少随机性，稳定输出)
  4. [低] max_agent_steps: 3 → 2 (加速推理，减少无效轮次)
```

### 分析 JSON（写入 analysis.json）

将分析结果写入 `rllm_train/output/runs/<run_id>/analysis.json`：

```python
import json
analysis = {
    "run_id": "<run_id>",
    "round": 1,
    "reward": {"start": 0.25, "end": 0.45, "max": 0.48, "trend": "rising", "target": 0.8, "reached": False},
    "loss": {"start": 2.5, "end": 1.8, "trend": "falling"},
    "behavior": {
        "finish_rate": {"start": 0.6, "end": 0.2, "trend": "falling"},
        "finish_format_rate": {"start": 0.8, "end": 0.3, "trend": "falling"},
        "tool_usage_rate": {"start": 0.9, "end": 0.8, "trend": "stable"},
        "answer_coverage": {"start": 0.7, "end": 0.4, "trend": "falling"},
        "reward_variance": {"start": 0.15, "end": 0.01, "trend": "falling"},
        "truncation_rate": {"start": 0.05, "end": 0.45, "trend": "rising"},
        "avg_agent_steps": {"start": 2.1, "end": 2.8, "trend": "rising"},
        "completion_length_mean": {"start": 512, "end": 1020, "trend": "rising"},
        "error_distribution": {"no_finish": 0.33, "no_tool": 0.22, "calc_error": 0.15, "format_error": 0.12, "finish_format_error": 0.10, "wrong_answer": 0.08},
        "primary_error": "no_finish"
    },
    "performance": {"total_time_s": 192, "llm_pct": 72, "env_pct": 8, "logprob_pct": 12, "train_pct": 8, "tok_per_s": 11.8},
    "suggestions": [
        {"param": "num_epochs", "old": 2, "new": 4, "priority": "high", "reason": "reward still rising"},
        {"param": "num_problems", "old": 64, "new": 128, "priority": "high", "reason": "more training data"},
    ]
}
with open("rllm_train/output/runs/<run_id>/analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
```

## 独立使用

此 skill 可独立调用来分析任意已完成的训练：

```
/rllm-analyze <run_id>
```

如果未指定 run_id，自动查找最近一次训练的输出目录。
