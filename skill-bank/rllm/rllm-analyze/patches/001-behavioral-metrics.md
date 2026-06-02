---
id: 001-behavioral-metrics
target_section: effectiveness
action: replace
description: Add behavioral trend analysis from trajectory JSONL, loss=0 root cause diagnosis, and error pattern classification
status: proposed
source: manual
---

<!-- section:effectiveness -->
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

### 二、行为趋势分析（从 trajectory JSONL）

从 `trajectories/` 目录下的所有 JSONL 文件中，逐 step 聚合行为指标。这是诊断 loss=0 和 reward plateau 的核心数据源。

**提取方式**: 读取每个 step JSONL 文件，计算 6 个行为指标（与 rllm-monitor 相同的计算逻辑），然后跨 step 聚合趋势。

```python
import json, os, glob

def compute_behavioral_trends(traj_dir):
    step_files = sorted(glob.glob(os.path.join(traj_dir, "step_*.jsonl")))
    trends = {
        "finish_rate": [], "tool_usage_rate": [], "truncation_rate": [],
        "answer_coverage": [], "reward_variance": [], "avg_agent_steps": [],
    }
    for f in step_files:
        with open(f) as fh:
            records = [json.loads(line) for line in fh]
        if not records:
            continue
        total = len(records)
        def has_finish(r):
            return any("finish" in m.get("content","").lower() for m in r.get("chat_completions",[]))
        def has_calc(r):
            return any("calculate" in m.get("content","").lower() for m in r.get("chat_completions",[]))
        import re
        def has_num(r):
            return bool(re.search(r'-?\d+\.?\d*', r.get("response_text","")))
        def is_trunc(r):
            return r.get("num_steps",0) >= 3 and r.get("reward",0) == 0
        rewards = [r.get("reward",0.0) for r in records]
        mean_r = sum(rewards)/total
        trends["finish_rate"].append(sum(1 for r in records if has_finish(r)) / total)
        trends["tool_usage_rate"].append(sum(1 for r in records if has_calc(r)) / total)
        trends["truncation_rate"].append(sum(1 for r in records if is_trunc(r)) / total)
        trends["answer_coverage"].append(sum(1 for r in records if has_num(r)) / total)
        trends["reward_variance"].append((sum((r-mean_r)**2 for r in rewards)/total)**0.5)
        trends["avg_agent_steps"].append(sum(r.get("num_steps",0) for r in records)/total)
    return trends
```

**趋势分析维度**:

| 指标 | 健康趋势 | 异常趋势 | 诊断含义 |
|------|---------|---------|---------|
| finish_rate | 稳定 > 50% | 下降到 < 20% | 模型格式退化，忘了如何提交答案 |
| tool_usage_rate | 稳定 > 60% | 下降到 < 30% | 模型退化为纯文本生成 |
| truncation_rate | < 20% | 持续上升到 > 50% | max_completion_length 不够，推理链被截断 |
| answer_coverage | > 60% | 下降到 < 30% | 输出格式崩坏，无法解析答案 |
| reward_variance | > 0.1 | 趋近 0 | GRPO 学习信号消失，loss=0 的根因 |
| avg_agent_steps | 1.5-2.5 | 上升到 > 2.8 | 模型空转，无法在有限步内完成 |

### 三、Loss=0 根因诊断

当 training_log 显示 loss=0 时，不要直接跳到"增大 lr"。先检查行为指标，定位具体原因：

```
loss=0 时的行为诊断路径:
├── reward_variance < 0.02
│   └── GRPO 无对比信号 → 增大 temperature（用 entropy_bonus fix_preset）
├── finish_rate < 30%
│   └── 模型不会调 finish → 需要调整 system prompt 或添加格式 reward
├── answer_coverage < 50%
│   └── 输出格式崩坏 → 检查 max_completion_length 是否截断了 tool call
├── truncation_rate > 50%
│   └── 输出被截断 → 增大 max_completion_length（用 increase_max_completion_length fix_preset）
└── 全部正常但 reward < 0.3
    └── 模型能力不足 → 降低 difficulty 或换更大模型
```

### 四、错误模式分类

从 trajectory JSONL 的 `chat_completions` 和 `response_text` 中，分类统计每种错误类型的占比：

| 错误类型 | 识别方式 | 影响 |
|----------|---------|------|
| 未调 finish | chat_completions 中无 finish 调用 | 模型不知道如何提交答案 |
| 只调 calculate 不提交 | 有 calculate 无 finish | 模型在推理但忘了最后一步 |
| 答案格式错误 | 有 finish 但参数非数字 | 模型提交了非数值答案 |
| 计算正确但答案错 | finish 提交了数字但与 expected_answer 不符 | 模型推理有误 |
| 完全无工具调用 | chat_completions 中无 calculate/finish | 模型退化为纯文本生成 |
| 输出被截断 | num_steps == max_agent_steps 且 reward=0 | max_completion_length 不够 |

**错误分布统计**: 在分析报告中列出各错误类型的占比，优先级按占比排序。
<!-- /section:effectiveness -->

<!-- section:decision-tree-v2 -->
### 调参决策树（修订版）

在原有决策树基础上，增加行为指标驱动的诊断分支。

```
reward 未达标？
├── loss=0（无学习信号）
│   ├── reward_variance < 0.02
│   │   └── GRPO 无对比信号 → temperature += 0.2 或增加 num_generations
│   ├── finish_rate < 30%
│   │   └── 模型不会调 finish → 调整 system prompt 或添加格式辅助 reward
│   ├── answer_coverage < 50%
│   │   └── 输出格式崩坏 → 检查 max_completion_length 是否截断
│   └── 全部正常 → 模型能力不足，降低 difficulty 或换更大模型
│
├── loss > 0 但 reward 不升
│   ├── reward_variance 高但 finish_rate 低
│   │   └── 模型在探索但格式不对 → 调整 agent prompt
│   └── reward_variance 低
│       └── 探索不足 → 增大 temperature
│
├── reward 在上升
│   ├── loss 在降 → 增加 epochs 或 problems（需要更多训练）
│   └── loss 不降 → 增大 lr 或 num_generations（学习信号不足）
│
├── reward 停滞 (plateau)
│   ├── finish_rate 下降趋势
│   │   └── 格式退化导致 plateau → early stop，下轮减少 num_problems
│   ├── loss 也停滞 → 调整 temperature、尝试不同 loss_type
│   └── loss 在降 → 增加 num_generations（探索不足）
│
├── reward 震荡
│   └── 降低 lr、增大 grad_accum_steps（训练不稳定）
│
└── reward 下降
    ├── finish_rate 同步下降 → 格式退化导致 reward 下降
    └── finish_rate 稳定 → 过拟合或 lr 过大，降低 lr
```

### 行为指标驱动的调参建议

基于行为指标趋势，生成更精准的调参建议：

| 行为指标异常 | 调参建议 | 原因 |
|-------------|---------|------|
| finish_rate < 20% 持续 3 步 | 调整 system prompt 或增加格式 reward | 模型忘了如何提交答案 |
| tool_usage_rate < 30% | 检查 agent 初始化是否正确传递了工具定义 | 模型不知道有 calculate 工具 |
| truncation_rate > 50% | max_completion_length 翻倍 | 推理链被截断 |
| answer_coverage < 30% | 检查 reward_fn 能否解析模型输出 | 输出格式与 reward 函数不匹配 |
| reward_variance < 0.02 | temperature += 0.2 或 num_generations += 2 | GRPO 需要 reward 对比才能计算梯度 |
| avg_agent_steps > 2.8 | 减少 max_agent_steps 或优化 agent prompt | 模型在空转 |
<!-- /section:decision-tree-v2 -->

<!-- section:output-format-v2 -->
### 分析报告格式（修订版）

在原有报告格式基础上，增加行为分析段。

```
训练分析报告 [Run: <run_id>]
================================

效果分析:
  Reward:    0.25 → 0.45 (↑80%)  目标: 0.80  未达标
  Loss:      2.5 → 1.8 (↓28%)    趋势: 持续下降

行为分析:
  工具使用:    calculate 78%, finish 45%, 无工具 22%
  错误分布:    未调 finish 33%, 无工具 22%, 计算错误 15%, 格式错误 12%, 答案错误 18%
  主要错误:    未调 finish — 模型生成了推理过程但未提交最终答案
  趋势:       finish_rate 从 60% 降至 20%，后期格式退化严重
  学习信号:    reward_variance 从 0.15 降至 0.01，后期 GRPO 无梯度

性能分析:
  总耗时:    3m12s
  LLM 推理:  72% (2m18s)  ← 主要瓶颈
  吞吐量:    11.8 tok/s

调参建议:
  1. [高] temperature: 0.7 → 0.9  (reward_variance < 0.02，GRPO 无学习信号)
  2. [高] max_completion_length: 1024 → 2048  (truncation_rate > 50%)
  3. [中] max_agent_steps: 3 → 5  (avg_agent_steps 接近上限)
```

### 分析 JSON 格式（修订版）

在原有 analysis.json 基础上，增加 `behavior` 字段：

```python
import json
analysis = {
    "run_id": "<run_id>",
    "round": 1,
    "reward": {"start": 0.25, "end": 0.45, "max": 0.48, "trend": "rising", "target": 0.8, "reached": False},
    "loss": {"start": 2.5, "end": 1.8, "trend": "falling"},
    "behavior": {
        "finish_rate": {"start": 0.6, "end": 0.2, "trend": "falling"},
        "tool_usage_rate": {"start": 0.9, "end": 0.8, "trend": "stable"},
        "answer_coverage": {"start": 0.7, "end": 0.4, "trend": "falling"},
        "reward_variance": {"start": 0.15, "end": 0.01, "trend": "falling"},
        "truncation_rate": {"start": 0.05, "end": 0.45, "trend": "rising"},
        "avg_agent_steps": {"start": 2.1, "end": 2.8, "trend": "rising"},
        "error_distribution": {
            "no_finish": 0.33, "no_tool": 0.22, "calc_error": 0.15,
            "format_error": 0.12, "wrong_answer": 0.18
        },
        "primary_error": "no_finish"
    },
    "performance": {"total_time_s": 192, "llm_pct": 72, "env_pct": 8, "logprob_pct": 12, "train_pct": 8, "tok_per_s": 11.8},
    "suggestions": [
        {"param": "temperature", "old": 0.7, "new": 0.9, "priority": "high", "reason": "reward_variance < 0.02, GRPO has no learning signal"},
        {"param": "max_completion_length", "old": 1024, "new": 2048, "priority": "high", "reason": "truncation_rate > 50%"},
    ]
}
with open("rllm_train/output/runs/<run_id>/analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
```
<!-- /section:output-format-v2 -->
