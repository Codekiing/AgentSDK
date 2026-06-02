---
description: Monitor rllm_train training progress in real-time. Tracks reward trends,
  training speed, and detects anomalies like loss explosion or process crashes.
metadata:
  categories:
  - machine-learning
  - monitoring
  version: 1.0.0
name: rllm-monitor
---


# rllm-monitor — 训练过程监控

你负责实时监控 rllm_train 训练进度，向用户汇报关键指标，并检测异常。

## 监控目标

- 日志文件: `rllm_train/output/runs/<run_id>/training_log.txt`
- 性能统计: `rllm_train/output/runs/<run_id>/perf_stats.json`（训练结束后生成）
- 轨迹文件: `rllm_train/output/runs/<run_id>/trajectories/`（训练过程中逐步生成）

## 监控方式

### 实时监控（训练进行中）

使用 Monitor 工具监控训练日志：

```bash
tail -f rllm_train/output/runs/<run_id>/training_log.txt | grep -E --line-buffered "/[0-9]+|···|Error|Traceback|FAILED|OOM|Training Report"
```

### 定期检查（训练进行中）

每隔一段时间读取日志文件尾部，提取关键指标：

```bash
tail -20 rllm_train/output/runs/<run_id>/training_log.txt
```

### 训练日志格式

rllm_train 的 TrainingLogger 输出格式（参考 `rllm_train/logger.py`）：

进度行格式（每个 step 会输出多行子步骤 + 一行汇总）:
```
    ··· step 1/128: generating 4 trajectories...
    ··· trajectory 1/4 done (reward=1.000)
    ··· trajectory 2/4 done (reward=0.000)
    ··· trajectory 3/4 done (reward=1.000)
    ··· trajectory 4/4 done (reward=1.000)
  1/128     4    0.750      6.9s     88.4    29m57s
    ··· computing logprobs...
    ··· training update...
```

子步骤行以 `···` 开头，汇总行以 `step/total` 格式开头。

训练完成标志:
```
Training Report
==============
```

### Monitor 可靠性设计

#### 1. grep 模式通用化

当前（会失效）:
```bash
tail -f ... | grep -E "/64|···|Error"
```

改为（通用）:
```bash
tail -f ... | grep -E --line-buffered "^\s*[0-9]+/[0-9]+|···|Error|Traceback|OOM|Training Report|reward="
```

#### 2. 双重监控策略

主监控: Monitor 工具，persistent=true，做实时流式通知
备用监控: 当主监控连续 2 分钟无输出时，用 tail -20 读日志尾部

切换逻辑 (由 rllm-train 编排层管理):
  if 收到 Monitor 通知: `last_notification_time = now`
  if `now - last_notification_time > 120s`:
      执行 tail -20 检查训练状态
      if 训练仍在进行: 重启 Monitor (TaskStop 旧的 + 启动新的)
      if 训练已完成: 进入 Phase 5

#### 3. Monitor 生命周期管理

rllm-train 编排层增加:
  Phase 4 启动 Monitor 后，记录 monitor_task_id
  训练完成后: TaskStop(monitor_task_id) 清理
  Monitor 超时退出且训练未完成: 自动重启新 Monitor
  用户请求停止训练: 先 TaskStop(monitor_task_id)，再 kill 训练进程

#### 4. Monitor 健康检查

启动 Monitor 后 30s 内如果无任何输出:
  检查日志文件是否存在且在增长
  检查 grep 模式是否匹配到内容
  如果日志在增长但 grep 无匹配: 报告 grep 模式可能有误，切换到宽松模式

### 主动轮询强制规则 (ACTIVE POLLING MANDATE)

#### 核心规则：Monitor 必须主动轮询，绝不被动等待

**定义**:
- **主动监控 (Active Monitoring)**: 优先启动一次持久 `tail -f training_log.txt | grep --line-buffered '\[MONITOR_STEP\]\|Training Report\|Traceback\|Error\|OOM'` 流式监控；训练进程每个 step 自带 `[MONITOR_STEP]` 固定 schema，包含 Len 和行为指标。
- **主动轮询 (Active Polling)**: 当持久流式监控不可用或 120s 无新输出时，编排层执行 `tail -30 training_log.txt` 兜底，提取最新 `[MONITOR_STEP]` 并汇报关键指标。
- **被动等待 (Passive Waiting)**: 启动一个后台 while 循环或 background task 等待训练完成，期间不做任何 tail 或汇报

**强制规则**:

1. **主动监控是唯一的监控手段**。编排层 MUST 优先启动一个持久流式日志监控:
   ```bash
   tail -f rllm_train/output/runs/<run_id>/training_log.txt | grep -E --line-buffered '\[MONITOR_STEP\]|Training Report|Traceback|Error|OOM'
   ```
   训练进程每完成一个 step 会自行输出一行 `[MONITOR_STEP]`，其中包含固定 schema 的 Len 和行为指标。只有当流式监控不可用、启动后 120s 无输出、或需要诊断异常时，才执行兜底:
   ```bash
   tail -30 rllm_train/output/runs/<run_id>/training_log.txt
   ```

2. **禁止被动等待**。编排层 MUST NOT:
   - 创建 `while true; do sleep; done` 后台循环来等待训练完成
   - 仅依赖 `TaskOutput` 阻塞等待而不做中间汇报
   - 在任何 step 之后停止 tail 检查
   - 将"等待训练完成"作为唯一的监控策略

3. **每个 step 必须汇报，且格式固定**。每收到一行 `[MONITOR_STEP]`，编排层 MUST 原样或等价输出一行训练状态摘要。兜底 tail 检查时也必须输出同一字段顺序；缺失字段统一填 `—`，不得临时增减字段。
   ```
   Step X/Y | R Z.ZZZ | Rstd Z.ZZZ | Loss L.LLLL | Grad G.GGGG | Ent E.EEEE | Clip C.CC | Len N | Finish P% | FmtOK P% | Tool P% | Ans P% | tok/s T.T | Time S.Ss | ETA ~Mm | Status OK/WARN/STOP
   ```
   字段含义固定为：`R=avg_reward`, `Rstd=reward_variance`, `Loss=policy_loss`, `Grad=grad_norm`, `Ent=entropy`, `Clip=clipped_ratio`, `Len=completion_length_mean`, `Finish=finish_rate`, `FmtOK=finish_format_rate`, `Tool=tool_usage_rate`, `Ans=answer_coverage`, `Time=step_time`。

   如果没有新 step 完成，也必须输出同一格式，复用最新已完成 step 的指标，并将 `Status` 写为 `RUN(no-new-step)`；如果还没有任何已完成 step，则所有指标填 `—`。

   `Status` 必须优先暴露长度/截断风险：当 `Clip >= 0.80` 或 `Len >= max_completion_length * 0.90` 时写 `WARN(length-limit)`；连续 3 个已完成 step 命中该条件时写 `STOP(length-limit)` 并触发 `increase_max_completion_length`。

4. **轮询不可中断**。只有以下情况可以停止主动轮询:
   - 训练已完成（检测到 "Training Report" 或后台进程退出）
   - 检测到严重异常（OOM、崩溃）需要停止训练
   - 用户明确要求停止

5. **后台进程是辅助，不是替代**。训练进程可以后台运行（`run_in_background: true`），但后台进程状态检查只是主动轮询的补充信息来源，不能替代 tail 日志检查。

#### 执行模式

正确的监控循环 (伪代码):

```
train_process = Bash("torchrun ...", run_in_background=true)
monitor = Bash("tail -f training_log.txt | grep -E --line-buffered '\\[MONITOR_STEP\\]|Training Report|Traceback|Error|OOM'", run_in_background=true)
last_step = 0
last_monitor_output = now

while 训练未完成:
    # 1. 优先消费持久流式监控输出；每行 [MONITOR_STEP] 已含完整固定 schema
    for line in monitor.new_lines():
        if "[MONITOR_STEP]" in line:
            向用户汇报 line 去掉前缀后的固定 schema
            last_step = extract_step(line)
            last_monitor_output = now
        check_anomalies(line)

    # 2. 仅当 120s 无流式输出时，执行一次 tail 兜底检查
    if now - last_monitor_output > 120s:
        log_output = Bash("tail -30 training_log.txt")
        current_step = extract_latest_monitor_step(log_output)
        向用户汇报最新固定 schema；若无新 step，Status=RUN(no-new-step)
        last_monitor_output = now

    # 3. 检查训练是否完成
    if "Training Report" in recent_output or 训练进程已退出:
        TaskStop(monitor)
        break
```

错误的监控模式 (禁止):

```
# 错误: 只启动后台等待，不做中间汇报
train_process = Bash("torchrun ...", run_in_background=true)
TaskOutput(train_process, block=true, timeout=600000)  # 阻塞等待，中间无输出
# 用户在 30 分钟内看不到任何训练状态
```

## 汇报内容

### 进度汇报（每次主动轮询）

每次轮询必须输出同一行固定 schema；缺失字段填 `—`，不得按本轮拿到的指标临时增减字段。

```
Step X/Y | R Z.ZZZ | Rstd Z.ZZZ | Loss L.LLLL | Grad G.GGGG | Ent E.EEEE | Clip C.CC | Len N | Finish P% | FmtOK P% | Tool P% | Ans P% | tok/s T.T | Time S.Ss | ETA ~Mm | Status OK/WARN/STOP
```

### 异常检测（修订）

| 异常 | 检测方式 | 处理 |
|---|---|---|
| Reward 归零 | 连续 5 步 reward=0 | 建议停止训练 (非仅报告) |
| Reward 崩溃 | reward 从 >0.3 降到 0 且持续 3 步 | 立即建议停止，诊断为 lr 过高或 forgetting |
| Loss 爆炸 | loss > 10 或 NaN/Inf | 立即建议停止 |
| OOM | "out of memory" | 立即建议停止 |
| 进程崩溃 | Traceback + 进程退出 | 报告错误 |
| 训练卡住 | 超过 120s 无输出 | 报告，检查进程状态 |

### Early Stopping 机制

Monitor 检测到以下条件时，向编排层发送 STOP 建议:

1. 连续 5 步 reward=0 且当前 step > total_steps * 0.2
   → "训练已崩溃，建议停止。连续 5 步 reward=0，继续训练不会恢复。"

2. Epoch 切换后 reward 断崖 (需要按 epoch 计算)
   → "进入 Epoch N 后 reward 从 X 降到 0，疑似 catastrophic forgetting，建议停止。"

3. 模型输出异常 (从 trajectory 文件检测)
   → "模型输出格式退化，不再使用 tool_call，建议停止。"

### Epoch 边界监控

计算 epoch 边界: `epoch_boundary = num_problems / (batch_size * grad_accum)`

当 step 跨越 epoch 边界时:
  读取最近 3 步的 reward
  与上一个 epoch 最后 3 步的 reward 对比
  如果下降 > 50%: 发出 catastrophic forgetting 预警

<!-- section:behavioral-metrics -->
### 训练日志行为指标提取

每步训练完成后，训练进程会直接向 `training_log.txt` 写入一行 `[MONITOR_STEP]` 固定 schema 日志，包含 Len 和行为指标。这些指标能直接解释 loss=0 和 reward plateau 的根因，并避免 monitor 每步 Read trajectory 文件导致频繁权限确认。

**首选数据来源**: `rllm_train/output/runs/<run_id>/training_log.txt` 中的 `[MONITOR_STEP]` 行

示例：

```text
[MONITOR_STEP] Step 1/64 | R 0.500 | Rstd 0.500 | Loss 0.1000 | Grad 0.3000 | Ent 0.2000 | Clip 0.10 | Len 123 | Finish 75% | FmtOK 100% | Tool 100% | Ans 75% | tok/s 12.3 | Time 4.2s | ETA ~8m | Status OK
```

**兜底数据来源**: 如果旧 run 没有 `[MONITOR_STEP]` 行，才读取 `rllm_train/output/runs/<run_id>/trajectories/step_XXXX.jsonl` 计算指标。

**兜底提取方式**：

```python
import json

def extract_behavioral_metrics(step_file):
    with open(step_file) as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    if total == 0:
        return {}

    # chat_completions 中是否出现 finish 调用
    def has_finish(record):
        for msg in record.get("chat_completions", []):
            content = msg.get("content", "")
            if "finish" in content.lower() or '"name": "finish"' in content:
                return True
        return False

    # chat_completions 中是否出现 calculate 调用
    def has_calculate(record):
        for msg in record.get("chat_completions", []):
            content = msg.get("content", "")
            if "calculate" in content.lower() or '"name": "calculate"' in content:
                return True
        return False

    # response_text 中是否含数字
    def has_numeric_answer(record):
        import re
        text = record.get("response_text", "")
        return bool(re.search(r'-?\d+\.?\d*', text))

    # 是否被截断（达到 max_steps 且 reward=0）
    def is_truncated(record):
        return record.get("num_steps", 0) >= 3 and record.get("reward", 0) == 0

    finish_count = sum(1 for r in records if has_finish(r))
    calculate_count = sum(1 for r in records if has_calculate(r))
    numeric_count = sum(1 for r in records if has_numeric_answer(r))
    trunc_count = sum(1 for r in records if is_truncated(r))
    rewards = [r.get("reward", 0.0) for r in records]
    steps = [r.get("num_steps", 0) for r in records]

    return {
        "finish_rate": finish_count / total,
        "tool_usage_rate": calculate_count / total,
        "truncation_rate": trunc_count / total,
        "answer_coverage": numeric_count / total,
        "reward_variance": (sum((r - sum(rewards)/total)**2 for r in rewards) / total) ** 0.5,
        "avg_agent_steps": sum(steps) / total,
    }
```

**8 个行为指标定义**:

| 指标 | 含义 | 正常范围 | 异常信号 |
|------|------|---------|---------|
| finish_rate | 调用了 finish 的轨迹占比 | > 50% | < 20% = 格式退化 |
| finish_format_rate | 调了 finish 且参数为数字的占比 | > 80% | < 50% = finish 格式错误 |
| tool_usage_rate | 使用了 calculate 的轨迹占比 | > 60% | < 30% = 模型不用工具 |
| truncation_rate | 被截断（达到 max_steps 且 reward=0）的轨迹占比 | < 20% | > 50% = max_completion_length 不够 |
| answer_coverage | 产生了可解析数字答案的轨迹占比 | > 60% | < 30% = 输出格式崩坏 |
| reward_variance | 同一步内各 generation 的 reward 标准差 | > 0.1 | < 0.02 = GRPO 无学习信号 |
| avg_agent_steps | 所有轨迹的平均交互轮次 | 1.5-2.5 | > 2.8 = 模型空转 |
| completion_length_mean | 平均 completion token 数 | < max_completion_length×0.7 | > max_completion_length×0.9 = 即将截断 |

**汇报格式**: 不再追加第二种行为指标行。行为指标必须并入主动轮询的固定单行 schema：

```
Step X/Y | R Z.ZZZ | Rstd Z.ZZZ | Loss L.LLLL | Grad G.GGGG | Ent E.EEEE | Clip C.CC | Len N | Finish P% | FmtOK P% | Tool P% | Ans P% | tok/s T.T | Time S.Ss | ETA ~Mm | Status OK/WARN/STOP
```
<!-- /section:behavioral-metrics -->

<!-- section:circuit-breaker-extension -->
### 行为指标熔断条件（C7-C9）

在现有 C0-C6 熔断基础上，新增 3 个基于行为指标的熔断条件。**每步完成后必须检查**。

- [ ] **C7 — 行为退化（格式遗忘）**: 连续 ≥3 步 finish_rate < 20%。模型忘记了如何调用 finish 提交答案，退化为只调 calculate 或纯文本输出。
  fix_preset: `diagnose`（需要完整分析诊断根因）

- [ ] **C8 — 无学习信号（GRPO 失效）**: 连续 ≥4 步 reward_variance < 0.02。同一步内所有 generation 得到几乎相同的 reward，GRPO 算不出 advantage，梯度为 0。这是 loss=0 的直接根因。
  fix_preset: `entropy_bonus`（lr/2 + temperature+0.2，增大探索多样性）

- [ ] **C9 — 输出崩坏**: 连续 ≥3 步 answer_coverage < 30%。模型输出不含任何可解析的数字答案，reward 函数无法给分。
  fix_preset: `diagnose`（可能是 max_completion_length 截断了 tool call 或模型完全退化）

- [ ] **C11 — 长度上限/截断**: `clipped_ratio >= 0.80` 或 `completion_length_mean >= max_completion_length * 0.90`。连续 2 个已完成 step 命中时，固定输出的 `Status` 必须写 `WARN(length-limit)`；连续 3 个已完成 step 命中时，触发熔断，`Status` 写 `STOP(length-limit)`。
  fix_preset: `increase_max_completion_length`（max_completion_length / max_response_length ×2，优先解决截断，再判断 lr 或探索问题）

**C8 与 C0 的关系**: C0 检测 loss=0（结果），C8 检测 reward_variance≈0（原因）。C8 比 C0 更早触发（reward_variance 趋近 0 先于 loss 完全归零），且能直接定位根因。

**fix_preset 新增映射**:

| 命中 Check | fix_preset | 含义 |
|---|---|---|
| C7 行为退化 | `diagnose` | 不跳过 Phase 5，走完整分析 |
| C8 无学习信号 | `increase_temperature` | temperature += 0.2 |
| C9 输出崩坏 | `diagnose` | 不跳过 Phase 5，走完整分析 |
| C11 长度上限/截断 | `increase_max_completion_length` | max_completion_length / max_response_length ×2 |
<!-- /section:circuit-breaker-extension -->

<!-- section:data-surfacing-extension -->
### 训练日志数据表面化

在现有数据表面化准则基础上，优先使用训练进程写入的 `[MONITOR_STEP]` 行：

| 时机 | 操作 | 工具 | 捕获内容 |
|------|------|------|---------|
| 每步完成时 | tail `training_log.txt` 并解析最新 `[MONITOR_STEP]` | Bash | reward/loss/Len/行为指标/状态 |
| `[MONITOR_STEP]` 缺失或异常诊断时 | Read `trajectories/step_XXXX.jsonl` | Read | 兜底模型行为数据（工具调用、答案、reward 分布） |

**频率**: 每检测到新 step 完成时只需要 tail 日志；不要默认每步 Read trajectory 文件。只有旧 run 没有 `[MONITOR_STEP]`、日志字段缺失、或需要完整错误诊断时，才读取最新 trajectory 文件。

**注意**: 这样可以做到每 step 输出 Len，同时减少 Claude Code 因每步 Read 大 JSONL 文件产生的权限确认。
<!-- /section:data-surfacing-extension -->

### Reward 峰值回落检测

在现有异常检测基础上增加:

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| Reward 峰值回落 | 当前 reward < 历史峰值 * 0.5 且持续 3 步 | 建议 early stopping |

实现: Monitor 维护 max_reward 变量，每步更新。当连续 3 步 reward < max_reward * 0.5 时发出 STOP 建议。

### Loss=0 检测

在现有异常检测基础上增加:

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| 训练无效果 (loss=0) | 连续 N 步 loss=0 且 reward > 0 | 报告 "训练无实际学习效果，模型预训练能力已覆盖当前难度" |

检测逻辑:
- 当连续 5 步 loss=0 且 avg_reward > 0.5 时触发
- 建议: "当前难度过低。建议提高难度 (mixed → mixed-hard) 或增加问题数量。"
- 不建议停止训练（reward 仍在达标），但标记为 "无学习信号"

轨迹证据:
- R3/R5: loss=0, avg_reward=0.77，模型参数未更新但 reward 达标
- 这种情况下继续训练是浪费时间，应提高难度获取学习信号

### 格式退化检测

在现有异常检测基础上增加:

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| 格式退化 | 连续 3+ 步: 所有 trajectory 使用 max_agent_steps 且 reward=0 | 建议 early stop |

检测逻辑:
- 当连续 3 步满足以下全部条件时触发:
  - 所有 trajectory 使用了 max_agent_steps (如 3 步)
  - 所有 trajectory reward=0
  - 模型重复调用同一工具而不调用 finish
- 告警: "检测到格式退化: 模型忘记了 finish 工具的使用方式。建议 early stop，下轮减少 num_problems 15-20%。"
- 与 "reward 归零" 检测的区别: 格式退化关注的是 agent 行为模式 (不调用 finish)，而非单纯的 reward 数值

轨迹证据:
- run_1777726900 step 41-43: 所有 trajectory 使用 3 步全部调用 calculate，从不调用 finish
- 与 step 12 不同 (step 12 正确使用了 calculate→finish 流程，只是答案错误)

| Loss 持续为零 | 连续 10 步 loss=0 且 step > total_steps * 0.25 | 报告: "Loss 持续为 0，GRPO 可能未产生有效梯度。如果 reward 高，说明任务太简单；如果 reward 低，检查 num_generations 和 temperature" |

### 7B 模型 GPU 训练异常检测（A100 80GB）

| 异常 | 检测方式 | 处理 |
|---|---|---|
| Response 长度撞限 | avg_response_len == max_completion_length 且 reward=0 | 报告，增大 max_completion_length 或检查模型是否陷入重复生成 |
| Grad norm 过高 | grad_norm > 100 | 报告，建议降低 lr 或添加 max_grad_norm clipping |
| Entropy 快速下降 | 4 步内 entropy 下降 > 70% | 警告过早收敛，建议提高 temperature |
| Reward 震荡 | 连续 4 步 reward 标准差 > 0.15 | 报告训练不稳定，检查 lr 和 batch_size |
| 单卡 OOM | CUDA out of memory + 模型 > 3B | 建议减小 batch_size/num_generations 或使用多卡 FSDP |
| 训练时间异常长 | 每 step rollout > 60s（7B 模型） | 检查是否在 CPU 上 fallback

### GRPO loss=0 异常检测

| 异常 | 检测方式 | 处理 |
|---|---|---|
| GRPO loss 持续为 0 | 连续 >= 4 步 loss=0.0 且 num_generations >= 3 | **严重异常**：即使 reward 有波动，loss=0 意味着策略梯度未生效。可能原因：(1) reward 传递方式有误，GRPOTrainer 未正确接收 per-completion reward；(2) rollout_func 返回的 logprobs 不正确；(3) env_mask 导致有效 token 过少。建议停止训练，检查 train.py 中 math_reward_fn 的返回值格式是否与 TRL GRPOTrainer 的期望一致。 |
| Loss=0 但 reward 上升 | loss=0 全程 + epoch 间 reward 有提升 | **警告**：reward 提升可能来自 base model 在不同 prompt 上的表现差异，而非 GRPO 学习效果。需要对比 base model 零样本 reward 和训练后 reward 来确认是否有真实提升。 |

### 熔断机制（Circuit Breaker）

当上述异常检测规则触发时，除了报告给用户，还需执行以下结构化输出流程。

#### 日志格式升级

rllm_train/logger.py 已升级，每 step 日志行包含 6 维指标：

```
Step  Trajs  Reward     Loss   Entropy   GradNorm   Rollout    tok/s      ETA
 1/16    64   0.250   0.8234    1.2045     0.3421    12.3s   150.2    2m30s
```

监控过程中，从每行汇总行提取以下指标（`—` 视为缺失）：
- `avg_reward`: Reward 列
- `loss`: Loss 列
- `entropy`: Entropy 列
- `grad_norm`: GradNorm 列
- `tok/s`: tok/s 列

#### 6 维熔断 Checklist

每次日志更新后，除了已有的异常检测规则，额外逐项检查：

- [ ] **C1 — 策略崩溃**: 最近 3 step avg_reward 全为 0，且之前存在连续 ≥2 step avg_reward > 0
- [ ] **C2 — 训练发散**: loss 连续 5 step 上升（严格单调递增），或出现 NaN/Inf
- [ ] **C3 — 梯度爆炸**: grad_norm > 100，或连续 3 step grad_norm 上升幅度 > 50%/step
- [ ] **C4 — 策略坍缩**: entropy 连续 3 step 下降，且最新值 < 0.01
- [ ] **C5 — 吞吐异常**: tok/s 骤降 > 80%（相对前 3 step 平均值），持续 ≥2 step
- [ ] **C6 — 无效启动**: 已完成 ≥3 step，所有 avg_reward 均 = 0，且 loss 波动 < 1%（max - min < 0.01 * avg）

与已有异常规则的关系：本 checklist 是补充，不替代已有的 early-stopping、loss-zero、format-degradation 等规则。如果已有规则先触发，按已有规则处理。

#### 熔断触发后的操作

当 C1-C6 任一命中时，执行以下步骤：

**Step 1: 中止训练进程**

```bash
# 使用 TaskStop 停止训练后台任务
```

**Step 2: 写精简 analysis.json**

在 `rllm_train/output/runs/<run_id>/` 下创建 `analysis.json`：

```json
{
  "completed": false,
  "abort_reason": "<命中的 check 编号>: <一句话描述>",
  "abort_step": "<当前 step>/<总 step>",
  "metrics_snapshot": {
    "reward_trend": [0.3, 0.35, 0.0, 0.0, 0.0],
    "loss_trend": [0.8, 0.7, 0.9, 1.2, 1.5],
    "entropy_trend": [1.2, 1.1, 0.8, 0.3, 0.01],
    "grad_norm_trend": [0.3, 0.4, 0.6, 1.2, 2.5],
    "tok_s_trend": [150, 148, 145, 30, 10]
  },
  "fix_preset": "<预设修复方案>",
  "skip_full_analysis": true
}
```

`metrics_snapshot` 中的 trend 数组取最近 5-10 step 的值（有多少填多少）。

**Step 3: 输出结构化信号**

向编排者输出以下格式的信号（必须是监控结束前的最后一行输出）：

```
=== CIRCUIT_BREAK ===
circuit_break: true
abort_reason: <C1-C6 编号>: <描述>
fix_preset: <预设方案>
analysis_json: rllm_train/output/runs/<run_id>/analysis.json
=== END_CIRCUIT_BREAK ===
```

#### fix_preset 映射表

| 命中 Check | fix_preset | 含义 |
|---|---|---|
| C1 策略崩溃 | `lr_half` | learning_rate ÷ 2 |
| C2 训练发散 | `lr_half` | learning_rate ÷ 2 |
| C2 (NaN) | `lr_tenth` | learning_rate ÷ 10 |
| C3 梯度爆炸 | `lr_half_grad_clip` | lr ÷ 2 + max_grad_norm=1.0 |
| C4 策略坍缩 | `entropy_bonus` | lr ÷ 2 + temperature + 0.2 |
| C5 吞吐异常 | `retry_same` | 配置不变，重试一次 |
| C6 无效启动 | `diagnose` | 不跳过 Phase 5，走完整分析 |

#### 未命中熔断时

正常完成训练时，输出标准完成报告（见 completion section），**不输出 CIRCUIT_BREAK 信号**。编排者通过"没有 CIRCUIT_BREAK 信号"判断训练正常完成，继续走 Phase 5。

## 训练完成检测

训练完成的标志：
1. 后台任务正常退出（exit code 0）
2. 日志中出现 "Training Report" 字样
3. `perf_stats.json` 文件生成

训练完成后，读取最终统计并汇报：

```
训练完成 [第 1 轮]:
  总耗时:     3m12s
  最终 Reward: 0.45 (从 0.25 开始)
  Reward 趋势: 0.25 → 0.31 → 0.38 → 0.45
  总 Steps:    16
  平均速度:    11.8 tok/s
```

## 数据表面化准则

Hooks 只捕获 Claude Code 工具调用的 input/response。Monitor 工具的 grep 输出不被 PostToolUse hook 记录为完整事件。为确保训练关键数据进入轨迹系统，监控过程中必须用 Read/Bash 工具明确读取以下数据:

| 时机 | 操作 | 工具 | 捕获内容 |
|------|------|------|---------|
| 训练启动后 | 读取 config.json 完整内容 | Read | 训练配置（lr, epochs, problems 等） |
| 训练过程中 | 定期 tail training_log.txt 最后 30 行 | Bash | reward 趋势、step 进度 |
| 异常发生时 | 读取完整错误段 | Read/Bash | 错误上下文、Traceback |
| 训练结束时 | 读取 perf_stats.json | Read | 性能统计 |
| 训练结束时 | tail training_log.txt 最后 50 行 | Bash | 最终 Training Report |

这些 Read/Bash 调用是 Monitor grep 的必要补充，不是替代。Monitor 负责实时通知，Read/Bash 负责将完整数据带入对话供 hooks 捕获。

### 定期 Tail 检查频率

除 Monitor 实时流式监控外，增加定期 tail 检查:
- 每 2 分钟或每 10 步（以先到者为准）执行一次 `tail -30 training_log.txt`
- 这确保训练数据通过 Read/Bash 工具被 hooks 捕获到轨迹中
- Monitor 的 grep 输出不被 PostToolUse hook 记录，因此 tail 检查是数据表面化的关键补充
