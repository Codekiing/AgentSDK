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
- **主动轮询 (Active Polling)**: 编排层每隔固定间隔执行 `tail -30 training_log.txt`，提取并汇报关键指标给用户
- **被动等待 (Passive Waiting)**: 启动一个后台 while 循环或 background task 等待训练完成，期间不做任何 tail 或汇报

**强制规则**:

1. **主动轮询是唯一的监控手段**。编排层 MUST 在训练过程中持续执行:
   ```bash
   tail -30 rllm_train/output/runs/<run_id>/training_log.txt
   ```
   间隔: 每 2 分钟 (120s) 或每观察到新 step 完成时，取先到者。

2. **禁止被动等待**。编排层 MUST NOT:
   - 创建 `while true; do sleep; done` 后台循环来等待训练完成
   - 仅依赖 `TaskOutput` 阻塞等待而不做中间汇报
   - 在任何 step 之后停止 tail 检查
   - 将"等待训练完成"作为唯一的监控策略

3. **每次轮询必须汇报**。每个 tail 检查后，编排层 MUST 向用户输出一行训练状态摘要，格式:
   ```
   Step X/Y | Reward: Z.ZZ | Loss: L.LL | 速度: XX tok/s | ETA: ~XXm
   ```
   即使没有新 step 完成（日志无变化），也要报告"训练中，无新 step 输出"。

4. **轮询不可中断**。只有以下情况可以停止主动轮询:
   - 训练已完成（检测到 "Training Report" 或后台进程退出）
   - 检测到严重异常（OOM、崩溃）需要停止训练
   - 用户明确要求停止

5. **后台进程是辅助，不是替代**。训练进程可以后台运行（`run_in_background: true`），但后台进程状态检查只是主动轮询的补充信息来源，不能替代 tail 日志检查。

#### 执行模式

正确的监控循环 (伪代码):

```
train_process = Bash("torchrun ...", run_in_background=true)
last_step = 0

while 训练未完成:
    # 1. 主动读取日志
    log_output = Bash("tail -30 training_log.txt")
    current_step = extract_step(log_output)

    # 2. 如果有新 step，立即汇报
    if current_step > last_step:
        向用户汇报训练进度
        last_step = current_step

    # 3. 检查异常
    check_anomalies(log_output)

    # 4. 检查训练是否完成
    if "Training Report" in log_output or 训练进程已退出:
        break

    # 5. 等待下一轮轮询
    ScheduleWakeup(120s) 或 sleep 120s
```

错误的监控模式 (禁止):

```
# 错误: 只启动后台等待，不做中间汇报
train_process = Bash("torchrun ...", run_in_background=true)
TaskOutput(train_process, block=true, timeout=600000)  # 阻塞等待，中间无输出
# 用户在 30 分钟内看不到任何训练状态
```

## 汇报内容

### 进度汇报（每 N 步或用户询问时）

```
训练进度 [第 1 轮]:
  进度:     Step 8/128 (6%)
  Reward:   0.750 (趋势: ↑ 从 0.25 开始)
  速度:     88.4 tok/s, 每步 ~7s
  已用时间: 1m45s
  预计剩余: 25m30s
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

### 熔断检测标准 (Circuit Breaker Detection)

Monitor 是熔断检测的**唯一责任方**。编排层 (rllm-train) 不自行判断熔断，只根据 Monitor 发出的 STOP 信号执行中止流程。

#### 熔断信号格式

当 Monitor 判定需要熔断时，输出结构化 STOP 信号供编排层解析:

```
CIRCUIT_BREAK | severity=<L1|L2|L3> | abort_reason=<reason_code> | fix_preset=<preset_name>
```

| 字段 | 说明 |
|------|------|
| severity | L1=自动恢复, L2=建议恢复, L3=需深度分析 |
| abort_reason | 短标识符 (如 `OOM`, `REWARD_ZERO`, `FORGETTING`, `FORMAT_DEGRADATION`) |
| fix_preset | 预设修复方案名 (如 `halve_lr`, `reduce_problems`, `increase_gen`) |

#### 熔断触发条件与分级

##### L1: 自动恢复 (Auto-Recoverable)

无需深度分析，编排层直接应用 fix_preset 后重新训练。

| 条件 | 检测逻辑 | fix_preset | abort_reason |
|------|---------|-----------|-------------|
| OOM | 日志含 "out of memory" | `halve_completion_length` | `OOM` |
| num_generations 不整除 | 启动失败 + ValueError + "not divisible" | `adjust_generations` | `GEN_DIVISIBILITY` |
| 进程崩溃 (非 OOM) | Traceback + 进程退出 | `none` (编排层读取错误诊断) | `TRACEBACK` |

##### L2: 建议恢复 (Advised Recovery)

需要分析但不需要完整 15 层诊断。编排层应快速分析后决定。

| 条件 | 检测逻辑 | fix_preset | abort_reason |
|------|---------|-----------|-------------|
| lr 过高致策略崩溃 | reward 从 >0.3 骤降到 0 且连续 3 步不恢复 | `halve_lr` | `LR_CRASH` |
| catastrophic forgetting | Epoch 切换后 reward < 前一 epoch * 0.3 | `reduce_epochs` | `FORGETTING` |
| grad_accum 副作用 | 训练从第 1 步起 reward=0 且持续 >= 4 步 | `revert_grad_accum` | `GRAD_ACCUM` |

##### L3: 需深度分析 (Deep Analysis Required)

触发后编排层必须调用 `rllm-analyze-deep` (circuit_break=true)，不可用普通 rllm-analyze。

| 条件 | 检测逻辑 | fix_preset | abort_reason |
|------|---------|-----------|-------------|
| 格式退化 | 连续 3+ 步: 所有 trajectory 使用 max_agent_steps 且 reward=0，模型不调用 finish | `reduce_problems` | `FORMAT_DEGRADATION` |
| Reward 双峰/震荡 | 连续 6+ 步 reward 标准差 > 0.2 且无明显趋势 | `stabilize_training` | `REWARD_OSCILLATION` |
| Loss=0 异常 | 连续 >= 4 步 loss=0.0 且 num_generations >= 3，reward 无提升 | `increase_difficulty` | `LOSS_ZERO` |
| 后半段退化 | 后 50% steps avg_reward < 前 50% * 0.7 且非首次出现 | `reduce_problems` | `LATE_DEGRADATION` |
| KL 散度异常 | KL 持续 > 0.5 或持续 < 0.01 (如果日志含 KL 值) | `adjust_lr` | `KL_ANOMALY` |
| GRPO advantage 为零 | 连续 >= 3 步 advantage_std < 0.01 (如果日志含 advantage 值) | `increase_generations` | `ADVANTAGE_ZERO` |

#### 熔断变更历史

| 日期 | 变更 | 原因 |
|------|------|------|
| 2026-05-31 | 新增熔断检测标准，L1/L2/L3 分级，STOP 信号结构化 | 将熔断逻辑从 rllm-train 编排层迁移到 rllm-monitor，实现职责分离 |
| 2026-05-31 | 新增 `CIRCUIT_BREAK` 信号格式 | 编排层不再自行判断熔断原因，由 Monitor 提供 fix_preset |

#### 设计原则

1. **Monitor 负责检测 + 分级**，编排层只负责执行中止流程和路由到正确的分析器
2. L1/L2 可由编排层直接处理（应用 fix_preset 后回到 Phase 2）
3. L3 必须经过 rllm-analyze-deep 深度分析，不可跳过
4. 编排层不自行判断"这是什么异常"——异常类型由 Monitor 通过 STOP 信号告知
5. 同一 run 内连续 2 次 L3 熔断 → Monitor 发出 `CIRCUIT_BREAK | severity=L3 | abort_reason=CONSECUTIVE_L3`，编排层强制停止

### Epoch 边界监控

计算 epoch 边界: `epoch_boundary = num_problems / (batch_size * grad_accum)`

当 step 跨越 epoch 边界时:
  读取最近 3 步的 reward
  与上一个 epoch 最后 3 步的 reward 对比
  如果下降 > 50%: 发出 catastrophic forgetting 预警

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
