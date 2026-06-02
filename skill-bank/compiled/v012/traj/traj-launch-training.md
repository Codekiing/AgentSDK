---
description: Launches a new CLI-1 session for rllm-train execution. Supports interactive
  mode (new Terminal window via osascript) and non-interactive mode (claude -p background).
  Ensures each training task gets a fresh session_id.
metadata:
  categories:
  - trajectory
  - orchestration
  version: 1.0.0
name: traj-launch-training
---


# traj-launch-training — 启动训练 CLI

在 CLI-2 中调用，自动创建新的 Claude Code 进程执行训练任务。

每次启动都是新进程 = 新 session_id，天然满足"CLI-1 每次新建"约束。

## 参数

- round: 轮次号（必需，或 "next" 自动计算）
- 训练描述: 传给 rllm-train 的描述
- --auto: 非交互模式，后台执行

示例:
```
/traj-launch-training round=1 | 用 qwen-0.5b 训练, reward >= 0.8
/traj-launch-training next | 用 qwen-0.5b 训练, reward >= 0.8
/traj-launch-training round=1 --auto | 用 qwen-0.5b 训练, reward >= 0.8
```

## 执行步骤

### 1. 解析参数

提取 round 号、训练描述、是否 --auto。

如果 round=next:
```python
from traj_opt.round_state import RoundState
round_num = RoundState().find_pending_training()
if round_num is None:
    输出 "没有待训练的轮次。上一轮可能尚未完成优化。"
    退出
```

### 2. 前置检查

- 如果 round > 1，检查上一轮 optimization_complete:
  ```python
  from traj_opt.round_state import RoundState
  rs = RoundState()
  prev = rs.read_status(round_num - 1)
  if not prev or prev.get("status") != "optimization_complete":
      输出 "Round {round_num - 1} 尚未完成优化，请先执行 /traj-train-optimize round={round_num - 1}"
      退出
  ```
- 检查当前 round 是否已有 status（避免重复启动）:
  ```python
  current = rs.read_status(round_num)
  if current:
      输出 "Round {round_num} 已存在 (status={current['status']})，跳过"
      退出
  ```
- 确认 `.claude/skills/rllm-train/SKILL.md` 存在

### 3. 启动训练

准备日志目录:
```bash
mkdir -p traj_opt/output/rounds/round_{N}
```

#### 交互式（默认）

用 osascript 打开新 Terminal 窗口，通过 heredoc 避免引号嵌套问题:
```bash
PROJECT_DIR=$(pwd)
osascript << ENDSCRIPT
tell application "Terminal"
    activate
    set projectDir to "$PROJECT_DIR"
    set trainCmd to "claude \"/rllm-train round={N} | {描述}\""
    do script "cd " & projectDir & " && " & trainCmd
end tell
ENDSCRIPT
```

#### 非交互式（--auto）

后台启动 claude -p:
```bash
claude -p --permission-mode auto \
  "/rllm-train round={N} | {描述}" \
  > traj_opt/output/rounds/round_{N}/cli1.log 2>&1 &
echo $! > traj_opt/output/rounds/round_{N}/cli1.pid
```

### 4. 输出

交互式:
```
Round {N} 训练已在新终端窗口中启动。
训练描述: {描述}

训练完成后回到此 CLI 执行:
  /traj-train-optimize round={N}
```

非交互式:
```
Round {N} 训练已在后台启动:
  PID:  {pid}
  日志: traj_opt/output/rounds/round_{N}/cli1.log

等待训练完成后执行:
  /traj-train-optimize round={N}
```

### Linux 服务器（无桌面环境）

检测运行环境:
```bash
uname -s
```

如果是 Linux 且没有 osascript 命令，采用手动引导模式:

1. 输出以下指令块，让用户在另一个终端中手动执行:

```
============================================
请在另一个终端中执行以下步骤:
============================================

1. 打开新终端，进入项目目录:
   cd {PROJECT_DIR}

2. 激活 Python 虚拟环境（如果项目使用 venv）:
   source test/bin/activate

3. 启动 Claude Code:
   claude

4. 在 Claude Code 中输入:
   /rllm-train round={N} -- {描述}

训练完成后（看到 Training Report），回到当前终端执行:
  /traj-train-optimize round={N}
============================================
```

2. 用 AskUserQuestion 等待用户确认已在另一个终端启动训练:
   - 问题: "是否已在新终端中启动训练？"
   - 选项: "已启动" / "需要帮助"

3. 用户确认后输出:
```
Round {N} 等待训练完成。
训练完成后在当前 CLI 执行: /traj-train-optimize round={N}
```
