---
id: "001-linux-server"
target_section: "steps"
action: append
description: "Linux 服务器启动方式：检测无桌面环境时，输出指令引导用户在新终端手动启动交互式 Claude Code"
source: "手动补充，解决 Linux 服务器环境兼容性"
created: "2026-05-18"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

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
