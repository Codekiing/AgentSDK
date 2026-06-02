---
id: "007-robust-monitor-agent"
target_section: "monitoring-methods"
action: append
description: >-
  用 skill-bank/rllm/rllm-monitor/monitor_agent.py 替代内联 Python 轮询脚本，支持 inode 检测、文件截断恢复、
  按 step 编号去重、自动退出。增加 CronCreate 兜底：Monitor 被 kill 后每 2 分钟 oneshot tail 补救。
source: "2026-05-27 链路问题诊断: Monitor 频繁被 kill, 日志截断后 pos 错位, step 去重失效"
created: "2026-05-27"

depends_on:
  - "006-step-event-echo"
conflicts_with: []

status: active
superseded_by: ""
---

### 监控脚本标准化

不再在 Monitor 工具中内联 Python 脚本。统一使用 rllm-monitor skill 自带的 `monitor_agent.py`（位于 `skill-bank/rllm/rllm-monitor/`）：

**流式监控 (stream 模式)**:
```bash
python skill-bank/rllm/rllm-monitor/monitor_agent.py stream rllm_train/output/runs/<run_id>/training_log.txt --timeout 3600
```

内建保护:
- **inode 检测**: 日志文件被重建 (inode 变化) 时自动重置 pos=0, 清空 seen_steps
- **文件截断恢复**: current_size < pos 时重置
- **按 step 编号去重**: 提取 `Step (\d+)/` 而非文本 key, 日志重现后不丢 step
- **Training Report 自动退出**: 检测到后立即 exit, 不继续等待
- **10 分钟无输出警告**: 避免训练卡死后 monitor 静默

**兜底轮询 (oneshot 模式)**:
```bash
python skill-bank/rllm/rllm-monitor/monitor_agent.py oneshot rllm_train/output/runs/<run_id>/training_log.txt --last 3
```

输出最近 3 个 MONITOR_STEP, 或终端事件 (Training Report/Traceback)。

### CronCreate 双重监控策略

主监控: Monitor 工具, persistent=false, timeout=3600000ms, 使用 stream 模式
备用监控: CronCreate, 每 2 分钟执行一次 oneshot 模式

编排层执行逻辑:
  1. 启动 Monitor (stream 模式) — 如果启动失败, 降级为 CronCreate oneshot
  2. 同时设置 CronCreate("*/2 * * * *", oneshot 模式, recurring=true)
  3. 每收到 Monitor 事件 (step) → 逐条回显, 更新 last_event_time
  4. 收到 Training Report / Traceback / OOM → 停止 Monitor + CronCreate 清理
  5. Monitor 超时退出且训练未完成 → CronCreate 仍在运行, 自动补位
  6. CronCreate 事件中包含 MONITOR_STEP → 同样逐条回显
  7. CronCreate 事件中包含 DETECTED: Training Report → 停止 CronCreate, 进入 Phase 5

错误模式（禁止）:
- 只启动 Monitor 不设 CronCreate 兜底
- Monitor 被 kill 后不做任何补救等用户说"继续"
- 在 oneshot 输出重复步骤时重新回显已汇报的 step
