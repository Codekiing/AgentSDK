---
id: "verl-backend-routing"
target_section: "phase0"
action: append
description: "Add VERL backend detection in Phase 0: auto-detect VERL for 7B+ models, route training to VERL-specific config/launch/monitor when backend=verl"
source: "2026-05-28 VERL backend integration"
created: "2026-05-28"

depends_on:
  - "remote-backend-routing"
  - "rllm-config:008-verl-config-generation"
  - "rllm-run:001-verl-launch"
  - "rllm-monitor:verl-log-patterns"
conflicts_with: []

status: active
superseded_by: ""
---

### VERL Backend 检测

在 Remote Backend 检测之后，额外检查 VERL 训练关键词：

| 关键词 | 含义 |
|---|---|
| `verl`, `VERL`, `verl backend` | 显式使用 VERL 后端 |
| 模型 >= 7B (如 qwen-7b, qwen-14b, llama-7b 等) | 自动检测为 VERL（7B+ 默认走 VERL） |
| `trl`, `TRL`, `trl backend` | 显式指定 TRL 后端（覆盖自动检测） |

优先级规则（从高到低）：
1. 显式 `trl`/`TRL` 关键词 → backend=trl（向后兼容，即使 7B+ 也走 TRL）
2. 显式 `verl`/`VERL` 关键词 → backend=verl
3. 模型 >= 7B → backend=verl（自动检测）
4. 默认 → backend=trl

### Phase 路由（VERL 模式）

当 `backend=verl` 时，Phase 映射变更：

| Phase | 本地-TRL (默认) | VERL 模式 (backend=verl) |
|---|---|---|
| Phase 1 (需求澄清) | rllm-clarify | rllm-clarify (不变) |
| Phase 2 (配置生成) | rllm-config | rllm-config (传入 backend=verl) |
| Phase 3 (启动训练) | rllm-run | rllm-run (传入 backend=verl) |
| Phase 4 (过程监控) | rllm-monitor | rllm-monitor (传入 backend=verl) |
| Phase 5 (结果分析) | rllm-analyze-deep | rllm-analyze-deep (传入 backend=verl) |
| Phase 6 (最终报告) | 编排者执行 | 编排者执行 (标注 VERL 后端) |

### VERL 模式数据传递

```
Phase 2 → Phase 3: config.json + run_verl.sh (均在 rllm_train/output/runs/<run_id>/)
Phase 3 → Phase 4: run_id + 后台 task ID + training_log.txt (rllm_train/output/runs/<run_id>/training_log.txt)
Phase 4 → Phase 5: run_id + 训练完成确认
Phase 5 → Phase 2: deep_analysis.json (含 VERL 专属指标和调参建议)
```

### Phase 2 (VERL): 配置生成

**调用子 skill: rllm-config**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-config", args="初始配置 | backend=verl | <Phase 1 的需求摘要>")`
   - 调参循环: `Skill("rllm-config", args="调参 | backend=verl | run_id=<run_id> | <调参建议>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-config/SKILL.md`

输入: 需求摘要 + backend=verl 标记
输出: config.json + run_verl.sh (均在 rllm_train/output/runs/<run_id>/)
完成标志: config.json 和 run_verl.sh 均已生成

### Phase 3 (VERL): 启动训练

**调用子 skill: rllm-run**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-run", args="<run_id>")`
   - rllm-run 从 config.json 读取 backend 字段，自动选择 VERL 启动路径
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-run/SKILL.md`

输入: run_id
输出: 后台 task ID + training_log.txt 路径
完成标志: Ray 集群初始化成功，VERL 训练进程已启动

### Phase 4 (VERL): 过程监控

**调用子 skill: rllm-monitor**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-monitor", args="<run_id>")`
   - rllm-monitor 从 config.json 读取 backend 字段，使用 monitor_agent_verl.py
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-monitor/SKILL.md`

输入: run_id + 后台 task ID
输出: 训练完成确认 / CIRCUIT_BREAK

Monitor 使用 VERL 专用日志解析器，检测：
- Ray worker 输出格式的奖励/损失指标
- 训练步数进度 (从 metric 行推断)
- VERL 特有的 OOM (Ray 资源不足)、vLLM 异常
- 进程退出 / Ray cluster 关闭作为训练完成标志

<!-- section:verl-routing -->
以上 VERL 路由规则自动生效。编排者检测到 backend=verl 后，按上表路由到对应子 skill。
<!-- /section:verl-routing -->
