---
id: "005-stateful-orchestration"
target_section: "state-tracking"
action: replace
description: >-
  让 rllm-train 编排层在每个 Phase 完成后使用 rllm_train/state_manager.py 原子写入
  training_state.json，从对话上下文依赖改为文件持久化。同时修复自动续跑：子 skill 完成后
  编排者必须从 training_state.json 恢复状态并自动进入下一 Phase。
source: "2026-05-27 链路问题诊断: 对话压缩丢失状态, 自动续跑无执行者, training_state.json 从未使用"
created: "2026-05-27"

depends_on:
  - "004-auto-continue-after-config"
conflicts_with: []

status: active
superseded_by: ""
---

### 状态持久化规则

**强制**: 编排者在每个 Phase 转换点必须通过 `rllm_train/state_manager.py` 原子写入 training_state.json。

#### Phase 到 state 的映射

| 时机 | 写入函数 | 关键字段 |
|------|---------|---------|
| 首次训练启动 | `state_manager.init(task_id, skill_package_id)` | task_id, round=1, current_phase=clarify |
| Phase 1 完成 | `state_manager.update_phase("config")` | current_phase=config |
| Phase 2 完成 | `state_manager.update_phase("run", run_id="xxx")` | current_phase=run, current_run_id |
| Phase 3 完成 | `state_manager.update_phase("monitor", run_id="xxx")` | current_phase=monitor |
| Phase 4 完成 (正常) | `state_manager.update_phase("analyze")` | current_phase=analyze |
| Phase 4 完成 (熔断) | `state_manager.increment_circuit_break()` | circuit_break_count +1 |
| Phase 5 完成 (未达标) | `state_manager.record_round_result(...)` | history 追加, round +1 |
| Phase 5 完成 (达标) | `state_manager.mark_completed()` | completed=true |
| Phase 6 完成 | `state_manager.mark_completed()` | completed=true |

写入方式:
```bash
python -c "
from rllm_train.state_manager import update_phase
update_phase('config')
"
```

#### 自动续跑实现

编排者调用子 skill 后当轮停止。下一轮开始时:

1. **首先读取 training_state.json**:
   ```bash
   python -c "
   from rllm_train.state_manager import get_next_phase
   import json
   print(json.dumps(get_next_phase()))
   "
   ```

2. **根据 current_phase 决定下一步**:
   | current_phase | 编排者动作 |
   |---|---|
   | clarify | Skill("rllm-clarify") |
   | config | 读取 config.json, 提取 run_id, 然后 Skill("rllm-run") |
   | run | Skill("rllm-run", args=run_id) |
   | monitor | Skill("rllm-monitor", args=run_id) |
   | analyze | Skill("rllm-analyze", args=run_id) |
   | analyze_complete | 读取 analysis.json, 判断停止条件 |

3. **熔断快速路径**:
   - 读取 `circuit_break_count`, 若 >=2 走完整 Phase 5
   - 从 analysis.json 读 fix_preset, 传给 rllm-config

4. **禁止等待用户输入"继续"**: 只要 `current_phase` 不是 `clarify` 或 `complete`,
   编排者必须立即进入对应 Phase, 不得输出"是否继续?"等待用户确认。

#### 恢复流程

编排者每轮开始时的标准恢复步骤:

```
1. 读取 training_state.json
2. 如果 current_phase != "clarify" 且 != "complete":
   → 说明上次未完成, 从 current_phase 继续
3. 如果 current_phase == "run" 或 "monitor":
   → 检查后台训练进程是否仍在运行
   → 如果在运行: 恢复监控
   → 如果已退出: 读日志判断完成/崩溃, 进入对应 Phase
4. 如果 completed == true:
   → 进入 Phase 6 输出最终报告
```

错误模式（禁止）:
- Phase 转换后不写 training_state.json, 只靠对话上下文记忆
- 下一轮开始时不读 training_state.json, 直接假设从头开始
- 子 skill 完成后输出"配置已生成, 是否继续?"而不自动进入下一 Phase
