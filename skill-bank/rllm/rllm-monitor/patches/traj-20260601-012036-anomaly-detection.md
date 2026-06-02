---
id: traj-20260601-012036-anomaly-detection
target_section: anomaly-detection
action: append
description: Monitor心跳超时检测：训练异常时的静默告警
status: proposed
source: trajectory-analysis
source_sessions: ["4d86a9e2-7985-4779-bdae-5b8e01b3edc8"]
---

### 训练静默超时检测

**症状**: 训练进程异常但无明显错误输出时(如R2 vLLM截断)，Monitor长时间静默无告警。

**检测规则**:
- 每30秒检查训练进程是否存活
- 超过120秒无reward更新时输出 `[WARN]` 级别告警
- 检测到进程退出但无正常结束标记时主动报告进程状态

**R2 案例**: vLLM因max_response_length超限截断所有输出，训练仅完成1步即停止，但Monitor未及时发现。

