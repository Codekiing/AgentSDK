---
id: "007-dataset-scope-isolation"
target_section: "tuning"
action: prepend
description: "按数据源隔离训练经验，防止历史 run 和合成数据规则影响外部数据集训练"
source: "2026-05-25 用户反馈: DeepScaler 外部数据集不应套用 simple/mixed/hard 和历史 reward=0.773 经验"
created: "2026-05-25"

depends_on:
  - "003-difficulty"
conflicts_with: []

status: active
superseded_by: ""
---

### 数据源作用域隔离（必须优先执行）

生成或调参配置前，先判定数据源作用域:

| 条件 | 数据源类型 | 可使用的经验规则 |
|---|---|---|
| `dataset_path` 为空 | 内置合成数据 | 可使用 simple/mixed/hard、0.5B mixed problem-count、difficulty scaling 等合成数据经验 |
| `dataset_path` 非空 | 外部 HuggingFace Dataset | 不使用合成数据 difficulty 经验；只根据当前 run 的 config/log/trajectory/analysis 调参 |
| `dataset="deepscaler"` 或路径含 deepscaler | DeepScaler 外部数据集 | 不使用 simple/mixed/hard 调参；只调整通用训练参数和当前样本选择参数 |

硬性规则:
1. 外部数据集下，`difficulty` 不展示、不写入新配置，也不会改变 `load_external_dataset()` 的样本分布。除非训练代码已实现基于数据集字段的过滤，否则不得声称 mixed/hard 会改变 DeepScaler 难度。
2. 历史 run 的 reward、loss、forgetting、plateau 结论只能用于相同数据源、相同模型量级、相同 reward 函数和相同训练代码路径。缺少任一条件时，只能作为背景风险提示，不能直接驱动参数选择。
3. 禁止在外部数据集配置中引用固定历史指标（例如 “mixed reward 已达 0.773”）作为切换 difficulty 或判断任务太简单的依据。
4. 调参循环必须优先读取当前 run 的 `analysis.json`。若 `analysis.json` 与 skill 中的历史经验冲突，以当前 run 为准。
5. `training_state.json` 只能作为当前训练会话的历史摘要，不得把其它目标、其它数据集或已标记 completed 的旧会话当作本次训练轮次依据。

DeepScaler/外部数据集可调参数优先级:
1. 进程异常/OOM/静默退出 → 降低 `num_problems` 或生成长度，保留可复现实验规模。
2. reward_variance 长期接近 0 → 增大 `temperature` 或 `num_generations`，但保持 TRL 整除约束。
3. calculator_error 高 → 不通过 difficulty 调参；应降低对工具不可执行题型的依赖，或在数据预处理/环境工具能力上处理。
4. completion 接近上限或 clipped_ratio 高 → 增大 `max_completion_length` / `max_response_length`。
5. reward 震荡且 grad_norm/loss 异常 → 降低 `learning_rate` 或增加稳定性约束。
