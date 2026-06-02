---
id: "003-difficulty"
target_section: "initial-config"
action: append
description: "增加 difficulty 配置指导，包括难度选择建议和调参时的难度调整规则"
source: "2026-04-30 训练实验, run_1777465401(太简单), run_1777512419(太难), run_1777516127(mixed最佳)"
created: "2026-04-30"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

### 难度配置

`difficulty` 只控制 rllm_train 的内置合成数据生成器 `generate_math_problems()`，适用条件是 `dataset_path` 为空且未使用外部数据集。

合成数据下的含义:
- `"simple"`: 100% 简单两数运算 (适合流程验证)
- `"hard"`: 100% 多步骤应用题
- `"mixed"`: 80% simple + 20% hard

外部数据集下的规则:
- 当 `dataset_path` 非空时，`difficulty` 不会改变样本分布；训练代码只执行 `load_from_disk(dataset_path)`、shuffle、select。
- 禁止因为旧 run 的 `difficulty` / reward 经验修改外部数据集配置，除非该数据集已有明确难度字段且当前代码实现了过滤逻辑。
- 对 DeepScaler 或自定义 HuggingFace Dataset，调参只能依据当前 run 的 `analysis.json`、日志和轨迹指标；不要套用合成数据的 `simple/mixed/hard` 经验。

初始配置推荐（仅合成数据）:

| 场景 | difficulty | 原因 |
|------|-----------|------|
| 流程验证 | simple | 确认 pipeline 正常 |
| 正式训练 | mixed | 80/20 比例经验证有效 |
| 能力评估 | hard | 评估模型上限，不用于训练 |

调参时的难度调整（仅合成数据）:
- reward=1.0 + loss=0 → 题目太简单，切换到 mixed 或 hard
- reward<0.1 + difficulty=hard → 太难，切换到 mixed
- mixed 下 reward 在 0.3-0.7 → 比例合适，保持不变
