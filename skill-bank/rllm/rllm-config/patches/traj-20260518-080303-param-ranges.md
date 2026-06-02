---
id: traj-20260518-080303-param-ranges
target_section: param-ranges
action: append
description: "7B 模型 GRPO 训练关键发现：num_generations 必须 >= 4 才能产生有效梯度信号"
status: proposed
source: trajectory-analysis
source_sessions: ["de0bd17e-d38c-4a96-9c1b-ab5bd4028ed3", "b16e9f00-3909-4d98-86d8-fff1ea282539"]
---

### 7B 模型 GRPO 关键约束（A100 80GB 单卡）

| 参数 | 7B 推荐值 | 原因 |
|------|----------|------|
| num_generations | >= 4（最低 3） | 2 个 completion 时 GRPO advantage 常为 0, 无有效梯度, loss=0 |
| difficulty | 不跨数据集自动修改 | `difficulty` 只对合成数据生效；外部数据集必须依据当前 run 的日志/轨迹/analysis 调参，不得套用历史 mixed/hard reward 经验 |
| batch_size | 1 | 单卡显存限制, batch=2 + gen=4 会 OOM |
| learning_rate | 2e-6 ~ 5e-6 | 7B 模型需要较低的 lr 避免 grad norm 爆炸 |

#### 显存预估

| num_generations | batch_size | 预计显存 | 是否可行 |
|----------------|-----------|---------|---------|
| 2 | 1 | ~30GB | 可行但无学习效果 |
| 4 | 1 | ~50GB | 可行（推荐） |
| 4 | 2 | ~75GB | 临界, 可能 OOM |
| 2 | 2 | ~55GB | 可行但无学习效果 |
