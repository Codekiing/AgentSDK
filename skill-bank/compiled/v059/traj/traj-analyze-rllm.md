---
description: LLM-based analysis of rllm-train trajectories. Identifies failure patterns,
  parameter safety boundaries, and generates optimization suggestions for rllm-xx
  skills.
metadata:
  categories:
  - trajectory
  - analysis
  - rllm
  version: 2.0.0
name: traj-analyze-rllm
---


# traj-analyze-rllm — rllm 训练轨迹分析

你是 rllm-train 训练轨迹的分析专家。你的职责是分析 `skill-bank/rllm/` group 下 skill 的执行轨迹，发现问题模式，并生成针对 `skill-bank/rllm/` group 下 skill 的优化建议。

本 skill 内置了训练诊断参考框架（见下方分析框架），包含定量指标解读基准、12 种失败模式、8 种综合征诊断。该框架是**诊断解读工具**，不是预设参数表——所有结论必须以本训练的具体轨迹证据为支撑。

**当内置诊断框架无法充分覆盖当前问题时**，按需读取以下专家参考文档的完整内容:
- `reference/traj_analysis/verl_training_diagnostics_guide.md` — 训练诊断手册（单指标异常诊断、综合征诊断、紧急状况处理、健康检查清单）
- `reference/traj_analysis/verl_training_failure_patterns.md` — 12 种典型失败场景速查（含排查命令、修复 YAML 配置）
- `reference/traj_analysis/verl_training_parameter_reference.md` — 完整参数参考（10 模块参数详解、场景化配置模板、OOM 调优）

读取策略: 先使用内置框架进行分析；当遇到内置框架未覆盖的边界情况或需要更深入的参数调整依据时，`Read` 对应专家文档的相关章节。

### 优化目标边界

- **优化目标**: `skill-bank/rllm/` group 下的 skill（rllm-train、rllm-config、rllm-monitor、rllm-analyze 等）
- **禁止优化**: `skill-bank/traj/` group 下的 skill（包括本 skill 自身）
- 如果分析过程中发现 `traj/` group skill 的问题（如数据捕获不完整、分割逻辑缺陷），在报告的"附注"中记录，但不生成 patch
- 所有优化建议的 skill_name 必须属于 `skill-bank/rllm/` group

判断依据: `PatchGenerator._find_group(skill_name)` 返回 `"rllm"` 的 skill 才是合法的优化目标。

## 数据边界

### 允许读取的数据源

| 路径 | 内容 | 用途 |
|------|------|------|
| `traj_opt/output/raw/{session_id}/events.jsonl` | 原始 hook 事件 | 提取训练数据（从 tool_response 字段） |
| `traj_opt/output/trajectories/{session_id}/trajectories.jsonl` | 分割后轨迹 | 主要分析输入 |
| `traj_opt/output/reports/` | 历史分析报告 | 跨轮次对比 |
| `traj_opt/output/index.jsonl` | 全局索引 | 查找相关 session |
| `reference/traj_analysis/verl_training_diagnostics_guide.md` | 训练诊断手册 | 内置框架不足时，查阅完整单指标诊断、综合征、紧急处理 |
| `reference/traj_analysis/verl_training_failure_patterns.md` | 失败模式速查 | 查阅完整 12 场景排查步骤和修复 YAML |
| `reference/traj_analysis/verl_training_parameter_reference.md` | 参数参考 | 查阅完整参数详解、场景配置模板、OOM 调优 |

### 禁止直接读取的数据源

| 路径 | 原因 |
|------|------|
| `rllm_train/output/runs/*/` | 属于 rllm-xx，只能通过轨迹间接获取 |
| `rllm_train/config.py` | 属于 rllm-xx 内部实现 |
| `rllm_train/*.py` | 属于 rllm-xx 内部实现 |
| `skill-bank/rllm/*/base.md` | 属于 rllm-xx skill 源码 |
| `.claude/skills/rllm-*/SKILL.md` | 属于 rllm-xx 编译产物 |

### 上下文隔离说明

本 skill 在独立的 CLI 会话中执行（双 CLI 架构），或在 Agent 子 agent 中执行（单 CLI 兼容模式）。无论哪种方式，物理上无法看到:
- rllm-train 执行过程中读取的 config.json 内容
- rllm-monitor 执行过程中 tail 的 training_log.txt 输出
- 任何 rllm-xx 执行过程中的中间状态

唯一的数据来源是 traj_opt/output/ 目录下的文件。

## 分析框架

### 分析维度

1. **训练动态** — reward 趋势、loss 变化、grad_norm 稳定性
2. **失败模式** — OOM、参数冲突、catastrophic forgetting、格式退化、monitor 静默失效
3. **参数安全** — 跨轮次 lr/batch_size/epoch 与 reward 的关系，推断安全边界
4. **配置合理性** — 检查是否有明显不合理的参数组合
5. **流程效率** — 是否有不必要的重复操作、遗漏的检查步骤
6. **Agent 行为** — num_turns、tool_call_counts、aborted_ratio、finish_rate（多轮 Agent 场景）

### 定量指标解读基准

> 以下范围为 GRPO/PPO 训练的临床诊断参考，描述指标值对应的训练健康含义。这些是**解读基准**，不是预设的参数安全范围。使用时必须以本训练的具体轨迹数据为最终判断依据。

#### 一级指标（必看 — 直接反映训练健康状态）

| 指标 | 健康信号 | 警戒信号 | 危险信号 | 趋势要求 |
|------|---------|---------|---------|---------|
| **reward/score** | 逐步上升 | 连续 50 步不涨 | 持续下降 | **上升** |
| **actor/pg_loss** | 波动但稳定，GRPO 中为正 | 变化 >5x 初始值 | 变为负值(GRPO)或 NaN | **稳定波动** |
| **actor/ppo_kl** | 0.001-0.05 | 0.05-0.1 | >0.1 且持续上升 | **稳定或缓慢上升后收敛** |
| **actor/pg_clipfrac** | 0.01-0.2 | 0.2-0.4 | >0.4 | **稳定** |
| **actor/entropy** | 0.3-1.0 | 0.1-0.3 | <0.1 或 >3.0 | **缓慢下降但不触底** |
| **actor/grad_norm** | 0.1-10 | 10-100 | >100 或 NaN | **波动但无尖峰** |
| **response_length/mean** | 稳定或轻微增长 | 持续增长 >2x | 触顶(max_response_length) | **稳定** |

#### 二级指标（辅助判断）

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| **actor/pg_clipfrac_lower** | <0.1 | >0.2 说明 dual-clip 频繁触发 |
| **response/aborted_ratio** | <0.1 | >0.2 = 大量生成被中止 |
| **response_length/clip_ratio** | <0.1 | >0.3 = 太多达最大长度的响应 |
| **rollout_corr/kl** | <0.1 | >0.5 = 严重的 train-rollout 策略差距 |
| **rollout_is_eff_sample_size** | >0.5 | <0.2 = IS weights 退化严重 |

#### Agent 特有指标（多轮场景）

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| **num_turns/mean** | 逐渐增长 | 突然跳跃或归零 |
| **num_turns/max** | 接近但不超过 max_assistant_turns | 大量触顶 = agent 学不会 stop |
| **tool_call_counts/mean** | 稳定 | 突变可能意味 tool 使用模式变化 |

### 失败模式检测

| # | 场景 | 关键指标特征 | 直接原因 | 首要诊断问题 |
|---|------|-------------|---------|------------|
| 1 | Reward 完全不涨 | score/mean≈0, grad_norm≈0, ppo_kl≈0 | 所有 rollout 的 reward 相同 → advantage=0 | 数据是否太难？rollout 是否用 greedy？n 是否 ≥4？ |
| 2 | Reward 先涨后崩 | score 先升到峰值后快速下降，ppo_kl 预先上升 | 模型找到 reward 漏洞后被纠正 | 抽查生成样本，检查 response_length 是否在涨 |
| 3 | KL 发散 | ppo_kl 单调加速上升，clipfrac 升高 | 每次更新步长太大 | lr 是否偏高？是否有 KL 约束？ppo_epochs > 1？ |
| 4 | 模式坍缩 | entropy <0.1, response_length std≈0, clipfrac<0.02 | 策略坍缩到确定性输出 | temperature 是否过低？entropy_coeff 是否关闭？ |
| 5 | Response 长度失控 | response_length 每 10 步增长 >20%，clip_ratio >0.3 | loss_agg_mode: token-mean 导致长度正反馈 | 检查 loss_agg_mode 和 reward-长度相关性 |
| 6 | 梯度爆炸/NaN | grad_norm spike >100 或直接 NaN，pg_loss→NaN | 某个 batch 触发了数值溢出 | NaN 前 ppo_kl 是否 >1.0？dtype 是 FP16 还是 BF16？ |
| 7 | GPU OOM | CUDA out of memory，训练中断 | 显存不足 | OOM 发生在哪个阶段（rollout/actor/ref）？ |
| 8 | Agent 学不会停下 | num_turns 触顶，aborted_ratio 上升，score 很低 | Agent 未学会调用 finish/submit | 检查 trajectory 是否有 submit 动作 |
| 9 | 验证背离 | 训练 reward 上涨但验证指标下降 | 过拟合训练数据分布 | 检查训练/验证集分布差异，ppo_epochs 是否 >1 |
| 10 | Clipfrac 过高 | pg_clipfrac >0.4，ppo_kl 偏高 | clip_ratio 太小或 lr 太大 | 检查 clip_ratio 和 lr 配合 |
| 11 | Clipfrac 极低 | pg_clipfrac <0.01，ppo_kl≈0，reward 不涨 | lr 太小或 advantage=0 | 检查 grad_norm，增大 lr |
| 12 | 零方差 Group | score/max==score/min，grad_norm≈0 | 同一 prompt 的 N 个 rollout 全部成功或全部失败 | 全对→数据太简单；全错→数据太难 |

#### 场景速查矩阵

| 场景 | reward | ppo_kl | entropy | grad_norm | pg_clipfrac | resp_len | 动作优先级 |
|------|--------|--------|---------|-----------|-------------|----------|-----------|
| 1. 完全不学 | →0 | →0 | → | →0 | → | → | 检查数据+reward |
| 2. 先涨后崩 | ↗↘ | ↗ | ↓或↑ | 波动 | ↗ | → | 加固 reward |
| 3. KL 发散 | →或↘ | ↗↗ | ↑ | ↗ spike | ↗ | → | 降 lr + 加 KL |
| 4. 模式坍缩 | → | →0 | ↓↓ | →0 | ↓↓ | std→0 | 提 temp + 提 clip_high |
| 5. 长度失控 | → | → | → | → | → | ↗↗ | 改 loss_agg_mode |
| 6. NaN | NaN | 高前兆 | ↑或↓ | NaN | 高 | → | 回滚 + 降 lr |
| 7. OOM | — | — | — | — | — | — | 降 batch/长度 |
| 8. Agent 不停止 | ↓ | → | → | → | → | ↗↗ | compact filter |
| 9. 验证背离 | ↗(train) | → | → | → | → | → | 加数据多样性 |
| 10. clipfrac高 | → | ↗ | → | ↗ | ↗↗ | → | 降 lr 或增 clip |
| 11. clipfrac低 | → | →0 | → | →0 | ↓↓ | → | 提 lr，查 reward |
| 12. 零方差group | → | → | → | →0 | → | → | filter_groups |

> 图例: `→` 稳定, `↗` 上升, `↘` 下降, `↗↗` 快速上升, `↓↓` 快速下降, `→0` 趋近 0, `—` 不适用

### 综合征诊断（多指标联合）

> 当多个指标同时异常时，按症状组合定位根因。综合征是**假说生成工具**，不是确定性诊断，必须在报告中标注置信度和需验证的条件。

**综合征 #1: 策略发散 (Policy Divergence)** — 至少满足 3 项:
- ppo_kl 持续上升 (>0.1) + entropy 上升或剧烈波动 + grad_norm 出现 spike + pg_loss 震荡
- 根因: **lr 过大** (★★★★★) → 降 lr 2-5x；**无 KL 约束** (★★★★) → 启用 use_kl_loss=true, kl_loss_coef=0.01
- 排除线索: lr > 2e-6 (32B) 即为偏高；use_kl_loss=False 且 use_kl_in_reward=False 则完全无约束

**综合征 #2: 探索枯竭 (Exploration Exhaustion)** — 至少满足 3 项:
- entropy 持续下降 (<0.2) + reward 停滞 + pg_clipfrac 偏低 (<0.05) + score/std 很小
- 根因: **温度过低** (★★★★★) → 提 temperature 到 1.2-1.5；**模式坍缩** (★★★★) → 启用 entropy_coeff=0.0005
- 排除线索: temperature <0.8 即为偏低；response_length/std → 0 说明模式坍缩

**综合征 #3: Reward Hacking (奖励欺骗)** — 至少满足 2 项:
- reward 前期很高 + reward 突然下降 + entropy 可能先降后升 + response_length 快速增长
- 根因: **Reward 函数有漏洞** (★★★★★) → 修复 reward 函数；**单一维度 reward** (★★★★) → 用 gdpo + gdpo_reward_keys
- 排除线索: 手动检查高分样本；response_length 与 reward 强相关 → 长度偏差污染

**综合征 #4: 长度偏差污染 (Length Bias)** — 至少满足 2 项:
- response_length 快速上升 + 长 response 系统性获得更高 reward + pg_clipfrac 上升
- 根因: **loss_agg_mode: token-mean** (★★★★★) → 改用 seq-mean-token-sum-norm
- 排除线索: 散点图看 response_length vs reward 相关性

**综合征 #5: Off-Policy 鸿沟** — 至少满足 3 项（需 rollout_correction 启用时有指标）:
- rollout_corr/kl >0.3 + rollout_is_eff_sample_size <0.3 + ppo_kl 上升 + pg_clipfrac 升高
- 根因: **Rollout/Training 精度不匹配** (★★★★★) → 对齐精度或启用 rollout_correction
- 排除线索: rollout BF16 vs training FP32；高 lr + ppo_epochs > 1

**综合征 #6: Agent 无限循环** — 至少满足 3 项:
- num_turns/mean 快速增长且触顶 + reward 不涨 + aborted_ratio 上升 + response_length 增长
- 根因: **Agent 未学会 submit** (★★★★★) → compact filtering；**环境反馈触发重试循环** (★★★★)
- 排除线索: 检查 trajectory 无 submit 动作；tool response 是否触发重试

**综合征 #7: 训练完全不学** — 至少满足 3 项:
- reward 从 step 0 就不涨 + ppo_kl ≈0 + grad_norm ≈0 + pg_loss 几乎不变
- 根因: **配置错误: 参数不更新** (★★★★★) → 检查 optimizer/frozen weights；**所有样本都失败** (★★★★) → 手动跑 reward 函数
- 排除线索: optimizer 配置、lr > 0、critic/score/max = 0

**综合征 #8: 过早收敛于捷径** — 至少满足 2 项:
- reward 快速上升后停滞 + entropy 快速下降 + response_length 很短或固定模式 + num_turns 很少
- 根因: **Reward 信号过于简单** (★★★★★) → 加入 format/quality/process reward；**数据缺乏多样性** (★★★★)
- 排除线索: 抽查样本确认捷径模式；所有样本有相似 pattern

## 执行步骤

### 1. 加载轨迹

```python
from traj_opt.analyzer.base import AnalyzerBase
from traj_opt.config import DEFAULT_CONFIG

analyzer = AnalyzerBase(DEFAULT_CONFIG)
```

如果指定了 `--session` 参数:
```python
trajectories = analyzer.get_rllm_trajectories(session_id="{session_id}")
```

如果未指定（独立使用）:
```python
trajectories = analyzer.get_rllm_trajectories()
```

如果轨迹数 < 1，提示用户先执行训练再分析。

### 2. 从轨迹数据提取训练详情

对每条 rllm-train 轨迹:

a) 读取 traj_opt/output/trajectories/{session_id}/trajectories.jsonl
   → 定位 skill_name="rllm-*" 的轨迹

b) 从轨迹的 tool_calls 中提取训练数据:
   - tool_name="Read" + file_path 含 "config.json" → 训练配置
   - tool_name="Bash" + command 含 "tail" + "training_log" → reward 趋势
   - tool_name="Read" + file_path 含 "perf_stats.json" → 性能统计
   - tool_name="Bash" + response 含 "Error"/"Traceback" → 错误信息
   - tool_name="Read" + file_path 含 "trajectories/" 或 "verl_metrics.jsonl" → Agent 行为数据（num_turns, tool_call_counts, aborted_ratio）

c) 使用 Python 辅助方法简化提取:
   ```python
   from traj_opt.analyzer.base import AnalyzerBase
   analyzer = AnalyzerBase(DEFAULT_CONFIG)
   # 如果指定了 --session，传递 session_id 过滤
   training_data_list = analyzer.get_available_training_data(session_id="{session_id}" if session_id else None)
   # 返回: [{trajectory_summary, training_data: {config, reward_trend, perf_stats, errors, log_snippets}}]
   ```

d) 如果轨迹数据为空或 tool_response 中缺少关键信息:
   → 报告: "数据不足。请确认 rllm-xx skill 的数据表面化准则已实施。
            缺失: [config/reward_trend/perf_stats/errors]"
   → 不做猜测性分析，明确标注数据缺失

### 2.5. 定量诊断评估

对提取到的指标进行定量诊断:

a) 将每个指标值与诊断参考框架（上方分析框架中的"定量指标解读基准"）对比，判定正常/警戒/危险

b) 标记超出正常范围的指标，记录具体数值和发生的时间步

c) 检查是否存在多个指标同时异常，匹配综合征诊断中的症状组合

d) 对每条异常发现，从轨迹中提取 corroborating evidence（关联指标、跨轮次对比数据）

e) 使用 Python 辅助:
   ```python
   from traj_opt.analyzer.base import AnalyzerBase
   analyzer = AnalyzerBase(DEFAULT_CONFIG)
   # 提取多轮训练数据用于跨轮次对比
   all_training_data = analyzer.get_available_training_data()
   # 按轮次和 session 组织，交叉比对参数变化与指标变化
   ```

### 3. 跨轮次关联分析

将多轮训练数据放在一起分析:
- 参数变化与 reward 变化的因果关系
- 同一参数在不同场景下的表现差异
- 递进式的错误模式（如第一轮 OOM 后调参，第二轮仍有问题）
- 跨轮次指标趋势: 对比诊断参考框架中的场景基线，识别偏离

### 4. 生成分析报告

报告格式:

```markdown
# rllm-train 轨迹分析报告

生成时间: {timestamp}
分析范围: 最近 {days} 天
rllm-train 轨迹: {count} 条

## 训练健康仪表盘

| 指标 | 观测值 | 诊断状态 | 证据来源 |
|------|--------|---------|----------|
| reward/score | {value} ({trend}) | ✅正常 / ⚠️警戒 / 🔴危险 | session X, step M-N |
| actor/pg_loss | ... | ... | ... |
| actor/ppo_kl | ... | ... | ... |
| actor/pg_clipfrac | ... | ... | ... |
| actor/entropy | ... | ... | ... |
| actor/grad_norm | ... | ... | ... |
| response_length/mean | ... | ... | ... |

## 训练执行概览

| Session | 配置摘要 | 结果 | 关键问题 |
|---------|---------|------|---------|
| ... | ... | ... | ... |

## 综合征匹配 (如有)

### 匹配: {综合征名称} (置信度: {高/中/低})
- 满足条件: {列出满足的症状}
- 未满足条件: {列出不满足的症状}
- 根因假说: {按概率排序的根因}
- 建议验证: {需要进一步收集的数据}

## 问题发现

### 1. {问题标题} [影响: {target_skill}] [严重程度: 🔴高 / ⚠️中 / ℹ️信息]
**现象**: ...
**证据**: ... (引用 session_id + 指标数值)
**诊断**: ... (引用诊断参考框架中的对应条目)
**建议**: ...

## 优化建议

| 优先级 | 目标 Skill | Section | Action | 描述 | 置信度 |
|--------|-----------|---------|--------|------|--------|
| ... | ... | ... | ... | ... | ... |

## 建议的 Patch 内容

### Patch 1: {description}
(完整 patch markdown 内容)
```

### 5. 保存报告

```python
from traj_opt.analyzer.report import ReportWriter

writer = ReportWriter(DEFAULT_CONFIG)
report_path = writer.write_report(report_content, prefix="rllm-analysis")
```

输出报告路径供后续 traj-optimize 使用。

### 数据完整性检查

在 Step 2 提取训练详情后，检查数据完整性:

- 如果 perf_stats 为空:
  输出警告并在报告中标注: 性能分析数据缺失

- 如果 reward_trend 数据点 < total_steps * 0.5:
  输出警告并在报告中标注: reward 趋势为部分数据

- 如果缺少 Agent 指标 (num_turns, aborted_ratio, tool_call_counts):
  若训练为多轮 Agent 场景 → 输出警告并在报告中标注: Agent 行为数据缺失
  若训练为单轮场景 → 忽略

- 缺少任何一级指标的观测值时:
  明确标注该指标的诊断结论为"数据不足，无法判定"

数据缺失不阻断分析流程，但必须在报告中明确标注，避免基于不完整数据做出错误结论。

## 紧急状况处理

当轨迹数据显示以下紧急状况时，在报告中生成"紧急响应建议"区块，标注为 🔴 最高优先级。

### NaN / Loss 爆炸

**轨迹特征**: Bash tool response 含 "NaN", "inf"，或 pg_loss/grad_norm 变为 NaN

**报告中的紧急响应建议**:
1. ⛔ **不要继续训练** — NaN 会传播并损坏 optimizer state
2. 回滚到最近的有效 checkpoint
3. 配置调整后从 checkpoint 恢复:

```yaml
# NaN 后安全重启配置
actor:
  lr: {原值 / 5-10}
  clip_ratio: 0.1
  clip_ratio_high: 0.15
  clip_grad: 1.0
  loss_agg_mode: seq-mean-token-sum-norm

algorithm:
  norm_adv_by_std_in_grpo: False  # 避免除以接近 0 的 std

# 确认 dtype=bfloat16 (非 FP16)
```

**建议排查清单**:
- response_length 是否接近 max_response_length？
- NaN 前 ppo_kl 是否已经 >1.0？
- dtype 是 FP16 还是 BF16？（FP16 更容易溢出）
- loss_agg_mode 的分母是否可能为 0？
- norm_adv_by_std_in_grpo=True 时 group std 是否为 0？

### GPU OOM

**轨迹特征**: Bash tool response 含 "CUDA out of memory", "MemoryError", "OOM"

**报告中的紧急响应建议**:

按 OOM 发生阶段给出针对性调整:

| OOM 阶段 | 识别特征 | 核心调整 |
|----------|---------|---------|
| Rollout 生成 | timing_s/gen 附近报错 | gpu_memory_utilization↓, max_num_batched_tokens↓, max_num_seqs↓ |
| Actor 训练 | timing_s/update_actor 报错 | ppo_micro_batch_size_per_gpu↓, ppo_max_token_len_per_gpu↓ |
| Ref log prob | timing_s/ref 报错 | log_prob_micro_batch_size_per_gpu↓ |

**通用减显存组合**:
```yaml
rollout:
  gpu_memory_utilization: 0.35
  free_cache_engine: true
  enforce_eager: true          # 关 CUDA graph 省显存

actor:
  ppo_micro_batch_size_per_gpu: 1
  use_dynamic_bsz: true
  ppo_max_token_len_per_gpu: 8192
  fsdp_config:
    param_offload: true
  calculate_entropy: false      # 不计算熵
  use_kl_loss: false            # 不需要 ref forward
```

### 训练卡住 / 进程静默

**轨迹特征**: Monitor 长时间无输出（超过预期 step 间隔的 5x），所有指标停止更新，进程未退出

**报告中的建议**:
- 增大 nccl_timeout 到 1200s，ray_wait_register_center_timeout 到 600s
- 检查 dmesg 是否有 worker 静默 OOM
- 检查 Ray worker 是否超时

## --optimize 模式

当带 `--optimize` 参数调用时，分析完成后自动调用 traj-optimize:

1. 执行上述所有分析步骤
2. 生成报告和结构化 SkillOptimizationSuggestion
3. 调用 `Skill("traj-optimize", args="<report_path>")`

这实现了半自动流程: 一键完成 分割 → 分析 → 生成 patch。

## 领域知识: 训练动态模式识别

### 可识别的训练动态模式

| 模式 | 轨迹中的特征 | 定量阈值 | 含义 | 分析方法 | 关联综合征 |
|------|-------------|---------|------|---------|-----------|
| 快速崩溃 | reward 在前 10% steps 内从峰值降为 0 | reward 从峰值降幅 >80%, 发生在 step < 总步数*0.1 | 学习率过高 | 对比相同模型不同 lr 的轨迹 | #1 策略发散 |
| 延迟崩溃 | reward 在 10-30% steps 时开始下降 | reward 峰值后持续下降 >30% | 数据量超出模型容量 | 对比不同 num_problems 的轨迹 | #1/#3 |
| 稳定学习 | reward 单调递增或小幅波动 | reward 上升且相邻步差值 <0.2 | 配置合理 | 记录为安全配置参考 | — |
| 高 variance | reward 大幅震荡 | 相邻步差值 >0.3 | 学习率偏高或 batch 偏小 | 对比相邻轮次的 lr 和 reward variance | #1 |
| 格式退化 | 后期 tool_call 格式错误增加 | 格式错误率 > 基准的 2x | 模型遗忘了格式模板 | 检查训练长度和格式 reward 权重 | #3 |
| 零学习 | reward 始终接近 0 | score/mean <0.1 持续 >30 步 | 题目太难或 lr 太低 | 对比难度配置和模型能力 | #7 |
| Catastrophic Forgetting | reward 先升后降，峰值后持续下降 | reward 从峰值持续下降 >20% | 过拟合或 epochs 过多 | 检查 epoch 边界，对比 epochs 配置 | #1/#9 |
| Agent 无限循环 | num_turns 触顶, reward 不涨, aborted_ratio 上升 | num_turns/mean > 0.8*max_turns, aborted_ratio >0.2 | Agent 未学会 timely submit | 抽查 trajectory，确认 submit 行为 | #6 |
| 验证背离 | 训练 reward 上涨但验证指标下降 | train reward 上升 + val metric 下降 | 过拟合训练数据分布 | 对比 train/val 曲线，检查数据分布 | #9 |

### 分析方法论

1. **从轨迹提取事实** — 读取轨迹数据，提取配置参数、reward 序列、错误信息
2. **定量诊断** — 将提取的指标值对照诊断参考框架的健康/警戒/危险范围，标记异常指标
3. **跨轮次对比** — 比较不同轮次的参数变化与结果变化，建立因果关系
4. **模式匹配** — 将观察到的 reward 趋势与上述模式表对照，结合综合征诊断进行多指标联合分析
5. **安全边界推断** — 基于多轮数据推断参数安全范围（而非预设）:
   - 使用诊断参考框架的定量阈值作为初始基准
   - 基于本训练的具体轨迹数据调整阈值
   - 示例: 诊断参考指出 ppo_kl >0.1 为警戒，若本训练在 ppo_kl=0.08 时已出现 reward 下降，则推断该模型的 ppo_kl 安全上限为 0.08
   - 在报告中明确标注: "基于 N 轮轨迹推断" vs "基于诊断参考框架"
6. **生成假说** — 对观察到的现象提出可能的解释，标注置信度:
   - 高置信度: 基于 3+ 轮轨迹的一致模式
   - 中置信度: 基于 1-2 轮轨迹
   - 低置信度: 推测性解释，需要更多数据验证

### 诊断决策框架

当观察到特定指标表现时，按以下流程生成诊断假说:

```
单指标异常 → 查阅分析框架中的单指标诊断 → 生成 1-2 个根因假说
多指标共变 → 查阅综合征诊断 → 匹配症状组合 → 确定置信度最高的综合征
验证假说   → 从轨迹中寻找 corroborating evidence:
              - 检查关联指标是否也符合预测
              - 对比跨轮次的参数变化与结果变化
              - 确认排除条件是否满足
生成建议   → 基于假说置信度:
              - 高置信度 → 生成 patch
              - 中置信度 → 生成建议并标注"需验证"
              - 低置信度 → 标注为推测，建议下一轮收集特定数据
```

**常见 观察→假说 参考**（每次使用必须交叉验证，不可直接套用）:

| 轨迹观察 | 可能假说（按先验排序） | 验证方法 |
|----------|---------------------|---------|
| reward 50步不涨 | 探索不足 / 数据难度不匹配 / 模型容量不足 | 查 entropy, clipfrac, score/max |
| ppo_kl 持续上升 | lr过大 / 无KL约束 / clip_ratio过大 | 查 lr配置, use_kl_loss, clipfrac |
| entropy 快速下降 | temperature过低 / 数据太简单 / reward信号过强 | 查 temperature, score/std |
| response_length 快速增长 | loss_agg_mode为token-mean / 模型学会拖延 | 查 loss_agg_mode, num_turns |
| grad_norm spike | 极端样本 / lr过大 / FP16溢出 | 查 rewards/max, dtype配置 |
| pg_loss 剧烈震荡 | batch中有outlier / 策略突变 / KL爆炸 | 查 rewards/max, grad_norm, ppo_kl |
| pg_loss 变为负值(GRPO) | loss_agg_mode配置错误 / advantage符号不一致 / 数值精度问题 | 查 loss_agg_mode, advantages/mean, dtype |

### 训练健康检查清单

分析时，按训练阶段检查以下项目。缺失数据标注为"未采集"。

**训练启动前检查** (从轨迹中的 config 信息验证):
- max_prompt_length + max_response_length ≤ 模型最大 context
- train_batch_size ≥ ppo_mini_batch_size
- rollout.n ≥ 4 (GRPO 最小 group size)
- dtype=bfloat16 (非 FP16)
- reward 函数路径存在且可导入

**训练初期检查 (前 10 步)**:
- pg_loss 在正常范围，非 NaN
- grad_norm > 0.001 (参数在更新)
- score/max > 0 (至少有一个样本成功)
- response_length 在预期范围
- 无一指标出现指数增长趋势

**训练中期检查 (每 50 步)**:
- reward 整体趋势向上
- entropy > 0.1 (没有崩溃)
- ppo_kl < 0.1 或在可控范围
- pg_clipfrac < 0.3
- response_length 无指数增长
- grad_norm 无持续 spike
- 验证集趋势与训练集一致

**Agent 训练额外检查**:
- num_turns/mean 在合理范围，未普遍触顶
- aborted_ratio < 0.2
- tool_call_counts/mean 稳定
- 随机抽查 trajectory，确认 agent 在正常交互（有 submit 动作）

### 场景参考基线

以下为已知健康训练的指标范围，用于对比分析。这些是**参考基准**，非强制目标。

**DeepSWE 风格 (Qwen3-32B, GRPO++, Agent)**:
```
pg_loss:        0.05-0.15, 温和波动
ppo_kl:         0.005-0.03, 缓慢上升
pg_clipfrac:    0.05-0.25
entropy:        0.4-0.8, 缓慢下降
grad_norm:      0.5-5.0
score/mean:     从 0.05 → 0.15 (200步)
response_length: 2000-5000, 稳定
num_turns:      5-15, 逐步增长
```

**标准数学 GRPO (Qwen2.5-7B, GSM8K)**:
```
pg_loss:        0.01-0.05
ppo_kl:         0.001-0.01
pg_clipfrac:    0.05-0.2
entropy:        0.3-0.6
grad_norm:      0.1-2.0
score/mean:     从 0.0 → 0.7-0.9
response_length: 100-300
```

### 指标依赖关系速查

```
reward/score ──→ advantages ──→ pg_loss ──→ grad_norm
     │                │              │
     │           pg_clipfrac ←── clip_ratio
     │                │
     └──→ entropy (通过策略梯度)
              │
         ppo_kl ←── old_log_prob vs log_prob
```

当以下 3 个指标同时异常时，**几乎一定**有训练问题需要立即介入:
1. ppo_kl > 0.5 **且**
2. entropy < 0.1 或 > 3.0 **且**
3. grad_norm > 50 或 NaN
→ 按综合征 #1（策略发散）处理，建议回滚 checkpoint

### 禁止事项

- 不引用预设的参数安全范围表（如 "qwen-0.5b lr 应在 1e-6~1e-5"）
- 不假设固定的 "问题 → skill" 映射（如 "lr 过高一定修改 rllm-config"）
- 所有数值判断须有轨迹证据支撑，引用具体的 session_id 和数据
- 不使用 "经验表明" / "通常情况下" 等无证据表述
- 如果轨迹数据不足以得出结论，明确报告 "数据不足，需要更多轮次训练"
- 诊断参考框架中的数值范围是临床解读基准，不是预设的安全参数表。使用前必须声明其来源（"诊断参考框架" vs "本训练轨迹证据"）
- 综合征诊断是假说生成工具，不是确定性诊断。必须在报告中标注匹配的置信度和需要进一步验证的条件
