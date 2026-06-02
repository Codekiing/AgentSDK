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

### 跨轮次量化对比模板

当存在多轮训练数据时，必须输出标准化的跨轮次对比表，而非定性描述。这能揭示单轮分析无法发现的趋势和因果关系。

#### 对比表模板

```markdown
## 跨轮次对比

| 指标 | Round N-1 | Round N | 变化 | 诊断 |
|------|-----------|---------|------|------|
| num_problems | {prev} | {curr} | {delta_pct}% | {分析参数变更的影响} |
| learning_rate | {prev} | {curr} | {delta} | {分析lr变更的影响} |
| batch_size | {prev} | {curr} | {delta_pct}% | {分析等效batch变更} |
| num_generations | {prev} | {curr} | {delta} | {分析GRPO group size} |
| temperature | {prev} | {curr} | {delta} | {分析探索空间变化} |
| entropy_coeff | {prev} | {curr} | {delta} | {分析entropy约束} |
| val-acc (final) | {prev} | {curr} | {delta_pct}% | {分析最终效果} |
| val-acc trend | {prev_trend_icon} | {curr_trend_icon} | {trend_change} | {分析趋势变化} |
| entropy (start→end) | {prev_start}→{prev_end} | {curr_start}→{curr_end} | {delta_decline} | {分析entropy动态} |
| entropy decline rate | {prev_rate}/step | {curr_rate}/step | {delta_rate}% | {量化崩塌速度} |
| PPO_KL (mean) | {prev_kl} | {curr_kl} | {delta_kl} | {分析policy更新强度} |
| pg_clipfrac (mean) | {prev_cf} | {curr_cf} | {delta_cf} | {分析更新约束} |
| grad_norm (mean) | {prev_gn} | {curr_gn} | {delta_gn} | {分析梯度健康度} |
```

#### 数据提取辅助代码

```python
import json, numpy as np

def extract_round_metrics(run_dir):
    """Extract standardized metrics from a run directory for cross-round comparison."""
    metrics_path = f'{run_dir}/verl_metrics.jsonl'
    config_path = f'{run_dir}/config.json'
    
    with open(config_path) as f:
        config = json.load(f)
    
    scores, ents, kls, cfs, gns = [], [], [], [], []
    with open(metrics_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            data = d.get('data', {})
            s = data.get('critic/score/mean')
            e = data.get('actor/entropy')
            k = data.get('actor/ppo_kl')
            c = data.get('actor/pg_clipfrac')
            g = data.get('actor/grad_norm')
            if s is not None: scores.append(s)
            if e is not None: ents.append(e)
            if k is not None: kls.append(k)
            if c is not None: cfs.append(c)
            if g is not None: gns.append(g)
    
    n = len(scores)
    return {
        'config': config,
        'n_steps': n,
        'score_mean': np.mean(scores), 'score_first': scores[0], 'score_last': scores[-1],
        'score_trend': 'up' if scores[-1] > scores[0] else 'down',
        'entropy_first': ents[0], 'entropy_last': ents[-1],
        'entropy_decline_rate': (ents[0] - ents[-1]) / n if n > 0 else 0,
        'kl_mean': np.mean(kls) if kls else 0, 'kl_max': np.max(kls) if kls else 0,
        'clipfrac_mean': np.mean(cfs) if cfs else 0,
        'grad_norm_mean': np.mean(gns) if gns else 0,
        'grad_norm_max': np.max(gns) if gns else 0,
    }
```

#### 实证案例: Round 1 vs Round 2

```markdown
| 指标 | Round 1 (0.5B) | Round 2 final (0.5B) | 变化 | 诊断 |
|------|---------------|---------------------|------|------|
| num_problems | 32 | 512 | +1500% | 数据量过大淹没学习信号 |
| learning_rate | 5e-6 | 5e-6 | 不变 | lr不是退化原因 |
| batch_size | 2 | 8 | +300% | 等效batch=256过大 |
| num_generations | 8 | 8 | 不变 | — |
| val-acc (final) | 0.488 | 0.334 | -31.6% | 严重退化: 训练有害 |
| val-acc trend | ↗ (0.486→0.488) | ↘ (0.444→0.334) | 趋势反转 | 从学习变为遗忘 |
| entropy (start→end) | 0.279→0.218 | 0.289→0.092 | Δ多-0.14 | 崩塌幅度加倍 |
| entropy decline rate | 0.0013/step | 0.0016/step | +23% | 崩塌加速 |
| PPO_KL (mean) | 0.0 | 0.0 | 不变 | 系统性: KL=0非根本原因 |
| pg_clipfrac (mean) | 0.0 | 0.0 | 不变 | 系统性: 模型更新幅度极小 |
| grad_norm (mean) | 2.46 | 1.60 | -35% | 梯度减弱: 学习信号衰减 |
```

**关键洞察**: 仅看Round 2无法判断5e-6的lr是否合适。通过跨轮次对比发现:
- R1 lr=5e-6 + 32 problems = val=0.488 ✅
- R2 lr=5e-6 + 512 problems = val=0.334 ❌
- **根因不是lr, 而是num_problems!** lr=5e-6在少量数据上是有效的

这个洞察**只能通过跨轮次对比获得**，单轮分析会错误地将lr标记为问题。

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

### 重新分析协议 (Re-analysis Protocol)

当对已优化过的轮次进行重新分析时 (如用户显式请求 `/traj-train-optimize round=N`, 且该轮状态为 `optimization_complete`):

#### 强制步骤

1. **读取上一版报告**: 
   ```
   从 round status 获取上一版报告路径:
   prev_report = status["optimization"]["report_path"]
   ```

2. **对比分析**: 识别上一版报告的遗漏和不足:
   - 上一版发现了哪些问题? (列出清单)
   - 哪些指标/维度未被分析? (检查: val趋势? entropy崩塌速率? 中间run? 跨轮次对比?)
   - 上一版的 patch 是否被接受? 如果为0, 为什么? (说明未被接受的原因)

3. **深挖一层**: 必须比上一版多发现至少一个维度:
   - 如果上一版只看最终run → 本次分析所有中间run (Multi-run Intra-round)
   - 如果上一版仅看train指标 → 本次重点分析val指标
   - 如果上一版定性描述 → 本次加入定量速率/比例分析
   - 如果上一版单轮分析 → 本次加入跨轮次量化对比

4. **标注增量**: 在报告中明确使用标记区分:
   - ✅ 确认上一版的发现 (列出确认项)
   - 🆕 本次新发现 (列出新发现项)
   - ❌ 推翻上一版的发现 (需充分证据, 注明推翻理由)

#### Round 2重新分析案例 (方法示范)

上一版报告 (20260601-040650) vs 本次重新分析:

| 上一版发现 | 本次深化 | 突破方法 |
|-----------|---------|---------|
| ✅ PPO_KL=0 标记为严重 | 确认为系统性(KL=0本身不致命, R1实证val=0.488) | 跨轮次对比 |
| ✅ num_problems=512过载 | 量化实证: val从0.444→0.334单调下降 | val趋势分析 |
| (未发现) | 🆕 Entropy崩塌: 0.289→0.092, 速率0.0016/step | 速率量化 |
| (未发现) | 🆕 3次7B实验的安全边界教训 | 多run分析 |
| (未发现) | 🆕 验证背离: 11次val检查持续退化 | 新失败模式#13 |
| ✅ Epoch未完成 | 确认: training/epoch=1 vs config=2 | 重复验证 |
| ✅ RoundState未写入 | 确认未修复, 强化为强制步骤 | 状态检查 |

**关键教训**: 上一版最大的遗漏是**没有分析val-acc趋势**。train score震荡看起来"正常"(~0.46), 但val在默默退化(0.444→0.334)。这导致上一版的patch没有针对性地解决验证背离问题。

**反面案例自查**: 每次分析完成后，对照此表自查:
- [ ] 是否检查了val指标趋势? (不仅是最终值)
- [ ] 是否分析了所有中间run? (不仅是最终run)
- [ ] 是否量化了指标变化速率? (不仅是方向)
- [ ] 是否做了跨轮次量化对比? (不仅是单轮)
- [ ] 是否区分了相关性和因果性? (不仅是共现)

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

### 失败模式 #13: 验证背离 (Validation Divergence) [新增 — Round 2实证]

| 属性 | 值 |
|------|-----|
| 场景 | 训练 reward 震荡或不变，但验证集指标持续单调下降 |
| 关键指标 | val-acc 连续 ≥3 次检查下降，train score 无明显上升趋势 |
| 直接原因 | 模型在训练集上过拟合或学习到有害模式，该模式在验证集上无效 |
| 首要诊断 | 检查 num_problems 是否过大？epochs 是否完成？entropy 是否崩塌？ |

**指标特征**:
- `critic/score/mean`: 震荡但无明显趋势 (看起来"正常"，但实际有问题)
- `val-aux/*/reward/mean@1`: 持续单调下降 (每次val检查都在退化)
- `actor/entropy`: 通常伴随下降 (但早期可能仍在正常范围)
- train-val gap: 持续扩大

**与已有模式的区别**:
- 不同于 #1 (完全不学): train reward 并不为0, 只是不涨
- 不同于 #9 (验证背离原版): 原版侧重 train↑ val↓, 本模式是 train→ val↓
- **关键特征**: train side 看起来"还好", 但 val side 在默默退化 — 这是最隐蔽的危险模式

**排查步骤**:
1. 检查 val-acc 最近4次检查的趋势 (连续下降≥3次→确认)
2. 检查 entropy 趋势 (是否在崩塌?)
3. 检查 num_problems vs model size (是否数据过多?)
4. 检查 epochs 完成情况 (是否完成配置的epoch数?)
5. 对比训练/验证集的 problem difficulty 分布

**Case Study: Round 2** (session: c29e2662-4417-4a6b-ac04-ecbaa84daeaa):
```
Step 10:  val-acc=0.444, train_score=0.594  ← 起始: val已低于R1(0.486)
Step 50:  val-acc=0.372, train_score=0.471  ← val降16%, train看起来正常
Step 100: val-acc=0.359, train_score=0.476  ← val再降, train仍震荡
Step 128: val-acc=0.334, train_score=0.406  ← val降25%, train才刚显下降
```
诊断: 512 problems × 1 epoch + entropy_coeff=0 → entropy崩塌 + 验证背离
修复: num_problems→48, epochs→2, entropy_coeff=0.001 → 参考Round 1实证(val=0.488)

### Entropy 崩塌速率量化 [新增诊断指标]

当 entropy 呈下降趋势时，计算崩塌速率有助于在早期预警:

```
entropy_decline_rate = (entropy_start - entropy_end) / n_steps
```

临床参考阈值 (0.5B-7B模型, temp=0.3-0.85):
- rate < 0.001/step: ✅ 健康 (如 Round 1: 0.0013, val稳定)
- rate 0.001-0.002/step: ⚠️ 警戒 (如 Round 2: 0.0016, val退化)
- rate > 0.002/step: 🔴 危险 (预期val快速退化)

**注意**: 速率阈值受 temperature/entropy_coeff/model_size 影响，必须以本训练基线为参考。

### Multi-run Intra-round 分析 [新增方法论]

当一轮训练包含多次尝试 (多次 rllm-config + rllm-run):

1. **提取所有 run 的配置和指标** — 不只是最终的 run
2. **按时间排序** 重现训练探索路径: 每次配置变更试图解决什么问题?
3. **关联分析**: 配置变更 → 指标变化的因果关系
4. **中间 run 的教训** 可能比最终 run 更有价值:

Round 2案例 (session: c29e2662):
| Run | 模型 | LR | Entropy范围 | Score | 教训 |
|-----|------|-----|------------|-------|------|
| R2-R1 | 7B | 1e-7 | 0.275→0.438 | 0.446 | 7B需要≥1e-6 |
| R2-R2 | 7B | 1e-6 | 0.289→0.150 | 0.485 | 7B 1e-6可工作 |
| R2-R3 | 7B | 5e-6 | 0.038→0.072 | 0.961⚠️ | 7B上限~3e-6, 5e-6=崩塌 |
| R2-final | 0.5B | 5e-6 | 0.289→0.092 | 0.464 | 同lr对0.5B也偏高 |

→ 这些信息**只能从多run分析中获取**, 仅看final run会丢失7B vs 0.5B的关键对比数据。
