---
id: success-round2-cross-round-template
target_section: analysis-framework
action: append
description: "新增跨轮次量化对比模板: 标准化metrics对比表 + Python数据提取代码 + Round 1vs2实证案例"
status: proposed
source: round2-reanalysis-success
source_sessions: ["c29e2662-4417-4a6b-ac04-ecbaa84daeaa", "run_1780250271"]
---

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
