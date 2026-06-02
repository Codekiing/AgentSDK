---
id: "verl-log-patterns"
target_section: "monitoring-methods"
action: append
description: "VERL backend monitoring: read verl_metrics.jsonl with inline python, report 20+ metrics per step, run 8-dimension learning signal diagnosis, auto circuit-break on dead training"
source: "2026-05-28 VERL backend integration"
created: "2026-05-28"

depends_on: []
conflicts_with: []
status: active
superseded_by: ""
---

### VERL 日志监控 (backend=verl)

当 config.json 中包含 `"backend": "verl"` 时，监控 `verl_metrics.jsonl`（VERL file logger 输出，每步一行 JSON，含 67 个指标）。

**核心原则**: 使用 `tail + python3 -c` 内联模式（避免外部脚本被 auto mode 阻止）。**Cron 轮询必须读取全量历史做诊断，不能只看最后 3 步。**

#### 流式监控

使用 Monitor 工具，command 为：

```bash
tail -f rllm_train/output/runs/<run_id>/verl_metrics.jsonl 2>/dev/null | python3 -c "
import sys, json
last = 0
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        s = d['step']
        if s > last:
            last = s
            data = d['data']
            r = data.get('critic/score/mean', 0)
            l = data.get('actor/pg_loss', 0)
            g = data.get('actor/grad_norm', 0)
            e = data.get('actor/entropy', 0)
            cl = data.get('response_length/mean', 0)
            adv = data.get('critic/advantages/mean', 0)
            ep = data.get('training/epoch', 0)
            tp = data.get('perf/throughput', 0)
            ab = data.get('response/aborted_ratio', 0)
            print(f'Step {s} | R {float(r):.4f} | Loss {float(l):+.4f} | Grad {float(g):.4f} | Ent {float(e):.3f} | Adv {float(adv):+.3f} | Abort {float(ab):.1%} | Len {float(cl):.0f} | Ep {float(ep):.2f} | tok/s {float(tp):.0f}')
    except: pass
"
```

timeout_ms=3600000 (1 小时)。

#### 兜底轮询（含 8 维学习信号诊断 + 自动熔断）

CronCreate 每 3 分钟执行。**必须读取全量 jsonl 历史**，对最后一步输出完整指标，对全量历史运行诊断检测。

```bash
python3 -c "
import json, sys, os
from pathlib import Path

run_dir = Path('rllm_train/output/runs/<run_id>')
jsonl = run_dir / 'verl_metrics.jsonl'
log = run_dir / 'training_log.txt'

if not jsonl.exists():
    print('ONESHOT: model loading, no metrics yet')
    # Check for crash in training log
    if log.exists():
        tail = log.read_text()[-4096:]
        for kw in ['Traceback', 'CUDA out of memory', 'Killed', 'SIGTERM']:
            if kw in tail:
                print(f'DETECTED: {kw} in training log')
                break
    sys.exit(0)

# Load full history
steps = []
for line in jsonl.read_text().strip().split('\n'):
    if not line.strip(): continue
    try: steps.append(json.loads(line))
    except: pass

if not steps:
    print('ONESHOT: empty metrics')
    sys.exit(0)

# ── Last step full metrics ──
last = steps[-1]
data = last['data']
s = last['step']
print(f'=== Step {s} ===')
metrics = [
    ('Reward', 'critic/score/mean', '.4f'), ('R_max', 'critic/score/max', '.4f'),
    ('R_min', 'critic/score/min', '.4f'), ('Reward(KL)', 'critic/rewards/mean', '.4f'),
    ('Advantage', 'critic/advantages/mean', '.4f'), ('Adv_max', 'critic/advantages/max', '.4f'),
    ('pg_loss', 'actor/pg_loss', '+.4f'), ('grad_norm', 'actor/grad_norm', '.4f'),
    ('entropy', 'actor/entropy', '.4f'), ('clipfrac', 'actor/pg_clipfrac', '.4f'),
    ('ppo_kl', 'actor/ppo_kl', '.4f'), ('lr', 'actor/lr', '.2e'),
    ('resp_len', 'response_length/mean', '.0f'), ('resp_clip', 'response_length/clip_ratio', '.3f'),
    ('abort', 'response/aborted_ratio', '.3f'), ('turns', 'num_turns/mean', '.1f'),
    ('t_gen', 'timing_s/gen', '.1f'), ('t_update', 'timing_s/update_actor', '.1f'),
    ('t_step', 'timing_s/step', '.1f'), ('throughput', 'perf/throughput', '.0f'),
    ('epoch', 'training/epoch', '.2f'), ('mfu', 'perf/mfu/actor_infer', '.3f'),
]
for label, key, fmt in metrics:
    v = data.get(key)
    if v is not None:
        print(f'  {label:12s} = {v:{fmt}}')

# ── 8-dimension learning signal diagnosis ──
n = len(steps)
scores = [s['data'].get('critic/score/mean', 0) for s in steps]
score_maxs = [s['data'].get('critic/score/max', 0) for s in steps]
score_mins = [s['data'].get('critic/score/min', 0) for s in steps]
grads = [s['data'].get('actor/grad_norm', 0) for s in steps]
ents = [s['data'].get('actor/entropy', 0) for s in steps]
advs = [s['data'].get('critic/advantages/mean', 0) for s in steps]
aborts = [s['data'].get('response/aborted_ratio', 0) for s in steps]
w = min(5, n)  # window size

alerts = []

# S0: No learning signal — grad_norm < 0.01 for last w steps
if n >= 5 and all(g < 0.01 for g in grads[-w:]):
    alerts.append(('S0', 'no_learning_signal', f'grad_norm < 0.01 for {w} steps', grads[-w:]))

# S1: Entropy collapse — entropy < 0.01 for 3+ steps
if n >= 3 and all(e < 0.01 for e in ents[-3:]):
    alerts.append(('S1', 'entropy_collapse', f'entropy < 0.01 for 3 steps', ents[-3:]))

# S2: Entropy explosion — entropy > 2.0 for 2+ steps (R3 crash pattern)
if n >= 2 and all(e > 2.0 for e in ents[-2:]):
    alerts.append(('S2', 'entropy_explosion', f'entropy > 2.0 for 2 steps — catastrophic forgetting', ents[-2:]))

# S3: Zero reward deadlock — score/mean=0 AND score/max=0 for 3+ steps
zero_all = sum(1 for i in range(n) if scores[i] == 0 and score_maxs[i] == 0)
if n >= 3 and all(scores[i] == 0 and score_maxs[i] == 0 for i in range(n-3, n)):
    alerts.append(('S3', 'zero_reward_deadlock', f'all-zero reward for 3 steps ({zero_all}/{n} total)', []))

# S4: GRPO signal dead — advantage≈0 AND score/max>0.5 for 5+ steps (model plateaued)
if n >= 5 and all(abs(advs[i]) < 0.001 and score_maxs[i] > 0.5 for i in range(n-5, n)):
    alerts.append(('S4', 'grpo_signal_dead', f'advantage≈0 with score/max>0.5 for 5 steps — model plateaued on easy problems', []))

# S5: Aborted ratio spike — aborted_ratio > 0.5 for 3+ steps
if n >= 3 and all(a > 0.5 for a in aborts[-3:]):
    alerts.append(('S5', 'abort_spike', f'aborted_ratio > 0.5 for 3 steps — rollout engine failure', aborts[-3:]))

# S6: Grad explosion — grad_norm > 100
if any(g > 100 for g in grads[-w:]):
    alerts.append(('S6', 'grad_explosion', f'grad_norm > 100 at step {n}', [max(grads[-w:])]))

# S7: Reward variance zero — score/max = score/min for 5+ steps (no GRPO diversity)
same_score = sum(1 for i in range(n) if abs(score_maxs[i] - score_mins[i]) < 0.001)
if n >= 5 and all(abs(score_maxs[i] - score_mins[i]) < 0.001 for i in range(n-5, n)):
    alerts.append(('S7', 'reward_variance_zero', f'identical max/min for 5 steps ({same_score}/{n} total) — GRPO dead', []))

# ── Report diagnosis ──
print(f'\nDiagnosis ({n} steps):')
print(f'  reward: avg={sum(scores)/n:.4f} max={max(scores):.4f} min={min(scores):.4f}')
print(f'  grad:   avg={sum(grads)/n:.4f} max={max(grads):.4f}')
print(f'  entropy: {ents[0]:.3f}->{ents[-1]:.3f} avg={sum(ents)/n:.3f}')
print(f'  abort:  avg={sum(aborts)/n:.3f}')
print(f'  S0-S7 checks: ', end='')
statuses = []
for i in range(8):
    sid = f'S{i}'
    statuses.append(f'{sid}:ALERT' if any(a[0] == sid for a in alerts) else f'{sid}:ok')
print(' | '.join(statuses))

# ── Circuit break on critical alerts ──
CRITICAL = {'S0', 'S1', 'S2', 'S3', 'S5', 'S6'}  # auto-stop
WARN = {'S4', 'S7'}  # warn but continue

critical_hits = [a for a in alerts if a[0] in CRITICAL]
warn_hits = [a for a in alerts if a[0] in WARN]

if critical_hits:
    print()
    # fix_preset mapping
    preset_map = {
        'S0': 'lr_half', 'S1': 'entropy_bonus', 'S2': 'entropy_bonus',
        'S3': 'diagnose', 'S5': 'diagnose', 'S6': 'lr_tenth',
    }
    hit = critical_hits[0]
    preset = preset_map.get(hit[0], 'diagnose')
    print(f'=== CIRCUIT_BREAK ===')
    print(f'circuit_break: true')
    print(f'abort_reason: {hit[0]}: {hit[2]}')
    print(f'fix_preset: {preset}')
    print(f'analysis_json: {run_dir}/analysis.json')
    print(f'=== END_CIRCUIT_BREAK ===')
elif warn_hits:
    for w in warn_hits:
        print(f'WARN: {w[0]}: {w[2]}')
"
```

#### 完成检测

训练完成后检查 jsonl 行数是否达到预期（从 config.json 读取 total_epochs × num_problems / train_batch_size）。

#### VERL 异常检测（8 维学习信号诊断）

| ID | 条件 | 阈值 | 熔断 |
|----|------|------|------|
| S0 | 无学习信号 | grad_norm < 0.01 连续 5+ 步 | fix_preset=lr_half |
| S1 | Entropy 坍塌 | entropy < 0.01 连续 3+ 步 | fix_preset=entropy_bonus |
| S2 | Entropy 爆炸 | entropy > 2.0 连续 2+ 步 | fix_preset=entropy_bonus |
| S3 | 零 reward 死锁 | score/mean=0 且 score/max=0 连续 3+ 步 | fix_preset=diagnose |
| S4 | GRPO 信号消失 | advantage≈0 且 score/max>0.5 连续 5+ 步 | WARN (不熔断) |
| S5 | Rollout 失败 | aborted_ratio > 0.5 连续 3+ 步 | fix_preset=diagnose |
| S6 | 梯度爆炸 | grad_norm > 100 | fix_preset=lr_tenth |
| S7 | Reward 方差归零 | score/max = score/min 连续 5+ 步 | WARN (不熔断) |

S0-S3, S5-S6 触发自动熔断（输出 CIRCUIT_BREAK + 写入 analysis.json）。S4, S7 仅告警不中断。
