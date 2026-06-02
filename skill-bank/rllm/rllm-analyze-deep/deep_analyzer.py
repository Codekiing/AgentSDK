#!/usr/bin/env python3
"""
15-layer deep training analyzer for VERL GRPO training.

Usage:
    python deep_analyzer.py <run_dir> [--target 0.7]

Design principle: Every layer produces structured output. Layers cannot be skipped.
If a layer lacks data, it reports "MISSING_DATA" instead of silently skipping.
Tuning suggestions are generated ONLY after all 15 layers complete.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Layer result dataclass ──


@dataclass
class LayerResult:
    layer_id: int
    name: str
    status: str = "PENDING"  # PENDING | OK | WARN | ALERT | MISSING_DATA | SKIPPED
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


# ── Main analyzer ──


class DeepAnalyzer:
    def __init__(self, run_dir: str, target_reward: float = 0.7):
        self.run_dir = Path(run_dir)
        self.target = target_reward
        self.results: list[LayerResult] = []

        # VERL backend
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            self.config = json.loads(config_path.read_text())
        else:
            self.config = {}

        # Metrics path — always VERL
        self.metrics_path = self.run_dir / "verl_metrics.jsonl"
        self.log_path = self.run_dir / "training_log.txt"
        self.traj_dir = self.run_dir / "trajectories"
        self.val_dir = self.run_dir / "validation"

        # Loaded data
        self.steps: list[dict] = []
        self.scores: list[float] = []
        self.score_maxs: list[float] = []
        self.score_mins: list[float] = []
        self.grads: list[float] = []
        self.ents: list[float] = []
        self.losses: list[float] = []
        self.advs: list[float] = []
        self.kls: list[float] = []
        self.clipfracs: list[float] = []
        self.aborts: list[float] = []
        self.resp_lens: list[float] = []
        self.resp_clips: list[float] = []
        self.prompt_clips: list[float] = []
        self.tps: list[float] = []
        self.epochs: list[float] = []

    # ── Phase 1: Data Collection ──

    def collect_metrics(self) -> bool:
        """Load VERL metrics. Returns True if successful."""
        if not self.metrics_path.exists():
            return False
        with open(self.metrics_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    self.steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for s in self.steps:
            d = s["data"]
            self.scores.append(d.get("critic/score/mean", 0))
            self.score_maxs.append(d.get("critic/score/max", 0))
            self.score_mins.append(d.get("critic/score/min", 0))
            self.grads.append(d.get("actor/grad_norm", 0))
            self.ents.append(d.get("actor/entropy", 0))
            self.losses.append(d.get("actor/pg_loss", 0))
            self.advs.append(d.get("critic/advantages/mean", 0))
            self.kls.append(d.get("actor/ppo_kl", 0))
            self.clipfracs.append(d.get("actor/pg_clipfrac", 0))
            self.aborts.append(d.get("response/aborted_ratio", 0))
            self.resp_lens.append(d.get("response_length/mean", 0))
            self.resp_clips.append(d.get("response_length/clip_ratio", 0))
            self.prompt_clips.append(d.get("prompt_length/clip_ratio", 0))
            self.tps.append(d.get("perf/throughput", 0))
            self.epochs.append(d.get("training/epoch", 0))
        return True

    def collect_trajectory_samples(self, n_high: int = 3, n_low: int = 3) -> list[dict]:
        """Extract trajectory samples from highest and lowest reward steps."""
        samples = []
        if not self.traj_dir.exists() or not self.scores:
            return samples

        # Rank steps by reward
        ranked = sorted(enumerate(self.scores), key=lambda x: x[1], reverse=True)
        steps_to_sample = set()
        for idx, _ in ranked[:n_high]:
            steps_to_sample.add(self.steps[idx]["step"])
        for idx, _ in ranked[-n_low:]:
            steps_to_sample.add(self.steps[idx]["step"])

        for step_num in sorted(steps_to_sample):
            path = self.traj_dir / f"{step_num}.jsonl"
            if not path.exists():
                continue
            records = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            scores_list = [r.get("score", 0) for r in records]
            step_avg = sum(scores_list) / len(scores_list) if scores_list else 0
            n_total = len(records)
            n_ones = sum(1 for s in scores_list if s == 1)
            n_zeros = sum(1 for s in scores_list if s == 0)

            # Get a score=1 and score=0 example
            ones = [r for r in records if r.get("score", 0) == 1]
            zeros = [r for r in records if r.get("score", 0) == 0]

            sample = {
                "step": step_num,
                "n_total": n_total,
                "n_score_1": n_ones,
                "n_score_0": n_zeros,
                "score_1_example": {
                    "output": (ones[0].get("output", "")[:500] if ones else ""),
                    "gt": ones[0].get("gts", "") if ones else "",
                },
                "score_0_example": {
                    "output": (zeros[0].get("output", "")[:500] if zeros else ""),
                    "gt": zeros[0].get("gts", "") if zeros else "",
                },
                "output_lens": [len(r.get("output", "")) for r in records],
            }
            samples.append(sample)
        return samples

    # ── Phase 2: Layer-by-layer analysis ──

    def _add_result(self, result: LayerResult):
        self.results.append(result)

    def run_all_layers(self):
        """Execute all 15 layers in order. Each layer MUST produce a result."""
        n = len(self.scores)

        # ── Layer 1: Reward/Score Trends ──
        r = LayerResult(1, "Reward/Score Trends")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            half = n // 2
            first_half = sum(self.scores[:half]) / half if half > 0 else 0
            second_half = sum(self.scores[half:]) / (n - half) if n - half > 0 else 0
            r.metrics = {
                "reward_avg": sum(self.scores) / n,
                "reward_max": max(self.scores),
                "reward_min": min(self.scores),
                "reward_std": (sum((s - sum(self.scores) / n) ** 2 for s in self.scores) / n) ** 0.5,
                "score_max_always_1": all(m == 1.0 for m in self.score_maxs),
                "score_min_always_0": all(m == 0.0 for m in self.score_mins),
                "first_half_avg": first_half,
                "second_half_avg": second_half,
                "trend": "up" if second_half > first_half * 1.05 else ("down" if second_half < first_half * 0.95 else "flat"),
                "last_5_avg": sum(self.scores[-5:]) / 5 if n >= 5 else 0,
            }
            r.findings.append(f"Avg reward = {r.metrics['reward_avg']:.4f} (target={self.target})")
            r.findings.append(f"Trend: {r.metrics['trend']} (1st={first_half:.4f}, 2nd={second_half:.4f})")

            # Key insight: does Rmax always equal 1.0?
            if r.metrics["score_max_always_1"]:
                r.findings.append("Rmax is ALWAYS 1.0 — every batch has at least one solvable problem")
            if r.metrics["score_min_always_0"]:
                r.findings.append("Rmin is ALWAYS 0.0 — every batch has at least one unsolvable problem")
            if r.metrics["score_max_always_1"] and r.metrics["score_min_always_0"]:
                r.findings.append("⚠️ Reward oscillation likely driven by batch problem difficulty, not learning")
                r.alerts.append("DIFFICULTY_DRIVEN_OSCILLATION")

            if r.metrics["reward_avg"] >= self.target:
                r.status = "OK"
            else:
                r.status = "WARN"
        self._add_result(r)

        # ── Layer 2: KL / Reward Gap ──
        r = LayerResult(2, "KL / Reward Gap")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            kl_avg = sum(self.kls) / n if self.kls else 0
            r.metrics = {"ppo_kl_avg": kl_avg}
            if kl_avg < 0.0001:
                r.findings.append("PPO_KL ≈ 0 — policy barely updated")
                r.status = "ALERT"
                r.alerts.append("POLICY_NOT_UPDATING")
            elif kl_avg > 0.1:
                r.findings.append("PPO_KL high — policy updating aggressively")
                r.status = "WARN"
            else:
                r.findings.append(f"PPO_KL = {kl_avg:.6f} — normal range")
                r.status = "OK"
        self._add_result(r)

        # ── Layer 3: GRPO Group Variance ──
        r = LayerResult(3, "GRPO Group Variance")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            adv_avg = sum(self.advs) / n if self.advs else 0
            adv_abs = [abs(a) for a in self.advs]
            zero_var_steps = sum(1 for i in range(n) if abs(self.score_maxs[i] - self.score_mins[i]) < 0.001)
            r.metrics = {
                "advantage_avg": adv_avg,
                "advantage_abs_avg": sum(adv_abs) / n if adv_abs else 0,
                "zero_variance_steps": zero_var_steps,
                "zero_variance_pct": zero_var_steps / n * 100 if n > 0 else 0,
            }
            if adv_avg < 0.0001 and adv_avg > -0.0001:
                r.findings.append("Advantage ≈ 0 — GRPO group contrast is minimal")
                r.status = "ALERT"
                r.alerts.append("GRPO_SIGNAL_WEAK")
            else:
                r.findings.append(f"Advantage avg = {adv_avg:.6f}")
                r.status = "OK"
        self._add_result(r)

        # ── Layer 4: PPO Update Strength ──
        r = LayerResult(4, "PPO Update Strength")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            clipfrac_avg = sum(self.clipfracs) / n if self.clipfracs else 0
            grad_avg = sum(self.grads) / n if self.grads else 0
            r.metrics = {
                "clipfrac_avg": clipfrac_avg,
                "clipfrac_max": max(self.clipfracs) if self.clipfracs else 0,
                "grad_norm_avg": grad_avg,
                "grad_norm_max": max(self.grads) if self.grads else 0,
                "pg_loss_avg": sum(self.losses) / n if self.losses else 0,
            }
            if clipfrac_avg < 0.001:
                r.findings.append("clipfrac = 0 — updates never hit clip boundary. PPO update is extremely weak.")
                r.status = "ALERT"
                r.alerts.append("UPDATE_TOO_WEAK")
            elif clipfrac_avg > 0.3:
                r.findings.append(f"clipfrac = {clipfrac_avg:.3f} — too many updates clipped, policy may be changing too fast")
                r.status = "ALERT"
                r.alerts.append("UPDATE_TOO_STRONG")
            else:
                r.findings.append(f"clipfrac = {clipfrac_avg:.3f} — healthy update range")
                r.status = "OK"

            if grad_avg < 0.01:
                r.findings.append("grad_norm ≈ 0 — no effective gradient signal")
                r.alerts.append("NO_GRADIENT_SIGNAL")
                if r.status != "ALERT":
                    r.status = "WARN"
        self._add_result(r)

        # ── Layer 5: Critic [GRPO skips] ──
        r = LayerResult(5, "Critic")
        r.status = "SKIPPED"
        r.findings.append("GRPO does not use a critic model")
        self._add_result(r)

        # ── Layer 6: Exploration/Entropy ──
        r = LayerResult(6, "Exploration / Entropy")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            ent_start = self.ents[0]
            ent_end = self.ents[-1]
            ent_min = min(self.ents)
            temp = self.config.get("temperature", 0.7)
            r.metrics = {
                "entropy_start": ent_start,
                "entropy_end": ent_end,
                "entropy_min": ent_min,
                "temperature": temp,
            }
            if ent_end < 0.01:
                r.findings.append(f"Entropy collapse ({ent_start:.3f}→{ent_end:.4f}) — exploration died")
                r.status = "ALERT"
                r.alerts.append("ENTROPY_COLLAPSE")
            elif ent_end > 2.0:
                r.findings.append(f"Entropy explosion ({ent_start:.3f}→{ent_end:.3f}) — catastrophic drift")
                r.status = "ALERT"
                r.alerts.append("ENTROPY_EXPLOSION")
            else:
                r.findings.append(f"Entropy stable: {ent_start:.3f}→{ent_end:.3f} (temp={temp})")
                r.status = "OK"
        self._add_result(r)

        # ── Layer 7: Output Truncation ──
        r = LayerResult(7, "Output Truncation / Aborts")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            resp_len_avg = sum(self.resp_lens) / n if self.resp_lens else 0
            resp_clip_avg = sum(self.resp_clips) / n if self.resp_clips else 0
            abort_avg = sum(self.aborts) / n if self.aborts else 0
            max_resp_len = self.config.get("max_response_length", 2048)
            r.metrics = {
                "response_length_avg": resp_len_avg,
                "response_clip_ratio": resp_clip_avg,
                "abort_ratio": abort_avg,
                "max_response_length": max_resp_len,
                "len_utilization": resp_len_avg / max_resp_len if max_resp_len > 0 else 0,
            }
            if resp_len_avg / max_resp_len > 0.9:
                r.findings.append(f"Response length near limit ({resp_len_avg:.0f}/{max_resp_len}) — likely truncation")
                r.status = "ALERT"
                r.alerts.append("RESPONSE_TRUNCATION")
            elif abort_avg > 0.05:
                r.findings.append(f"High abort ratio ({abort_avg:.1%}) — rollout failures")
                r.status = "ALERT"
                r.alerts.append("HIGH_ABORT_RATE")
            else:
                r.findings.append(f"Response len OK ({resp_len_avg:.0f}/{max_resp_len}, abort={abort_avg:.1%})")
                r.status = "OK"
        self._add_result(r)

        # ── Layer 8: Throughput ──
        r = LayerResult(8, "Throughput / Timing")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            r.metrics = {
                "tok_s_avg": sum(self.tps) / n if self.tps else 0,
                "step_count": n,
            }
            r.findings.append(f"Throughput: avg {r.metrics['tok_s_avg']:.0f} tok/s over {n} steps")
            r.status = "OK"
        self._add_result(r)

        # ── Layer 9: VRAM / OOM ──
        r = LayerResult(9, "VRAM / OOM Risk")
        log_tail = ""
        if self.log_path.exists():
            log_tail = self.log_path.read_text()[-4096:]
        oom_keywords = ["CUDA out of memory", "OOM", "OutOfMemoryError"]
        if any(kw in log_tail for kw in oom_keywords):
            r.findings.append("OOM detected in training log!")
            r.status = "ALERT"
            r.alerts.append("OOM_DETECTED")
        else:
            r.findings.append("No OOM detected in training log")
            r.status = "OK"

        # Check VRAM from metrics if available (VERL)
        mem_metrics = []
        for s in self.steps:
            d = s.get("data", {})
            for key in ["actor/perf/max_memory_allocated_gb", "critic/perf/max_memory_allocated_gb"]:
                if key in d:
                    mem_metrics.append(d[key])
        if mem_metrics:
            r.metrics["max_vram_gb"] = max(mem_metrics)
            r.findings.append(f"Max VRAM allocated: {max(mem_metrics):.1f} GB")
        self._add_result(r)

        # ── Layer 10: Validation Metrics ──
        r = LayerResult(10, "Validation Metrics")
        val_files = list(self.val_dir.glob("*.jsonl")) if self.val_dir.exists() else []
        if val_files:
            # Parse last validation file
            latest_val = sorted(val_files)[-1]
            val_records = []
            with open(latest_val) as f:
                for line in f:
                    if line.strip():
                        try:
                            val_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            val_scores = [r_.get("score", 0) for r_ in val_records]
            if val_scores:
                r.metrics = {
                    "val_samples": len(val_scores),
                    "val_avg_score": sum(val_scores) / len(val_scores),
                    "val_max_score": max(val_scores),
                    "val_min_score": min(val_scores),
                }
                r.findings.append(f"Validation: avg={r.metrics['val_avg_score']:.4f}, n={len(val_scores)}")
                r.status = "OK"
            else:
                r.findings.append("Validation files exist but contain no records")
                r.status = "WARN"
        else:
            r.findings.append("No validation data found (test_freq may not have triggered yet)")
            r.status = "MISSING_DATA"
        self._add_result(r)

        # ── Layer 11: Reward Function ──
        r = LayerResult(11, "Reward Function")
        reward_path = self.config.get("reward_path") or "custom_rewards/deepscaler_reward.py"
        abs_reward = Path(reward_path)
        if not abs_reward.exists():
            abs_reward = Path.cwd() / reward_path
        if abs_reward.exists():
            reward_code = abs_reward.read_text()
            # Check for binary reward pattern
            has_binary = "return 1.0" in reward_code or "return 0.0" in reward_code or 'score = 1' in reward_code
            has_style = "style" in reward_code and "rule" in reward_code
            r.metrics = {
                "reward_path": str(abs_reward),
                "code_lines": len(reward_code.split("\n")),
                "has_binary_reward": has_binary,
                "has_rule_style": has_style,
            }
            if has_binary:
                r.findings.append("Binary reward detected (0/1 only) — no partial credit, reward may be sparse")
                r.alerts.append("BINARY_REWARD")
            if has_style:
                r.findings.append("Rule-style reward — compares answer against ground_truth")
            r.status = "OK"
        else:
            r.findings.append(f"Reward function not found at {reward_path}")
            r.status = "MISSING_DATA"
        self._add_result(r)

        # ── Layer 12: Trajectory Samples (MANDATORY — cannot skip) ──
        r = LayerResult(12, "Trajectory Samples")
        samples = self.collect_trajectory_samples(n_high=2, n_low=2)
        if samples:
            r.samples = samples
            for s in samples:
                r.findings.append(
                    f"Step {s['step']}: {s['n_score_1']}/{s['n_total']} correct ({s['n_score_1']/s['n_total']*100:.0f}%). "
                    f"score=1 output: '{s['score_1_example']['output'][:100]}...'  |  "
                    f"score=0 output: '{s['score_0_example']['output'][:100]}...'"
                )

            # Check for quality issues
            all_ones = [s for s in samples if s["n_score_1"] == s["n_total"]]
            all_zeros = [s for s in samples if s["n_score_0"] == s["n_total"]]
            if all_zeros:
                r.findings.append("⚠️ Some steps have 100% zero-score — batch is too hard or model completely fails")
                r.alerts.append("ALL_ZERO_BATCHES")

            # Check if score=1 and score=0 outputs look qualitatively similar
            # (both doing math reasoning, just answer wrong ≠ model degradation)
            r.status = "OK"
        else:
            r.findings.append("No trajectory files found — cannot verify output quality")
            r.status = "MISSING_DATA"
            r.alerts.append("NO_TRAJECTORY_DATA")
        self._add_result(r)

        # ── Layer 13: Data Difficulty ──
        r = LayerResult(13, "Data Difficulty")
        if n == 0:
            r.status = "MISSING_DATA"
        else:
            # From the metrics pattern itself:
            # - Rmax always 1.0 = easy problems exist in every batch
            # - Rmin always 0.0 = hard problems exist in every batch
            score_spread = sum(self.score_maxs[i] - self.score_mins[i] for i in range(n)) / n
            r.metrics = {
                "avg_score_spread": score_spread,
                "rmax_always_1": all(m == 1.0 for m in self.score_maxs),
                "rmin_always_0": all(m == 0.0 for m in self.score_mins),
            }
            if r.metrics["rmax_always_1"] and r.metrics["rmin_always_0"]:
                r.findings.append("Dataset has wide difficulty range — easy and hard problems in every batch")
                r.findings.append("Reward oscillation is primarily driven by batch sampling, not training dynamics")
                r.alerts.append("WIDE_DIFFICULTY_RANGE")
            r.status = "OK"
        self._add_result(r)

        # ── Layer 14: Cross-Metric Correlation ──
        r = LayerResult(14, "Cross-Metric Correlation")
        if n < 5:
            r.status = "MISSING_DATA"
        else:
            def corr(a, b):
                ma, mb = sum(a) / len(a), sum(b) / len(b)
                sa = (sum((x - ma) ** 2 for x in a) / len(a)) ** 0.5
                sb = (sum((x - mb) ** 2 for x in b) / len(b)) ** 0.5
                if sa == 0 or sb == 0:
                    return 0
                return sum((a[i] - ma) * (b[i] - mb) for i in range(len(a))) / (len(a) * sa * sb)

            pairs = [
                ("reward_vs_grad", self.scores, self.grads),
                ("reward_vs_entropy", self.scores, self.ents),
                ("reward_vs_resp_len", self.scores, self.resp_lens),
                ("reward_vs_advantage", self.scores, self.advs),
                ("reward_vs_clipfrac", self.scores, self.clipfracs),
                ("reward_vs_kl", self.scores, self.kls),
            ]
            r.metrics = {}
            for name, a, b in pairs:
                if a and b:
                    r.metrics[name] = corr(a, b)

            # Interpret
            interpretations = []
            if abs(r.metrics.get("reward_vs_grad", 0)) > 0.3:
                interpretations.append(f"R↔Grad r={r.metrics['reward_vs_grad']:+.2f} — reward drives gradient signal")
            else:
                interpretations.append(f"R↔Grad r={r.metrics.get('reward_vs_grad', 0):+.2f} — weak coupling")

            if abs(r.metrics.get("reward_vs_entropy", 0)) > 0.3:
                interpretations.append(f"R↔Ent r={r.metrics['reward_vs_entropy']:+.2f} — easier problems have lower entropy")
            else:
                interpretations.append(f"R↔Ent r={r.metrics.get('reward_vs_entropy', 0):+.2f} — entropy not reward-driven")

            if abs(r.metrics.get("reward_vs_kl", 0)) < 0.01 and abs(r.metrics.get("reward_vs_clipfrac", 0)) < 0.01:
                interpretations.append("R↔KL≈0 & R↔clipfrac≈0 — model parameters barely changing, reward = sampling noise")
                r.alerts.append("NO_LEARNING_CONFIRMED")

            r.findings = interpretations
            r.status = "OK"
        self._add_result(r)

        # ── Layer 15: Synthesis & Tuning Suggestions ──
        r = LayerResult(15, "Synthesis & Tuning Suggestions")

        # Collect all alerts from previous layers
        all_alerts = []
        for prev in self.results:
            all_alerts.extend(prev.alerts)

        # Classify the situation
        is_learning = "NO_LEARNING_CONFIRMED" not in all_alerts and "POLICY_NOT_UPDATING" not in all_alerts
        is_crashed = "ENTROPY_EXPLOSION" in all_alerts or "ENTROPY_COLLAPSE" in all_alerts
        has_warning = len(all_alerts) > 0

        suggestions = []

        if is_crashed:
            # Crash recovery comes first
            suggestions.append({
                "priority": 0,
                "action": "RECOVER_FROM_CRASH",
                "details": "Training crashed — must fix root cause before continuing",
                "changes": {}
            })
            if "ENTROPY_EXPLOSION" in all_alerts:
                # ── Differential diagnosis for entropy explosion ──
                # Not all S2 is caused by temperature. Check the actual config first.
                current_temp = self.config.get("temperature", 0.7)
                current_gen = self.config.get("num_generations", 4)
                current_lr = self.config.get("learning_rate", 5e-6)
                current_ppo_ep = self.config.get("ppo_epochs", 1)
                current_batch = self.config.get("batch_size", 16)

                # Check grad spike pattern: did grad_norm spike before entropy exploded?
                grad_spike_detected = False
                if len(self.grads) >= 5:
                    early_grads = self.grads[:max(1, len(self.grads)//2)]
                    late_grads = self.grads[len(self.grads)//2:]
                    if early_grads and late_grads:
                        early_avg = sum(early_grads)/len(early_grads)
                        late_max = max(late_grads)
                        if late_max > early_avg * 5:  # 5x spike over early average
                            grad_spike_detected = True

                # Temperature already safe?
                if current_temp <= 0.72:
                    suggestions.append({
                        "priority": 1,
                        "param": "temperature",
                        "direction": "KEEP",
                        "reason": f"Temperature already at safe level ({current_temp}). Entropy explosion NOT caused by temperature — look at update strength (lr, ppo_epochs).",
                        "from_layer": 6,
                    })
                else:
                    suggestions.append({
                        "priority": 1,
                        "param": "temperature",
                        "direction": "DECREASE",
                        "reason": f"Temperature ({current_temp}) above safe baseline (0.7). Reduce to prevent sampling drift.",
                        "from_layer": 6,
                    })

                # Generations already at minimum?
                if current_gen <= 4:
                    suggestions.append({
                        "priority": 2,
                        "param": "num_generations",
                        "direction": "KEEP",
                        "reason": f"num_generations already at minimum ({current_gen}). Not the cause of explosion.",
                        "from_layer": 3,
                    })
                else:
                    suggestions.append({
                        "priority": 2,
                        "param": "num_generations",
                        "direction": "DECREASE",
                        "reason": f"num_generations ({current_gen}) above stable baseline (4). Reduce to decrease noise amplification.",
                        "from_layer": 3,
                    })

                # Grad spike → lr too high
                if grad_spike_detected:
                    suggestions.append({
                        "priority": 1,
                        "param": "learning_rate",
                        "direction": "DECREASE",
                        "reason": f"grad_norm spiked {max(self.grads):.1f}x over early average before entropy exploded. lr is too high for 7B stability.",
                        "from_layer": 4,
                    })

                # Effective update strength analysis
                eff = current_lr * current_batch * current_ppo_ep
                if eff > 3e-4:
                    suggestions.append({
                        "priority": 1,
                        "param": "ppo_epochs",
                        "direction": "DECREASE",
                        "reason": f"Effective update strength (lr×batch×ppo_ep={eff:.0e}) is too high. Cut ppo_epochs to reduce per-batch parameter change.",
                        "from_layer": 4,
                    })
                elif eff < 1e-4:
                    suggestions.append({
                        "priority": 1,
                        "param": "learning_rate",
                        "direction": "INCREASE",
                        "reason": f"Effective update strength ({eff:.0e}) too low for 7B. Increase lr to generate PPO_KL signal.",
                        "from_layer": 4,
                    })
        elif not is_learning and not is_crashed:
            # Training is running but not learning
            suggestions.append({
                "priority": 1,
                "action": "STRENGTHEN_UPDATE_SIGNAL",
                "details": "PPO_KL=0 + clipfrac=0 + advantage≈0 — policy is not updating. Need stronger GRPO signal.",
                "from_layers": [2, 3, 4, 14],
            })
            if "DIFFICULTY_DRIVEN_OSCILLATION" in all_alerts:
                suggestions.append({
                    "priority": 1,
                    "param": "train_batch_size",
                    "direction": "INCREASE",
                    "reason": "Larger batches reduce inter-batch difficulty variance, making reward trend more meaningful.",
                    "from_layer": 1,
                })
            if "UPDATE_TOO_WEAK" in all_alerts:
                suggestions.append({
                    "priority": 1,
                    "param": "learning_rate",
                    "direction": "MAY_INCREASE",
                    "reason": "clipfrac=0 means updates never hit boundary. Can safely increase lr to strengthen updates.",
                    "from_layer": 4,
                })
                suggestions.append({
                    "priority": 2,
                    "param": "ppo_epochs",
                    "direction": "INCREASE",
                    "reason": "Multiple PPO epochs per batch may strengthen policy update without increasing noise.",
                    "from_layer": 4,
                })
            if "GRPO_SIGNAL_WEAK" in all_alerts:
                suggestions.append({
                    "priority": 2,
                    "param": "entropy_coeff",
                    "direction": "INCREASE",
                    "reason": "Weak advantage signal can be improved by encouraging exploration diversity within groups.",
                    "from_layer": 3,
                })
        else:
            suggestions.append({
                "priority": 1,
                "action": "CONTINUE_WITH_CAUTION",
                "details": "Training appears healthy. Monitor for plateaus and adjust conservatively.",
            })

        # Check reward gap
        reward_gap = self.target - (sum(self.scores) / len(self.scores)) if self.scores else 1.0
        if reward_gap > 0.2 and not is_crashed:
            suggestions.append({
                "priority": 3,
                "action": "LARGE_REWARD_GAP",
                "details": f"Reward gap of {reward_gap:.2f} remains. Multi-round tuning needed."
            })

        r.metrics = {
            "total_alerts": len(all_alerts),
            "alert_types": list(set(all_alerts)),
            "is_learning": is_learning,
            "is_crashed": is_crashed,
            "reward_gap": reward_gap,
        }
        r.findings = [f"Alerts: {all_alerts}" if all_alerts else "No alerts — training healthy"]
        r.samples = suggestions  # reuse samples field for suggestions
        r.status = "ALERT" if is_crashed else ("WARN" if has_warning else "OK")
        self._add_result(r)

    # ── Output ──

    def to_dict(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "backend": "verl",
            "target_reward": self.target,
            "n_steps": len(self.scores),
            "reward_avg": sum(self.scores) / len(self.scores) if self.scores else 0,
            "layers": [
                {
                    "id": r.layer_id,
                    "name": r.name,
                    "status": r.status,
                    "findings": r.findings,
                    "metrics": r.metrics,
                    "samples": r.samples,
                    "alerts": r.alerts,
                }
                for r in self.results
            ],
            "tuning_suggestions": self.results[-1].samples if self.results else [],
        }

    def save(self, output_path: str | None = None):
        if output_path is None:
            output_path = self.run_dir / "deep_analysis.json"
        else:
            output_path = Path(output_path)

        report = self.to_dict()
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return output_path

    def print_report(self):
        """Human-readable report to stdout."""
        report = self.to_dict()
        print(f"Deep Analysis Report: {report['run_dir']}")
        print(f"Backend: {report['backend']} | Steps: {report['n_steps']} | Target: {report['target_reward']}")
        print(f"Reward: avg={report['reward_avg']:.4f}")
        print()

        for layer in report["layers"]:
            icon = {"OK": "✅", "WARN": "⚠️", "ALERT": "🔴", "MISSING_DATA": "❓", "SKIPPED": "⏭️", "PENDING": "⏳"}
            print(f"  L{layer['id']:2d} {icon.get(layer['status'], '?')} {layer['name']:<30s} [{layer['status']}]")
            for f in layer["findings"]:
                print(f"      {f}")
            for a in layer.get("alerts", []):
                print(f"      🚨 {a}")
            if layer["id"] == 12 and layer.get("samples"):
                print(f"      [Trajectory samples available — see deep_analysis.json for full content]")
            print()

        print("Tuning Suggestions:")
        for s in report.get("tuning_suggestions", []):
            print(f"  P{s.get('priority','?')}: {s.get('action','?')}")
            if s.get("param"):
                print(f"      {s['param']}: {s.get('direction','?')}")
            print(f"      {s.get('reason','?')}")
            print()


# ── CLI entry ──

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deep_analyzer.py <run_dir> [--target 0.7] [--output path]")
        sys.exit(1)

    run_dir = sys.argv[1]
    target = 0.7
    output = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--target" and i + 1 < len(sys.argv):
            target = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    analyzer = DeepAnalyzer(run_dir, target)
    if not analyzer.collect_metrics():
        print(f"ERROR: Cannot load metrics from {analyzer.metrics_path}")
        sys.exit(1)

    analyzer.run_all_layers()
    saved_path = analyzer.save(output)
    analyzer.print_report()
    print(f"\nFull report saved to: {saved_path}")

    # Exit code: non-zero if alerts found
    has_critical = any(r.status == "ALERT" for r in analyzer.results)
    sys.exit(2 if has_critical else 0)
