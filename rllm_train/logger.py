"""
Training logger with real-time progress table and post-training report.
"""

import os
import time
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    step: int = 0
    num_trajectories: int = 0
    avg_reward: float = 0.0
    rollout_time: float = 0.0
    tokens_per_second: float = 0.0
    loss: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    reward_from_fn: float = 0.0
    clipped_ratio: float = 0.0
    mean_terminated_length: float = 0.0
    step_time: float = 0.0
    reward_variance: float = 0.0
    completion_length_mean: float = 0.0
    finish_rate: float = 0.0
    finish_format_rate: float = 0.0
    tool_usage_rate: float = 0.0
    answer_coverage: float = 0.0
    truncation_rate: float = 0.0
    parser_error_rate: float = 0.0
    malformed_tool_rate: float = 0.0
    synthetic_finish_rate: float = 0.0
    calculator_error_rate: float = 0.0
    symbolic_calculator_error_rate: float = 0.0
    successful_calculate_rate: float = 0.0
    no_finish_rate: float = 0.0
    valid_finish_rate: float = 0.0
    avg_agent_steps: float = 0.0
    pass1: float = 0.0
    passK: float = 0.0


class TrainingLogger:
    def __init__(self, verbose=True, log_file=None):
        self.verbose = verbose
        self.total_steps: int = 0
        self.num_epochs: int = 0
        self.total_trajectories: int = 0
        self.start_time: float = 0.0
        self._rollout_start: float = 0.0
        self._header_printed = False
        self.max_completion_length: int = 0
        self._length_limit_hits: int = 0

        self.step_records: list[StepRecord] = []
        self._current: StepRecord | None = None
        self._trl_metrics: list[dict] = []
        self._monitor_emitted_complete: set[int] = set()  # steps with full TRL metrics emitted

        self._log_file = None
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._log_file = open(log_file, "w")

    def _print(self, text=""):
        print(text)
        if self._log_file:
            self._log_file.write(text + "\n")
            self._log_file.flush()

    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _fmt_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    # ── Setup phase ──────────────────────────────────────────

    def log_training_start(self, config):
        self.start_time = time.time()
        self.max_completion_length = int(getattr(config, "max_completion_length", 0) or 0)
        self._print(config.summary())
        self._print()

    def log_model_loaded(self, model_name, device, param_count=None):
        param_str = f", {param_count/1e6:.1f}M params" if param_count else ""
        self._print(f"  Model loaded: {model_name} ({device}{param_str})")

    def log_dataset_ready(self, size, sample=None):
        self._print(f"  Dataset ready: {size} problems")
        summary = sample if isinstance(sample, dict) else getattr(sample, "filter_summary", None)
        if summary:
            self._print(
                "  Dataset filter: "
                f"{summary['filter']} "
                f"({summary['original_size']} -> {summary['filtered_size']} -> {summary['selected_size']})"
            )

    def log_trainer_ready(self, num_epochs):
        self.num_epochs = num_epochs
        self._print(f"  Trainer ready: {num_epochs} epoch(s)")
        self._print()

    # ── Progress table ───────────────────────────────────────

    def _print_header(self):
        if self._header_printed:
            return
        self._header_printed = True
        header = (
            f"{'Step':>6}  {'Trajs':>5}  {'Reward':>7}  "
            f"{'Loss':>8}  {'Entropy':>8}  {'GradNorm':>9}  "
            f"{'Rollout':>8}  {'tok/s':>7}  {'ETA':>8}"
        )
        self._print(header)
        self._print("─" * len(header))

    def _print_step_row(self, rec: StepRecord):
        elapsed = time.time() - self.start_time
        remaining_steps = self.total_steps - rec.step if self.total_steps > 0 else 0
        avg_step = elapsed / rec.step if rec.step else 0
        eta = self._fmt_time(remaining_steps * avg_step) if avg_step and remaining_steps > 0 else "—"

        tps_str = f"{rec.tokens_per_second:.1f}" if rec.tokens_per_second > 0 else "—"
        loss_str = f"{rec.loss:.4f}" if rec.loss != 0 else "—"
        entropy_str = f"{rec.entropy:.4f}" if rec.entropy != 0 else "—"
        grad_str = f"{rec.grad_norm:.4f}" if rec.grad_norm != 0 else "—"

        total_str = str(self.total_steps) if self.total_steps > 0 else "?"
        row = (
            f"{rec.step:>3}/{total_str:<3} "
            f"{rec.num_trajectories:>5}  "
            f"{rec.avg_reward:>7.3f}  "
            f"{loss_str:>8}  "
            f"{entropy_str:>8}  "
            f"{grad_str:>9}  "
            f"{self._fmt_time(rec.rollout_time):>8}  "
            f"{tps_str:>7}  "
            f"{eta:>8}"
        )
        self._print(row)

    # ── Called from rollout.py ────────────────────────────────

    def log_rollout_start(self, step, num_prompts, num_generations):
        self._rollout_start = time.time()
        self._current = StepRecord(step=step)
        self._print_header()
        total_str = str(self.total_steps) if self.total_steps > 0 else "?"
        self._print(f"    ··· step {step}/{total_str}: generating {num_generations} trajectories...")

    def log_rollout_done(self, step, num_trajectories, rewards=None, rollout_stats=None, behavior_metrics=None):
        elapsed = time.time() - self._rollout_start
        self.total_trajectories += num_trajectories

        rec = self._current or StepRecord(step=step)
        rec.num_trajectories = num_trajectories
        rec.rollout_time = elapsed
        if rewards:
            rec.avg_reward = sum(rewards) / len(rewards)
        if rollout_stats:
            rec.tokens_per_second = rollout_stats.tokens_per_second
            rec.completion_length_mean = rollout_stats.avg_response_len
        if behavior_metrics:
            rec.reward_variance = float(behavior_metrics.get("reward_variance", 0.0))
            rec.completion_length_mean = float(behavior_metrics.get("completion_length_mean", rec.completion_length_mean))
            rec.finish_rate = float(behavior_metrics.get("finish_rate", 0.0))
            rec.finish_format_rate = float(behavior_metrics.get("finish_format_rate", 0.0))
            rec.tool_usage_rate = float(behavior_metrics.get("tool_usage_rate", 0.0))
            rec.answer_coverage = float(behavior_metrics.get("answer_coverage", 0.0))
            rec.truncation_rate = float(behavior_metrics.get("truncation_rate", 0.0))
            rec.parser_error_rate = float(behavior_metrics.get("parser_error_rate", 0.0))
            rec.malformed_tool_rate = float(behavior_metrics.get("malformed_tool_rate", 0.0))
            rec.synthetic_finish_rate = float(behavior_metrics.get("synthetic_finish_rate", 0.0))
            rec.calculator_error_rate = float(behavior_metrics.get("calculator_error_rate", 0.0))
            rec.symbolic_calculator_error_rate = float(behavior_metrics.get("symbolic_calculator_error_rate", 0.0))
            rec.successful_calculate_rate = float(behavior_metrics.get("successful_calculate_rate", 0.0))
            rec.no_finish_rate = float(behavior_metrics.get("no_finish_rate", 0.0))
            rec.valid_finish_rate = float(behavior_metrics.get("valid_finish_rate", 0.0))
            rec.pass1 = float(behavior_metrics.get("pass1", 0.0))
            rec.passK = float(behavior_metrics.get("passK", 0.0))
            rec.avg_agent_steps = float(behavior_metrics.get("avg_agent_steps", 0.0))

        self._print_header()
        self._print_step_row(rec)
        self.step_records.append(rec)
        self._current = None

        # Emit [MONITOR_STEP] immediately after rollout completes.
        # TRL metrics (loss/grad/ent) are not available yet — they will
        # be filled in by _print_monitor_step when update_training_metrics
        # re-emits after the gradient update.
        self._print_monitor_step(rec)

    # ── Called from RllmCallback.on_log ──────────────────────

    def log_trajectory_done(self, step, idx, total, reward):
        self._print(f"    ··· trajectory {idx}/{total} done (reward={reward:.3f})")

    def log_logprob_start(self, step):
        self._print(f"    ··· computing logprobs...")

    def log_training_update_start(self, step):
        self._print(f"    ··· training update...")

    def update_training_metrics(self, logs: dict):
        self._trl_metrics.append(logs)

        target = self._current
        if target is None and self.step_records:
            target = self.step_records[-1]

        if target is not None:
            if "loss" in logs and target.loss == 0:
                target.loss = float(logs["loss"])
            if "entropy" in logs and target.entropy == 0:
                target.entropy = float(logs["entropy"])
            if "grad_norm" in logs and target.grad_norm == 0:
                target.grad_norm = float(logs["grad_norm"])
            if "learning_rate" in logs and target.lr == 0:
                target.lr = float(logs["learning_rate"])
            if "reward" in logs and target.avg_reward == 0:
                target.avg_reward = float(logs["reward"])
            if "completions/clipped_ratio" in logs:
                target.clipped_ratio = float(logs["completions/clipped_ratio"])
            if "completions/mean_terminated_length" in logs:
                target.mean_terminated_length = float(logs["completions/mean_terminated_length"])
            if "step_time" in logs:
                target.step_time = float(logs["step_time"])

        # Output parseable summary lines for monitor
        self._print_trl_summary(logs)
        if target is not None:
            self._print_monitor_step(target)

    def _print_trl_summary(self, logs: dict):
        """Output a grep-friendly summary line from TRL metrics."""
        reward = float(logs.get("reward", 0))
        loss = float(logs.get("loss", 0))
        entropy = float(logs.get("entropy", 0))
        grad_norm = float(logs.get("grad_norm", 0))
        clipped = float(logs.get("completions/clipped_ratio", 0))
        epoch = logs.get("epoch", "?")
        step_time = float(logs.get("step_time", 0))

        # Format: [TRL_STEP] key=value pairs — easy to grep/parse
        line = (
            f"[TRL_STEP] reward={reward:.4f} loss={loss:.6f} "
            f"entropy={entropy:.4f} grad_norm={grad_norm:.4f} "
            f"clipped_ratio={clipped:.2f} epoch={epoch} "
            f"step_time={step_time:.1f}s"
        )
        self._print(line)

    def _print_monitor_step(self, rec: StepRecord):
        # Emit [MONITOR_STEP] line. Called twice per step:
        #   1. From log_rollout_done() — partial (loss/grad/ent not yet set)
        #   2. From update_training_metrics() — complete after TRL gradient update
        # The poll mode handles dedup by keeping the last (most complete) line per step.

        length_limit = False
        if self.max_completion_length > 0:
            length_limit = rec.completion_length_mean >= self.max_completion_length * 0.9
        length_limit = length_limit or rec.clipped_ratio >= 0.8
        if length_limit:
            self._length_limit_hits += 1
        else:
            self._length_limit_hits = 0

        if self._length_limit_hits >= 3:
            status = "STOP(length-limit)"
        elif self._length_limit_hits >= 2:
            status = "WARN(length-limit)"
        else:
            status = "OK"

        total_str = str(self.total_steps) if self.total_steps > 0 else "?"
        eta = "—"
        elapsed = time.time() - self.start_time
        if rec.step and self.total_steps > rec.step:
            avg_step = elapsed / rec.step
            eta = self._fmt_time((self.total_steps - rec.step) * avg_step)

        # Use "—" for TRL-only metrics when not yet set (partial emission)
        loss_str = f"{rec.loss:.4f}" if rec.loss != 0 else "—"
        grad_str = f"{rec.grad_norm:.4f}" if rec.grad_norm != 0 else "—"
        ent_str = f"{rec.entropy:.4f}" if rec.entropy != 0 else "—"
        clip_str = f"{rec.clipped_ratio:.2f}" if rec.clipped_ratio != 0 else "—"

        line = (
            f"[MONITOR_STEP] Step {rec.step}/{total_str} "
            f"| R {rec.avg_reward:.3f} "
            f"| Rstd {rec.reward_variance:.3f} "
            f"| Loss {loss_str} "
            f"| Grad {grad_str} "
            f"| Ent {ent_str} "
            f"| Clip {clip_str} "
            f"| Len {rec.completion_length_mean:.0f} "
            f"| Finish {rec.finish_rate * 100:.0f}% "
            f"| FmtOK {rec.finish_format_rate * 100:.0f}% "
            f"| Tool {rec.tool_usage_rate * 100:.0f}% "
            f"| Ans {rec.answer_coverage * 100:.0f}% "
            f"| CalcErr {rec.calculator_error_rate * 100:.0f}% "
            f"| SymErr {rec.symbolic_calculator_error_rate * 100:.0f}% "
            f"| ParseErr {rec.parser_error_rate * 100:.0f}% "
            f"| NoFinish {rec.no_finish_rate * 100:.0f}% "
            f"| pass@1 {rec.pass1*100:.0f}% "
            f"| pass@K {rec.passK*100:.0f}% "
            f"| tok/s {rec.tokens_per_second:.1f} "
            f"| Time {rec.step_time or rec.rollout_time:.1f}s "
            f"| ETA ~{eta} "
            f"| Status {status}"
        )
        self._print(line)

    def set_perf_stats(self, rollout_stats):
        if not self.step_records or not rollout_stats:
            return
        for rec, rs in zip(self.step_records, rollout_stats):
            rec.tokens_per_second = rs.tokens_per_second

    # ── End of training ──────────────────────────────────────

    def log_training_done(self):
        pass

    def log_model_saved(self, path):
        self._print(f"  Model saved to {path}")

    # ── Training Report ──────────────────────────────────────

    def print_training_report(self, config, perf_summary, output_dir, eval_metrics=None):
        elapsed = time.time() - self.start_time
        records = self.step_records
        trl = self._trl_metrics

        self._print()
        self._print("=" * 60)
        self._print("                    Training Report")
        self._print("=" * 60)

        self._print()
        self._print("  Training Overview")
        self._print(f"    Model:          {config.model_name}")
        self._print(f"    Dataset:        {config.num_problems} problems, {config.num_epochs} epoch(s)")
        eval_split = getattr(config, 'eval_split', 0)
        if eval_split > 0:
            self._print(f"    Train/Eval:     {int(config.num_problems * (1 - eval_split))} / {int(config.num_problems * eval_split)}")
        self._print(f"    Total:          {len(records)} steps, {self.total_trajectories} trajectories, {self._fmt_time(elapsed)}")

        # Test set evaluation section
        if eval_metrics is not None:
            self._print()
            self._print("  Test Set Evaluation")
            self._print(f"    Test Reward:    {eval_metrics['avg_reward']:.4f} ± {eval_metrics['reward_std']:.4f}")
            self._print(f"    Test Samples:   {eval_metrics['num_samples']}")
            self._print(f"    Finish Rate:    {eval_metrics['finish_rate']*100:.0f}%")
            self._print(f"    Answer Coverage:{eval_metrics['answer_coverage']*100:.0f}%")
            # Overfitting check
            if records:
                train_reward = records[-1].get("reward", 0)
                gap = train_reward - eval_metrics['avg_reward']
                self._print(f"    Train - Test Gap:{gap:+.4f}" + (" (overfitting?)" if gap > 0.15 else ""))

        self._print()
        self._print("  Reward Trend")
        if records:
            self._print_reward_trend(records)

        self._print()
        self._print("  Training Dynamics")
        if trl:
            self._print_training_dynamics(trl)

        self._print()
        self._print("  Performance Breakdown")
        if perf_summary:
            self._print_perf_breakdown(perf_summary)

        self._print()
        self._print("  Output Files")
        traj_dir = os.path.join(output_dir, "trajectories")
        n_traj_files = len([f for f in os.listdir(traj_dir) if f.endswith(".jsonl")]) if os.path.isdir(traj_dir) else 0
        self._print(f"    Trajectories:   {traj_dir}/ ({n_traj_files} files, {self.total_trajectories} trajectories)")
        self._print(f"    Perf stats:     {os.path.join(output_dir, 'perf_stats.json')}")
        if config.save_model:
            self._print(f"    Model:          {os.path.join(output_dir, 'final_model/')}")
        self._print("=" * 60)

    def _print_reward_trend(self, records):
        rewards = [r.avg_reward for r in records]
        n = len(rewards)
        if n <= 2:
            self._print(f"    avg: {sum(rewards)/n:.3f} (min={min(rewards):.3f}, max={max(rewards):.3f})")
            return

        first_half = rewards[:n // 2]
        second_half = rewards[n // 2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if abs(avg_second - avg_first) < 0.001:
            self._print(f"    Stable at {avg_second:.3f} across all steps")
        else:
            delta = avg_second - avg_first
            direction = "+" if delta > 0 else ""
            pct = (delta / avg_first * 100) if avg_first else 0
            self._print(f"    Step 1-{n//2}:     avg {avg_first:.3f}")
            self._print(f"    Step {n//2+1}-{n}:     avg {avg_second:.3f} ({direction}{pct:.1f}%)")

        self._print(f"    Final:          {rewards[-1]:.3f} (min={min(rewards):.3f}, max={max(rewards):.3f})")

    def _print_training_dynamics(self, trl_logs):
        losses = [float(l.get("loss", 0)) for l in trl_logs if "loss" in l]
        entropies = [float(l.get("entropy", 0)) for l in trl_logs if "entropy" in l]
        grad_norms = [float(l.get("grad_norm", 0)) for l in trl_logs if "grad_norm" in l]

        if losses:
            first, last = losses[0], losses[-1]
            if first > 0:
                change = (last - first) / first * 100
                self._print(f"    Loss:           {first:.4f} -> {last:.4f} ({change:+.1f}%)")
            else:
                self._print(f"    Loss:           {first:.4f} -> {last:.4f}")

        if entropies:
            self._print(f"    Entropy:        {entropies[0]:.4f} -> {entropies[-1]:.4f}")

        if grad_norms:
            avg_gn = sum(grad_norms) / len(grad_norms)
            self._print(f"    Grad norm:      avg {avg_gn:.4f} (min={min(grad_norms):.4f}, max={max(grad_norms):.4f})")

    def _print_perf_breakdown(self, s):
        tb = s["time_breakdown"]
        ts = s["token_stats"]
        wall = s["total_wall_time"]
        bar_width = 30

        items = [
            ("Rollout", tb["rollout_pct"]),
            ("  LLM", tb["llm_pct"]),
            ("  Env", tb["env_pct"]),
            ("  Logprob", tb["logprob_pct"]),
            ("Training", tb["training_pct"]),
        ]

        self._print(f"    Wall time:      {self._fmt_time(wall)}")
        for label, pct in items:
            filled = int(bar_width * pct / 100)
            bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
            self._print(f"    {label:<12} {bar} {pct:>5.1f}%")

        self._print(f"    Throughput:     {ts['avg_tokens_per_second']} tok/s, avg {ts['avg_response_length']} tokens/response")
