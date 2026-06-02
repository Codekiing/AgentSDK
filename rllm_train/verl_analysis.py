"""
VERL training metrics extractor for rllm-analyze-deep.

Parses VERL training_log.txt (Ray worker + Tracking console output)
into structured metric time series compatible with the 16-layer diagnosis system.
"""

import json
import re
from pathlib import Path
from typing import Any


# Metric extraction patterns
_PATTERNS = {
    "step": re.compile(r"(?:step|global_step)\s*[:=]\s*(\d+)", re.I),
    "epoch": re.compile(r"epoch\s*[:=]\s*([\d.]+)", re.I),
    "score_mean": re.compile(r"(?:score|rewards)/mean\s*[:=]\s*([\d.-]+)"),
    "score_std": re.compile(r"(?:score|rewards)/std\s*[:=]\s*([\d.-]+)"),
    "grad_norm": re.compile(r"grad_norm\s*[:=]\s*([\d.-]+)"),
    "pg_loss": re.compile(r"pg_loss\s*[:=]\s*([\d.-]+)"),
    "policy_loss": re.compile(r"policy_loss\s*[:=]\s*([\d.-]+)"),
    "entropy": re.compile(r"entropy\s*[:=]\s*([\d.-]+)"),
    "lr": re.compile(r"(?:actor/)?lr\s*[:=]\s*([\d.e-]+)"),
    "kl": re.compile(r"(?:ppo_kl|approx_kl)\s*[:=]\s*([\d.-]+)"),
    "clipfrac": re.compile(r"(?:pg_clipfrac|clipfrac)\s*[:=]\s*([\d.-]+)"),
    "throughput": re.compile(r"(?:throughput|tokens_per_second)\s*[:=]\s*([\d.-]+)"),
    "step_time": re.compile(r"step_time\s*[:=]\s*([\d.-]+)"),
    "response_length": re.compile(r"(?:response_)?length/mean\s*[:=]\s*([\d.-]+)"),
    "error": re.compile(r"Traceback|Error|OOM|out of memory|Killed", re.I),
    "complete": re.compile(r"Training (?:completed|finished)", re.I),
    "val_score": re.compile(r"val(?:-core)?/.*?(?:score|reward)/mean@?\d*\s*[:=]\s*([\d.-]+)"),
}


def extract_verl_metrics(log_path: str) -> dict[str, Any]:
    """Extract structured metrics from a VERL training log.

    Returns a dict with:
        - rewards: list of [step, score_mean]
        - losses: list of [step, pg_loss]
        - grad_norms: list of [step, grad_norm]
        - entropies: list of [step, entropy]
        - throughputs: list of [step, tokens_per_second]
        - val_scores: list of [step, val_score_mean]
        - completed: bool
        - errors: list of error messages
        - total_steps: int
        - summary: dict with final metrics
    """
    if not Path(log_path).exists():
        return {"error": f"Log file not found: {log_path}", "completed": False}

    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()

    current_step = 0
    current_metrics: dict[str, float] = {}
    steps: list[dict] = []
    errors: list[str] = []
    val_scores: list[tuple[int, float]] = []
    completed = False

    for line in lines:
        # Check for errors
        if _PATTERNS["error"].search(line):
            if not re.search(r"WARNING|UserWarning|FutureWarning|Deprecated", line):
                errors.append(line.strip()[:200])

        # Check for completion
        if _PATTERNS["complete"].search(line):
            completed = True

        # Extract step number
        step_m = _PATTERNS["step"].search(line)
        if step_m:
            new_step = int(step_m.group(1))
            if new_step != current_step:
                # Save previous step
                if current_metrics:
                    current_metrics["step"] = current_step
                    steps.append(dict(current_metrics))
                current_step = new_step
                current_metrics = {}

        # Extract metrics
        for key, pattern in _PATTERNS.items():
            if key in ("step", "epoch", "error", "complete", "val_score"):
                continue
            m = pattern.search(line)
            if m:
                try:
                    current_metrics[key] = float(m.group(1))
                except ValueError:
                    pass

        # Extract val scores
        val_m = _PATTERNS["val_score"].search(line)
        if val_m:
            try:
                val_scores.append((current_step, float(val_m.group(1))))
            except ValueError:
                pass

    # Save last step
    if current_metrics:
        current_metrics["step"] = current_step
        steps.append(dict(current_metrics))

    # Build time series
    def _series(key: str) -> list:
        return [[s["step"], s[key]] for s in steps if key in s]

    rewards = _series("score_mean")
    losses = _series("pg_loss") or _series("policy_loss")
    grad_norms = _series("grad_norm")
    entropies = _series("entropy")
    throughputs = _series("throughput")

    # Summary
    final_reward = rewards[-1][1] if rewards else None
    max_reward = max(r[1] for r in rewards) if rewards else None
    final_loss = losses[-1][1] if losses else None

    return {
        "completed": completed,
        "total_steps": current_step,
        "num_error_events": len(errors),
        "errors": errors[-5:] if errors else [],
        "rewards": rewards,
        "losses": losses,
        "grad_norms": grad_norms,
        "entropies": entropies,
        "throughputs": throughputs,
        "val_scores": [[s, v] for s, v in val_scores],
        "summary": {
            "final_reward": final_reward,
            "max_reward": max_reward,
            "final_loss": final_loss,
            "reward_trend": [r[1] for r in rewards[-10:]] if rewards else [],
            "loss_trend": [l[1] for l in losses[-10:]] if losses else [],
            "completed": completed,
        },
    }


def extract_and_save(log_path: str, output_path: str | None = None) -> str:
    """Extract VERL metrics and save as JSON. Returns output path."""
    metrics = extract_verl_metrics(log_path)
    if output_path is None:
        output_path = str(Path(log_path).parent / "verl_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python {__file__} <training_log.txt> [output.json]")
        sys.exit(1)
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    path = extract_and_save(log_path, out_path)
    print(f"Metrics saved to: {path}")
    metrics = extract_verl_metrics(log_path)
    summary = metrics.get("summary", {})
    print(f"Steps: {metrics['total_steps']}, "
          f"Final reward: {summary.get('final_reward')}, "
          f"Completed: {metrics['completed']}")
