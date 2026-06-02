"""
VERL training monitor for rllm-monitor.
Reads verl_metrics.jsonl (VERL's file logger output) for structured per-step metrics.

Usage:
    python rllm_train/monitor_agent_verl.py stream <output_dir>
    python rllm_train/monitor_agent_verl.py oneshot <output_dir> [--last N]
"""

import json
import os
import re
import sys
import time

_COMPLETE_RE = re.compile(r"Training (?:completed|finished)", re.I)
_ERROR_RE = re.compile(
    r"Traceback|out of memory|Killed|SIGTERM|SIGKILL|"
    r"RuntimeError|RayActorError|Actor died|"
    r"CUDA error|NCCL error|HCCL",
    re.I,
)
_PROGRESS_RE = re.compile(r"Training Progress:\s+(\d+)%", re.I)

# Key metrics to always report, in display order
_KEY_METRICS = [
    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",
    "critic/rewards/mean",
    "actor/pg_loss",
    "actor/entropy",
    "actor/grad_norm",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/lr",
    "response_length/mean",
    "response_length/clip_ratio",
    "perf/mfu/actor_infer",
]


def _emit(line: str) -> None:
    print(line, flush=True)


def _metrics_jsonl_path(output_dir: str) -> str:
    return os.path.join(output_dir, "verl_metrics.jsonl")


def _training_log_path(output_dir: str) -> str:
    return os.path.join(output_dir, "training_log.txt")


def _format_step(step_data: dict) -> str:
    """Format all key metrics for a step into a single line."""
    s = step_data["step"]
    data = step_data["data"]
    parts = [f"Step {s}"]
    for key in _KEY_METRICS:
        v = data.get(key)
        if v is not None:
            if isinstance(v, float):
                parts.append(f"{key}={v:.4f}")
            else:
                parts.append(f"{key}={v}")
    return " | ".join(parts)


def _format_step_compact(step_data: dict) -> str:
    """Compact format: Step N | R 0.xxx | Loss 0.xxx | Grad 0.xxx."""
    s = step_data["step"]
    data = step_data["data"]
    parts = [f"Step {s}"]
    score = data.get("critic/score/mean")
    if score is not None:
        parts.append(f"R {score:.4f}")
    loss = data.get("actor/pg_loss")
    if loss is not None:
        parts.append(f"Loss {loss:+.4f}")
    grad = data.get("actor/grad_norm")
    if grad is not None:
        parts.append(f"Grad {grad:.4f}")
    ent = data.get("actor/entropy")
    if ent is not None:
        parts.append(f"Ent {ent:.3f}")
    clen = data.get("response_length/mean")
    if clen is not None:
        parts.append(f"Len {clen:.0f}")
    return " | ".join(parts)


def _check_errors(log_path: str) -> str | None:
    """Scan training_log.txt for fatal errors. Returns error message or None."""
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r", errors="replace") as f:
            # Only check last 5KB for efficiency
            f.seek(max(0, os.path.getsize(log_path) - 5120))
            tail = f.read()
    except OSError:
        return None
    for line in tail.splitlines():
        if _COMPLETE_RE.search(line):
            return "complete"
        if _ERROR_RE.search(line):
            if not re.search(r"WARNING|UserWarning|FutureWarning|Deprecated|truncation.*error|OMP_NUM_THREADS", line, re.I):
                # Avoid false positives from config values and known benign warnings
                return line.strip()[:200]
    return None


def stream(output_dir: str, timeout: int = 3600) -> None:
    """Stream verl_metrics.jsonl, reporting each new step with all metrics."""
    jsonl_path = _metrics_jsonl_path(output_dir)
    log_path = _training_log_path(output_dir)
    last_step = 0
    start = time.time()

    while not os.path.exists(jsonl_path):
        if time.time() - start > 300:
            _emit(f"TIMEOUT: {jsonl_path} did not appear within 300s")
            return
        time.sleep(3)

    try:
        st = os.stat(jsonl_path)
        inode = st.st_ino
        pos = st.st_size
    except OSError:
        pos = 0

    last_data_time = time.time()

    while True:
        if time.time() - start > timeout:
            _emit("TIMEOUT: monitor stream exceeded time limit")
            return

        # Check for errors every 15s
        if int(time.time()) % 15 == 0:
            err = _check_errors(log_path)
            if err == "complete":
                _emit("DETECTED: Training completed")
                return
            elif err:
                _emit(f"ERROR: {err}")
                return

        try:
            st = os.stat(jsonl_path)
            current_inode = st.st_ino
            current_size = st.st_size
        except OSError:
            time.sleep(3)
            continue

        if current_inode != inode or current_size < pos:
            inode = current_inode
            pos = 0
            last_step = 0

        if current_size > pos:
            last_data_time = time.time()
            try:
                with open(jsonl_path, "r", errors="replace") as f:
                    f.seek(pos)
                    data = f.read()
                    pos = f.tell()
            except (OSError, ValueError):
                time.sleep(3)
                continue

            for line in data.splitlines():
                if not line.strip():
                    continue
                try:
                    step_data = json.loads(line)
                    s = step_data["step"]
                    if s > last_step:
                        last_step = s
                        _emit(_format_step_compact(step_data))
                except (json.JSONDecodeError, KeyError):
                    continue
        else:
            if time.time() - last_data_time > 600:
                _emit("WARN: no new metrics for 10 minutes, training may be stalled")

        time.sleep(5)


def oneshot(output_dir: str, last_n: int = 1) -> None:
    """One-shot: read last N steps from verl_metrics.jsonl."""
    jsonl_path = _metrics_jsonl_path(output_dir)
    log_path = _training_log_path(output_dir)

    # Report metrics first (if available)
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, "r", errors="replace") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
        except OSError:
            lines = []

        if lines:
            for line in lines[-last_n:]:
                try:
                    _emit(_format_step_compact(json.loads(line)))
                except json.JSONDecodeError:
                    continue
            _emit("---")
            try:
                _emit(_format_step(json.loads(lines[-1])))
            except json.JSONDecodeError:
                pass
            return

    # No metrics yet: check for progress or errors
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", errors="replace") as f:
                f.seek(max(0, os.path.getsize(log_path) - 2048))
                tail = f.read()
        except OSError:
            tail = ""

        # Check completion
        if _COMPLETE_RE.search(tail):
            _emit("DETECTED: Training completed")
            return

        # Check progress
        m = _PROGRESS_RE.findall(tail)
        if m:
            _emit(f"PROGRESS: {m[-1]}% complete")
            return

        # Check errors (only when no metrics exist)
        err = _check_errors(log_path)
        if err:
            _emit(f"DETECTED: {err}")
            return

    _emit("ONESHOT: no metrics yet (model loading)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VERL training monitor (reads verl_metrics.jsonl)")
    sub = parser.add_subparsers(dest="mode")

    p_stream = sub.add_parser("stream")
    p_stream.add_argument("output_dir", help="Training output directory (rllm_train/output/runs/<run_id>)")
    p_stream.add_argument("--timeout", type=int, default=3600)

    p_shot = sub.add_parser("oneshot")
    p_shot.add_argument("output_dir", help="Training output directory")
    p_shot.add_argument("--last", type=int, default=1)

    args = parser.parse_args()
    if args.mode == "stream":
        stream(args.output_dir, timeout=args.timeout)
    elif args.mode == "oneshot":
        oneshot(args.output_dir, last_n=args.last)
    else:
        parser.print_help()
        sys.exit(1)
