"""
Robust training log monitor for rllm-monitor.

Usage:
    # Persistent streaming mode (stdout lines are monitor events)
    python -m rllm_train.monitor_agent stream <log_path>

    # One-shot tail mode (for CronCreate fallback, prints latest step)
    python -m rllm_train.monitor_agent oneshot <log_path>

    # One-shot with last N steps summary
    python -m rllm_train.monitor_agent oneshot <log_path> --last 5

    # Poll mode — incremental, stateful, designed for ScheduleWakeup loops.
    # Outputs only NEW [MONITOR_STEP] lines since the last poll.
    # Maintains state in <run_dir>/.monitor_poll_state.json
    python -m rllm_train.monitor_agent poll <log_path>

Handles:
    - File recreation (inode change) → position reset
    - File truncation → position reset
    - Training Report → auto-exit
    - Traceback/OOM → error signal
    - Step deduplication by step number (not text key)
    - Cross-invocation dedup via state file (poll mode)
"""

import json
import os
import re
import sys
import time

_MONITOR_PATTERN = re.compile(
    r"\[MONITOR_STEP\]|Training Report|Traceback|Error|OOM|out of memory", re.I
)
_STEP_NUM_PATTERN = re.compile(r"Step (\d+)/")
_TR_REPORT_PATTERN = re.compile(r"Training Report")


def _emit(line: str) -> None:
    print(line, flush=True)


def _extract_step_num(line: str) -> int | None:
    m = _STEP_NUM_PATTERN.search(line)
    return int(m.group(1)) if m else None


def _handle_line(line: str, seen_steps: set[int]) -> str | None:
    """Process one log line. Returns 'report'|'error'|None."""
    if not _MONITOR_PATTERN.search(line):
        return None

    if _TR_REPORT_PATTERN.search(line):
        _emit(line.strip())
        return "report"

    if any(kw in line for kw in ("Traceback", "Error", "OOM", "out of memory")):
        if "[MONITOR_STEP]" not in line and "[TRL_STEP]" not in line:
            _emit(line.strip())
            return "error"

    if "[MONITOR_STEP]" in line:
        out = line.split("[MONITOR_STEP]", 1)[1].strip()
        step_num = _extract_step_num(out)
        if step_num is not None and step_num in seen_steps:
            return None
        if step_num is not None:
            seen_steps.add(step_num)
        _emit(out)
        return "step"

    return None


# ── stream mode ──────────────────────────────────────────────

def stream(log_path: str, timeout: int = 3600) -> None:
    """Persistent streaming mode. Reads from current end of file forward."""
    seen_steps: set[int] = set()
    start = time.time()
    inode: int | None = None

    # Wait for file to exist
    while not os.path.exists(log_path):
        if time.time() - start > 120:
            _emit(f"TIMEOUT: log file {log_path} did not appear within 120s")
            return
        time.sleep(1)

    # Read from current end of file
    try:
        st = os.stat(log_path)
        inode = st.st_ino
        pos = st.st_size
    except OSError:
        pos = 0

    last_data_time = time.time()

    while True:
        if time.time() - start > timeout:
            _emit("TIMEOUT: monitor stream exceeded time limit")
            return

        try:
            st = os.stat(log_path)
            current_inode = st.st_ino
            current_size = st.st_size
        except OSError:
            time.sleep(1)
            continue

        # File was recreated (inode changed) or truncated → reset
        if current_inode != inode or current_size < pos:
            inode = current_inode
            pos = 0
            seen_steps.clear()

        if current_size > pos:
            last_data_time = time.time()
            try:
                with open(log_path, "r", errors="replace") as f:
                    f.seek(pos)
                    data = f.read()
                    pos = f.tell()
            except (OSError, ValueError):
                time.sleep(1)
                continue

            for line in data.splitlines():
                result = _handle_line(line, seen_steps)
                if result == "report":
                    return
                elif result == "error":
                    return
        else:
            # No new data — check for stall
            if time.time() - last_data_time > 600:
                _emit("WARN: no log output for 10 minutes, training may be stalled")

        time.sleep(1)


# ── oneshot mode ─────────────────────────────────────────────

def oneshot(log_path: str, last_n: int = 1) -> None:
    """One-shot mode: print the last N MONITOR_STEP lines. For CronCreate fallback."""
    if not os.path.exists(log_path):
        _emit("ONESHOT: log file not found")
        return

    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        _emit("ONESHOT: cannot read log file")
        return

    # Check for terminal events
    for line in lines:
        if "Training Report" in line:
            _emit("DETECTED: Training Report")
            return
        if any(kw in line for kw in ("Traceback", "Error", "OOM", "out of memory")):
            if "[MONITOR_STEP]" not in line and "[TRL_STEP]" not in line:
                _emit(f"DETECTED: {line.strip()[:200]}")
                return

    # Extract last N MONITOR_STEP lines
    monitor_lines = [l for l in lines if "[MONITOR_STEP]" in l]
    if monitor_lines:
        for l in monitor_lines[-last_n:]:
            out = l.split("[MONITOR_STEP]", 1)[1].strip()
            _emit(out)
    else:
        # Print last non-boilerplate line as progress indicator
        for l in reversed(lines):
            stripped = l.strip()
            if stripped and not stripped.startswith("[transformers]"):
                _emit(f"PROGRESS: {stripped[:200]}")
                break
        else:
            _emit("PROGRESS: log exists but no recognizable content yet")


# ── poll mode (incremental, stateful — for ScheduleWakeup loops) ──

def _load_poll_state(state_file: str) -> dict:
    """Load poll state from JSON file. Returns defaults if missing/corrupt."""
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            # Validate expected keys
            if "last_reported_step" in state:
                return state
        except (json.JSONDecodeError, IOError, KeyError):
            pass
    return {"last_reported_step": 0}


def _save_poll_state(state_file: str, last_reported_step: int) -> None:
    """Persist poll state so the next invocation knows where to resume."""
    try:
        state_dir = os.path.dirname(state_file)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump({
                "last_reported_step": last_reported_step,
                "last_check_time": time.time(),
            }, f)
    except (OSError, IOError):
        pass


def _extract_new_monitor_steps(
    lines: list[str], last_reported_step: int
) -> tuple[list[str], int]:
    """Extract [MONITOR_STEP] lines with step > last_reported_step.

    When the same step appears twice (partial from rollout_done,
    then complete from update_training_metrics), keeps the LAST one
    which has the most complete data (non-— loss/grad/ent).

    Returns (output_lines, max_step_seen).
    """
    # Use dict to dedup by step number — later entries overwrite earlier ones
    # (the complete emission from update_training_metrics comes after the
    # partial emission from log_rollout_done)
    step_map: dict[int, str] = {}
    max_step = last_reported_step

    for line in lines:
        if "[MONITOR_STEP]" not in line:
            continue
        sn = _extract_step_num(line)
        if sn is None:
            continue
        if sn <= last_reported_step:
            continue
        max_step = max(max_step, sn)
        out = line.split("[MONITOR_STEP]", 1)[1].strip()
        step_map[sn] = out  # overwrite — keep last (most complete) per step

    # Return in step order
    new = [step_map[s] for s in sorted(step_map)]
    return new, max_step


def poll(log_path: str, state_dir: str | None = None) -> None:
    """Poll mode: output only NEW [MONITOR_STEP] lines since last poll.

    Designed for ScheduleWakeup-based active polling loops.
    Uses a state file (<log_dir>/.monitor_poll_state.json) to track
    the last reported step across invocations.

    Output conventions (parseable by the calling skill):
        DETECTED: Training Report — training complete, stop polling
        DETECTED: <error>       — fatal error, stop polling
        <fixed-schema step>     — new step line (same format as MONITOR_STEP)
        PROGRESS: <msg>         — heartbeat, no new steps yet
        POLL: <msg>             — status/info message
    """
    if state_dir is None:
        state_dir = os.path.dirname(os.path.abspath(log_path))
    state_file = os.path.join(state_dir, ".monitor_poll_state.json")

    state = _load_poll_state(state_file)
    last_reported_step = state.get("last_reported_step", 0)

    # ── File existence check ──
    if not os.path.exists(log_path):
        _emit(f"POLL: log file not found at {log_path}, training may not have started yet")
        _save_poll_state(state_file, last_reported_step)
        return

    # ── Read full log ──
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        _emit("POLL: cannot read log file")
        return

    if not lines:
        _emit("POLL: log file is empty, waiting for first output...")
        _save_poll_state(state_file, last_reported_step)
        return

    full_text = "".join(lines)

    # ── Terminal event: Training Report ──
    if "Training Report" in full_text:
        # Output any steps we might have missed
        new_steps, max_step = _extract_new_monitor_steps(lines, last_reported_step)
        for out in new_steps:
            _emit(out)
        _emit("DETECTED: Training Report")
        # Clean up state file so next run starts fresh
        try:
            os.remove(state_file)
        except OSError:
            pass
        return

    # ── Fatal error detection ──
    fatal_keywords = ["Traceback", "CUDA out of memory", "Killed", "SIGTERM",
                      "RuntimeError", "HCCL timeout", "NPU error"]
    for line in lines:
        if any(kw in line for kw in fatal_keywords):
            if "[MONITOR_STEP]" not in line and "[TRL_STEP]" not in line:
                # Output any unreported steps first
                new_steps, max_step = _extract_new_monitor_steps(lines, last_reported_step)
                for out in new_steps:
                    _emit(out)
                _emit(f"DETECTED: {line.strip()[:300]}")
                _save_poll_state(state_file, max(max_step, last_reported_step))
                return

    # ── Extract and output new steps ──
    new_steps, max_step = _extract_new_monitor_steps(lines, last_reported_step)

    if new_steps:
        for out in new_steps:
            _emit(out)
        _save_poll_state(state_file, max_step)
    else:
        # No new steps — emit a progress heartbeat so the user knows we're alive
        heartbeat = _build_heartbeat(lines, last_reported_step)
        _emit(heartbeat)
        _save_poll_state(state_file, last_reported_step)


def _build_heartbeat(lines: list[str], last_reported_step: int) -> str:
    """Build a progress heartbeat line when there are no new steps."""
    # Find the last meaningful log line
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[transformers]"):
            continue
        # Found something — use as heartbeat
        truncated = stripped[:200]
        if last_reported_step > 0:
            return f"PROGRESS(step>{last_reported_step}): {truncated}"
        return f"PROGRESS: {truncated}"

    return "PROGRESS: waiting for training output..."


# ── CLI ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robust training log monitor")
    sub = parser.add_subparsers(dest="mode")

    p_stream = sub.add_parser("stream")
    p_stream.add_argument("log_path")
    p_stream.add_argument("--timeout", type=int, default=3600)

    p_shot = sub.add_parser("oneshot")
    p_shot.add_argument("log_path")
    p_shot.add_argument("--last", type=int, default=1)

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("log_path")
    p_poll.add_argument("--state-dir", default=None,
                        help="Directory for .monitor_poll_state.json (default: log dir)")

    args = parser.parse_args()
    if args.mode == "stream":
        stream(args.log_path, timeout=args.timeout)
    elif args.mode == "oneshot":
        oneshot(args.log_path, last_n=args.last)
    elif args.mode == "poll":
        poll(args.log_path, state_dir=args.state_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
