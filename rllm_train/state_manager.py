"""
Persistent training orchestration state.

Writes to training_state.json in rllm_train/output/. Used by rllm-train to recover
state across conversation context compaction and session restarts.

Phase-to-file mapping:
    Phase 1 (clarify): write after clarify output extracted
    Phase 2 (config): write after config.json generated
    Phase 3 (run): write after training process launched
    Phase 4 (monitor): write after monitor completes / circuit break
    Phase 5 (analyze): write after analysis.json generated
    Phase 6 (report): write after final report (mark completed)
"""

import json
import os
import time
from pathlib import Path

_STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rllm_train",
    "output",
)
_STATE_PATH = os.path.join(_STATE_DIR, "training_state.json")


def _ensure_dir():
    os.makedirs(_STATE_DIR, exist_ok=True)


def read() -> dict:
    """Read current training state. Returns empty dict if no state exists."""
    _ensure_dir()
    try:
        with open(_STATE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(state: dict) -> str:
    """Atomically write training state. Returns path written."""
    _ensure_dir()
    state["_updated_at"] = time.time()
    tmp_path = _STATE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, _STATE_PATH)
    return _STATE_PATH


def init(task_id: str, skill_package_id: str, **kwargs) -> str:
    """Create a fresh training_state.json for a new training task."""
    state = {
        "task_id": task_id,
        "skill_package_id": skill_package_id,
        "round": kwargs.get("round", 1),
        "current_phase": kwargs.get("current_phase", "clarify"),
        "current_run_id": kwargs.get("current_run_id", ""),
        "circuit_break_count": 0,
        "history": [],
        "target": kwargs.get("target", {}),
        "mode": kwargs.get("mode", "auto"),
        "completed": False,
        "_created_at": time.time(),
    }
    write(state)
    return _STATE_PATH


def update_phase(phase: str, run_id: str = "", **kwargs) -> str:
    """Update current_phase and optionally run_id. Merges kwargs."""
    state = read()
    state["current_phase"] = phase
    if run_id:
        state["current_run_id"] = run_id
    state.update(kwargs)
    write(state)
    return _STATE_PATH


def record_round_result(run_id: str, reward_start: float, reward_end: float,
                        config_changes: list | None = None) -> str:
    """Append a completed round to history and increment round counter."""
    state = read()
    state["current_run_id"] = run_id
    entry = {
        "round": state.get("round", 1),
        "run_id": run_id,
        "reward_start": reward_start,
        "reward_end": reward_end,
        "config_changes": config_changes or [],
        "_recorded_at": time.time(),
    }
    state.setdefault("history", []).append(entry)
    state["round"] = state.get("round", 1) + 1
    state["current_phase"] = "analyze_complete"
    write(state)
    return _STATE_PATH


def increment_circuit_break() -> int:
    """Increment circuit_break_count, return new count."""
    state = read()
    count = state.get("circuit_break_count", 0) + 1
    state["circuit_break_count"] = count
    write(state)
    return count


def reset_circuit_break() -> str:
    """Reset circuit_break_count to 0."""
    state = read()
    state["circuit_break_count"] = 0
    write(state)
    return _STATE_PATH


def mark_completed() -> str:
    """Mark training as completed."""
    state = read()
    state["completed"] = True
    state["current_phase"] = "complete"
    state["_completed_at"] = time.time()
    write(state)
    return _STATE_PATH


def get_next_phase() -> dict | None:
    """Read state and determine what Phase to execute next.
    Returns dict with 'phase' and context, or None if nothing to do.
    """
    state = read()
    if not state:
        return None
    if state.get("completed"):
        return None
    phase = state.get("current_phase", "")
    return {
        "phase": phase,
        "run_id": state.get("current_run_id", ""),
        "round": state.get("round", 1),
        "circuit_break_count": state.get("circuit_break_count", 0),
        "task_id": state.get("task_id", ""),
        "skill_package_id": state.get("skill_package_id", ""),
    }
