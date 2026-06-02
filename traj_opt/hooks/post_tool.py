#!/usr/bin/env python3
"""PostToolUse hook — captures tool calls to events.jsonl.

Called by Claude Code after every tool use. Reads hook JSON from stdin,
converts via HooksAdapter, and appends to the session's events file.

Must complete within 1 second. Fails silently to avoid disrupting Claude Code.
"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from traj_opt.adapter.hooks_adapter import HooksAdapter, read_stdin
from traj_opt.store.writer import EventWriter
from traj_opt.config import TrajectoryConfig


def detect_layer(stdin_json: Dict[str, Any]) -> str:
    """Detect layer from hook data based on skill name prefix."""
    tool_data = stdin_json.get("tool", {})
    tool_name = tool_data.get("name")

    if tool_name == "Skill":
        tool_input = tool_data.get("input", {})
        skill_name = tool_input.get("skill", "")

        if skill_name.startswith("rllm-"):
            return "rllm"
        elif skill_name.startswith("traj-"):
            return "traj"
        elif skill_name.startswith("meta-"):
            return "meta"

    return "rllm"


def write_current_session(session_id: str, layer: str) -> None:
    """Write session_id to a well-known file so the main session can discover it."""
    try:
        marker_dir = Path(__file__).resolve().parents[2] / "traj_opt" / "output" / layer
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_file = marker_dir / ".current_session"
        marker_file.write_text(session_id)
    except Exception:
        pass


def main() -> None:
    try:
        stdin_json = read_stdin()
        if not stdin_json:
            return

        session_id = stdin_json.get("session_id", "unknown")
        layer = detect_layer(stdin_json)

        # Write current session marker so main session code can discover the session_id.
        # Claude Code does not expose session_id as an environment variable,
        # so this file is the only reliable way for Phase 6.5 to find it.
        if session_id and session_id != "unknown":
            write_current_session(session_id, layer)

        config = TrajectoryConfig(layer=layer)

        adapter = HooksAdapter()
        event = adapter.adapt("PostToolUse", stdin_json)
        event.layer = layer

        writer = EventWriter(config)
        writer.write_event(event)
    except Exception:
        pass


if __name__ == "__main__":
    main()
