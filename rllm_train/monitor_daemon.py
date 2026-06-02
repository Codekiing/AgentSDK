"""
Autonomous training monitor daemon — runs independently of Claude Code session.

Launched as nohup background process alongside training. Survives session termination.
Writes monitoring status and alerts to files that can be checked anytime.

Usage:
    nohup python3 rllm_train/monitor_daemon.py <output_dir> \
        --auto-stop --pid-file /tmp/monitor_r10.pid &

Check status:
    cat <output_dir>/monitor_status.json
    cat <output_dir>/monitor_alerts.jsonl (if any)
    tail -20 <output_dir>/monitor_log.txt
"""

import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ──
_POLL_INTERVAL = 15  # seconds between metric checks
_LOG_TAIL_BYTES = 8192  # bytes to read from end of training log

_COMPLETE_RE = re.compile(r"Training (?:completed|finished|Report)", re.I)
_ERROR_RE = re.compile(
    r"Traceback|out of memory|Killed|SIGTERM|SIGKILL|"
    r"RuntimeError|RayActorError|Actor died|"
    r"CUDA error|NCCL error|HCCL|NaN",
    re.I,
)
_BENIGN_RE = re.compile(
    r"WARNING|UserWarning|FutureWarning|Deprecated|OMP_NUM_THREADS|"
    r"tokenizers|No CUDA runtime",
    re.I,
)

# Metrics whose change signals a problem
_DANGER_CHECKS = {
    "actor/pg_loss": {"max": 10.0, "check_nan": True},
    "actor/grad_norm": {"max": 100.0, "check_nan": True},
    "actor/entropy": {"min": 0.01},
}


class MonitorDaemon:
    def __init__(
        self,
        output_dir: str,
        auto_stop: bool = False,
        pid_file: str | None = None,
        poll_interval: int = _POLL_INTERVAL,
    ):
        self.output_dir = Path(output_dir)
        self.auto_stop = auto_stop
        self.pid_file = pid_file
        self.poll_interval = poll_interval

        self.metrics_path = self.output_dir / "verl_metrics.jsonl"
        self.log_path = self.output_dir / "training_log.txt"
        self.status_path = self.output_dir / "monitor_status.json"
        self.alerts_path = self.output_dir / "monitor_alerts.jsonl"
        self.daemon_log = self.output_dir / "monitor_log.txt"

        self._last_metrics_pos = 0
        self._last_step = 0
        self._start_time = time.time()
        self._alerts = []
        self._step_history = []  # [(step, score, entropy, grad_norm), ...]

    # ── public API ──

    def run(self) -> dict:
        """Main loop. Returns final status dict."""
        self._write_pid()
        self._log("Monitor daemon started")
        self._log(f"  output_dir: {self.output_dir}")
        self._log(f"  auto_stop: {self.auto_stop}")
        self._log(f"  poll_interval: {self.poll_interval}s")

        try:
            self._wait_for_metrics_file(timeout=600)
            while True:
                status = self._poll()
                self._write_status(status)

                if status["state"] in ("complete", "error"):
                    self._log(f"Monitor exiting: state={status['state']}")
                    self._cleanup_pid()
                    return status

                if self.auto_stop and status.get("should_stop"):
                    self._kill_training(status["stop_reason"])
                    status["state"] = "stopped_by_monitor"
                    self._write_status(status)
                    self._cleanup_pid()
                    return status

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self._log("Monitor interrupted")
            self._cleanup_pid()
            return {"state": "interrupted"}
        except Exception as e:
            self._log(f"Monitor crashed: {e}")
            self._cleanup_pid()
            return {"state": "crashed", "error": str(e)}

    # ── internals ──

    def _wait_for_metrics_file(self, timeout: int) -> None:
        deadline = time.time() + timeout
        while not self.metrics_path.exists():
            if time.time() > deadline:
                raise TimeoutError(f"Metrics file did not appear: {self.metrics_path}")
            # Check for early errors in log
            err = self._scan_log_errors()
            if err and err != "complete":
                self._log(f"Early error detected: {err}")
            time.sleep(5)
        self._log(f"Metrics file found after {time.time() - self._start_time:.0f}s")

    def _poll(self) -> dict:
        """Read new metrics, check for issues, return status dict."""
        status = {
            "state": "running",
            "last_step": self._last_step,
            "elapsed_s": time.time() - self._start_time,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "should_stop": False,
            "stop_reason": None,
        }

        # Read new metrics
        if self.metrics_path.exists():
            try:
                st = os.stat(str(self.metrics_path))
                if st.st_size > self._last_metrics_pos:
                    with open(self.metrics_path, "r") as f:
                        f.seek(self._last_metrics_pos)
                        new_data = f.read()
                        self._last_metrics_pos = f.tell()

                    for line in new_data.splitlines():
                        if not line.strip():
                            continue
                        try:
                            step_data = json.loads(line)
                            self._process_step(step_data, status)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass

        # Check log for completion / errors
        err = self._scan_log_errors()
        if err == "complete":
            status["state"] = "complete"
            status["final_step"] = self._last_step
        elif err:
            self._add_alert("error", err)
            status["state"] = "error"
            status["error"] = err

        # Build summary
        if self._step_history:
            recent = self._step_history[-10:]
            scores = [s[1] for s in recent if s[1] is not None]
            ents = [s[2] for s in recent if s[2] is not None]
            status["recent_avg_score"] = sum(scores) / len(scores) if scores else None
            status["recent_avg_entropy"] = sum(ents) / len(ents) if ents else None
            status["total_steps_seen"] = len(self._step_history)

        return status

    def _process_step(self, step_data: dict, status: dict) -> None:
        """Process a single metrics step. Check for danger conditions."""
        step = step_data.get("step", 0)
        if step <= self._last_step:
            return
        self._last_step = step
        data = step_data.get("data", {})

        score = data.get("critic/score/mean")
        entropy = data.get("actor/entropy")
        grad_norm = data.get("actor/grad_norm")
        pg_loss = data.get("actor/pg_loss")

        self._step_history.append((step, score, entropy, grad_norm))

        # Danger checks
        for metric, checks in _DANGER_CHECKS.items():
            val = data.get(metric)
            if val is None:
                continue
            if checks.get("check_nan") and (isinstance(val, float) and (val != val)):  # NaN check
                self._add_alert("danger", f"Step {step}: {metric}=NaN — training corrupt")
                if self.auto_stop:
                    status["should_stop"] = True
                    status["stop_reason"] = f"{metric}=NaN at step {step}"
            if "max" in checks and isinstance(val, (int, float)) and val == val and val > checks["max"]:
                self._add_alert("warning", f"Step {step}: {metric}={val:.2f} > threshold {checks['max']}")
            if "min" in checks and isinstance(val, (int, float)) and val == val and val < checks["min"]:
                self._add_alert("warning", f"Step {step}: {metric}={val:.4f} < threshold {checks['min']}")

        # Entropy collapse check
        if entropy is not None and entropy < 0.05 and len(self._step_history) > 5:
            recent_ents = [s[2] for s in self._step_history[-5:] if s[2] is not None]
            if recent_ents and all(e < 0.05 for e in recent_ents):
                self._add_alert("warning", f"Step {step}: Entropy collapsed ({entropy:.4f}) for 5+ steps")

    def _scan_log_errors(self) -> str | None:
        """Scan training log tail for completion/errors. Returns 'complete', error_msg, or None."""
        if not self.log_path.exists():
            return None
        try:
            size = os.path.getsize(str(self.log_path))
            with open(self.log_path, "r", errors="replace") as f:
                f.seek(max(0, size - _LOG_TAIL_BYTES))
                tail = f.read()
        except OSError:
            return None

        for line in tail.splitlines():
            if _COMPLETE_RE.search(line):
                return "complete"
            if _ERROR_RE.search(line) and not _BENIGN_RE.search(line):
                return line.strip()[:200]
        return None

    def _add_alert(self, level: str, message: str) -> None:
        """Log an alert to alerts file."""
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        self._alerts.append(alert)
        self._log(f"[{level.upper()}] {message}")
        try:
            with open(self.alerts_path, "a") as f:
                f.write(json.dumps(alert) + "\n")
        except OSError:
            pass

    def _write_status(self, status: dict) -> None:
        """Write current status to status file."""
        try:
            # Add summary
            status["alerts_count"] = len(self._alerts)
            if self._step_history:
                status["score_trend"] = self._compute_trend()
            with open(self.status_path, "w") as f:
                json.dump(status, f, indent=2, default=str)
        except OSError:
            pass

    def _compute_trend(self) -> str:
        """Compute score trend from step history."""
        if len(self._step_history) < 5:
            return "insufficient_data"
        first_5 = [s[1] for s in self._step_history[:5] if s[1] is not None]
        last_5 = [s[1] for s in self._step_history[-5:] if s[1] is not None]
        if not first_5 or not last_5:
            return "insufficient_data"
        diff = sum(last_5) / len(last_5) - sum(first_5) / len(first_5)
        if diff > 0.05:
            return "rising"
        elif diff < -0.05:
            return "falling"
        return "stable"

    def _kill_training(self, reason: str) -> None:
        """Kill the training process."""
        self._log(f"Auto-stopping training: {reason}")
        try:
            # Find and kill main_ppo process
            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", "main_ppo"],
                capture_output=True, text=True, timeout=5
            )
            pids = result.stdout.strip().split()
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    self._log(f"Sent SIGTERM to pid {pid}")
                except (OSError, ValueError):
                    pass
            # Give processes time to shutdown gracefully
            time.sleep(5)
            # Force kill remaining
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (OSError, ValueError):
                    pass
        except Exception as e:
            self._log(f"Failed to kill training: {e}")

    def _write_pid(self) -> None:
        if self.pid_file:
            try:
                with open(self.pid_file, "w") as f:
                    f.write(str(os.getpid()))
            except OSError:
                pass

    def _cleanup_pid(self) -> None:
        if self.pid_file and os.path.exists(self.pid_file):
            try:
                os.remove(self.pid_file)
            except OSError:
                pass

    def _log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            with open(self.daemon_log, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ── CLI ──

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Autonomous training monitor daemon (runs independently of Claude Code)"
    )
    parser.add_argument("output_dir", help="Training output directory")
    parser.add_argument("--auto-stop", action="store_true",
                        help="Auto-kill training on NaN/loss explosion")
    parser.add_argument("--pid-file", default=None,
                        help="Write PID to file for external management")
    parser.add_argument("--poll-interval", type=int, default=_POLL_INTERVAL,
                        help=f"Seconds between metric checks (default: {_POLL_INTERVAL})")
    args = parser.parse_args()

    daemon = MonitorDaemon(
        output_dir=args.output_dir,
        auto_stop=args.auto_stop,
        pid_file=args.pid_file,
        poll_interval=args.poll_interval,
    )
    final_status = daemon.run()
    print(json.dumps(final_status, indent=2, default=str))


if __name__ == "__main__":
    main()
