from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

WATCHDOG_TELEMETRY_SCHEMA_VERSION = 2


def _resolved_heartbeat_path(
    heartbeat_path: str | Path | None,
    cwd: str | Path | None,
) -> Path | None:
    if heartbeat_path is None:
        return None
    path = Path(heartbeat_path)
    if not path.is_absolute() and cwd is not None:
        path = Path(cwd) / path
    return path.resolve(strict=False)


def _terminate_process(process: subprocess.Popen, grace_s: float) -> tuple[str, int | None]:
    """Terminate first, then kill only if the grace interval expires."""
    if process.poll() is not None:
        return "already_exited", process.returncode
    try:
        process.terminate()
    except ProcessLookupError:
        return "already_exited", process.poll()
    try:
        return "terminate", process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return "terminate", process.poll()
        try:
            return "kill", process.wait(timeout=max(grace_s, 0.1))
        except subprocess.TimeoutExpired:
            return "kill_unconfirmed", process.poll()


def _append_record(handle, record: dict) -> None:
    if handle is None:
        return
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def run_watchdog(
    command: Sequence[str],
    *,
    restarts: int = 3,
    sleep_s: float = 2.0,
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
    timeout_s: float | None = None,
    terminate_grace_s: float = 2.0,
    heartbeat_path: str | Path | None = None,
    heartbeat_timeout_s: float | None = None,
    poll_interval_s: float = 0.05,
) -> int:
    """Run a command with bounded restart, timeout, and optional heartbeat checks."""
    if not command:
        raise ValueError("command must not be empty")
    if restarts < 0:
        raise ValueError("restarts must be non-negative")
    if sleep_s < 0:
        raise ValueError("sleep_s must be non-negative")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive when provided")
    if terminate_grace_s < 0:
        raise ValueError("terminate_grace_s must be non-negative")
    if heartbeat_timeout_s is not None and heartbeat_timeout_s <= 0:
        raise ValueError("heartbeat_timeout_s must be positive when provided")
    if heartbeat_timeout_s is not None and heartbeat_path is None:
        raise ValueError("heartbeat_timeout_s requires heartbeat_path")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    resolved_heartbeat = _resolved_heartbeat_path(heartbeat_path, cwd)
    log_handle = None
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = path.open("a", encoding="utf-8")

    attempts = restarts + 1
    last_watchdog_code = 1
    try:
        for attempt in range(1, attempts + 1):
            started_monotonic = time.monotonic()
            started_unix_s = time.time()
            outcome = "spawn_error"
            child_returncode: int | None = None
            watchdog_code = 127
            termination_action = "none"
            heartbeat_updates = 0
            last_heartbeat_marker: int | None = None
            last_heartbeat_seen = started_monotonic
            spawn_error: str | None = None

            if resolved_heartbeat is not None and resolved_heartbeat.exists():
                try:
                    last_heartbeat_marker = resolved_heartbeat.stat().st_mtime_ns
                except OSError:
                    last_heartbeat_marker = None

            try:
                process = subprocess.Popen(list(command), cwd=cwd)
            except OSError as exc:
                spawn_error = f"{type(exc).__name__}: {exc}"
                outcome = "spawn_error"
                watchdog_code = 127
            else:
                while True:
                    child_returncode = process.poll()
                    if child_returncode is not None:
                        if child_returncode == 0:
                            outcome = "success"
                            watchdog_code = 0
                        else:
                            outcome = "nonzero_exit"
                            watchdog_code = int(child_returncode)
                        break

                    now = time.monotonic()
                    elapsed = now - started_monotonic
                    timeout_reason: str | None = None
                    if timeout_s is not None and elapsed >= timeout_s:
                        timeout_reason = "timeout"

                    if timeout_reason is None and resolved_heartbeat is not None and heartbeat_timeout_s is not None:
                        try:
                            marker = resolved_heartbeat.stat().st_mtime_ns
                        except OSError:
                            marker = None
                        if marker is not None and marker != last_heartbeat_marker:
                            last_heartbeat_marker = marker
                            last_heartbeat_seen = now
                            heartbeat_updates += 1
                        if now - last_heartbeat_seen >= heartbeat_timeout_s:
                            timeout_reason = "heartbeat_timeout"

                    if timeout_reason is not None:
                        termination_action, child_returncode = _terminate_process(
                            process,
                            terminate_grace_s,
                        )
                        outcome = timeout_reason
                        watchdog_code = 124 if timeout_reason == "timeout" else 125
                        break
                    time.sleep(poll_interval_s)

            ended_unix_s = time.time()
            duration_s = time.monotonic() - started_monotonic
            restart_scheduled = outcome != "success" and attempt < attempts
            record = {
                "schema_version": WATCHDOG_TELEMETRY_SCHEMA_VERSION,
                "record_kind": "watchdog_attempt",
                "attempt": attempt,
                "max_attempts": attempts,
                "command": list(command),
                "cwd": None if cwd is None else str(Path(cwd)),
                "started_unix_s": started_unix_s,
                "ended_unix_s": ended_unix_s,
                "duration_s": duration_s,
                "outcome": outcome,
                "child_returncode": child_returncode,
                "watchdog_returncode": watchdog_code,
                "timeout_s": timeout_s,
                "terminate_grace_s": terminate_grace_s,
                "termination_action": termination_action,
                "heartbeat_path": None if resolved_heartbeat is None else str(resolved_heartbeat),
                "heartbeat_timeout_s": heartbeat_timeout_s,
                "heartbeat_updates": heartbeat_updates,
                "restart_scheduled": restart_scheduled,
                "restart_reason": outcome if restart_scheduled else None,
                "spawn_error": spawn_error,
            }
            _append_record(log_handle, record)
            last_watchdog_code = watchdog_code
            if outcome == "success":
                return 0
            if restart_scheduled and sleep_s:
                time.sleep(sleep_s)
        return int(last_watchdog_code or 1)
    finally:
        if log_handle is not None:
            log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restart a failing or hung inference command a bounded number of times."
    )
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--sleep-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--terminate-grace-s", type=float, default=2.0)
    parser.add_argument("--heartbeat-file", type=Path, default=None)
    parser.add_argument("--heartbeat-timeout-s", type=float, default=None)
    parser.add_argument("--poll-interval-s", type=float, default=0.05)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, normally after --")
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    return run_watchdog(
        command,
        restarts=args.restarts,
        sleep_s=args.sleep_s,
        cwd=args.cwd,
        log_path=args.log,
        timeout_s=args.timeout_s,
        terminate_grace_s=args.terminate_grace_s,
        heartbeat_path=args.heartbeat_file,
        heartbeat_timeout_s=args.heartbeat_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )


if __name__ == "__main__":
    sys.exit(main())
