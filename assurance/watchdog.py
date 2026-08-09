from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def run_watchdog(
    command: Sequence[str],
    *,
    restarts: int = 3,
    sleep_s: float = 2.0,
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
) -> int:
    """Run a command and retry failures without invoking a shell."""
    if not command:
        raise ValueError("command must not be empty")
    if restarts < 0:
        raise ValueError("restarts must be non-negative")
    if sleep_s < 0:
        raise ValueError("sleep_s must be non-negative")

    log_handle = None
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = path.open("a", encoding="utf-8")

    try:
        attempts = restarts + 1
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            completed = subprocess.run(list(command), cwd=cwd, check=False)
            duration_s = time.perf_counter() - started
            record = {
                "attempt": attempt,
                "returncode": int(completed.returncode),
                "duration_s": duration_s,
                "command": list(command),
            }
            if log_handle is not None:
                log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                log_handle.flush()
            if completed.returncode == 0:
                return 0
            if attempt < attempts and sleep_s:
                time.sleep(sleep_s)
        return int(completed.returncode or 1)
    finally:
        if log_handle is not None:
            log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Restart a failing inference command a bounded number of times.")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--sleep-s", type=float, default=2.0)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, normally after --")
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    return run_watchdog(command, restarts=args.restarts, sleep_s=args.sleep_s, cwd=args.cwd, log_path=args.log)


if __name__ == "__main__":
    sys.exit(main())
