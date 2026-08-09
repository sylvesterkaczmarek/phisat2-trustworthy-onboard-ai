from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def _write_manifest(path: Path, active: Path, previous: Path | None) -> None:
    payload = {
        "schema_version": 1,
        "active": str(active),
        "active_sha256": sha256_file(active),
        "previous": str(previous) if previous and previous.exists() else None,
        "previous_sha256": sha256_file(previous) if previous and previous.exists() else None,
        "updated_unix_s": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def promote(candidate: str | Path, active: str | Path, previous: str | Path, manifest: str | Path) -> dict:
    candidate = Path(candidate)
    active = Path(active)
    previous = Path(previous)
    manifest = Path(manifest)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    if active.exists():
        _atomic_copy(active, previous)
    _atomic_copy(candidate, active)
    _write_manifest(manifest, active, previous)
    return json.loads(manifest.read_text(encoding="utf-8"))


def rollback(active: str | Path, previous: str | Path, manifest: str | Path) -> dict:
    active = Path(active)
    previous = Path(previous)
    manifest = Path(manifest)
    if not previous.is_file():
        raise FileNotFoundError(f"no previous model at {previous}")

    active.parent.mkdir(parents=True, exist_ok=True)
    swap = active.with_name(active.name + ".swap")
    if active.exists():
        _atomic_copy(active, swap)
    _atomic_copy(previous, active)
    if swap.exists():
        _atomic_copy(swap, previous)
        swap.unlink()
    _write_manifest(manifest, active, previous)
    return json.loads(manifest.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomic known-good model promotion and rollback.")
    sub = parser.add_subparsers(dest="action", required=True)

    promote_parser = sub.add_parser("promote")
    promote_parser.add_argument("--candidate", type=Path, required=True)
    promote_parser.add_argument("--active", type=Path, required=True)
    promote_parser.add_argument("--previous", type=Path, required=True)
    promote_parser.add_argument("--manifest", type=Path, required=True)

    rollback_parser = sub.add_parser("rollback")
    rollback_parser.add_argument("--active", type=Path, required=True)
    rollback_parser.add_argument("--previous", type=Path, required=True)
    rollback_parser.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()
    if args.action == "promote":
        result = promote(args.candidate, args.active, args.previous, args.manifest)
    else:
        result = rollback(args.active, args.previous, args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
