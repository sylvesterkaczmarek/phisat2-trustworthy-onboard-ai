from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .policy import DecisionPolicy
from .runtime import OnnxRunner
from .utils import discover_tile_files


def load_policy(path: str | Path, runner: OnnxRunner) -> DecisionPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (2, 3):
        raise ValueError("unsupported calibration policy schema")
    if payload.get("model_sha256") != runner.model_sha256:
        raise ValueError("calibration policy belongs to a different model")
    if int(payload.get("bands", -1)) != runner.spec.bands or int(payload.get("size", -1)) != runner.spec.size:
        raise ValueError("calibration policy input shape does not match model")
    if payload.get("schema_version") == 3:
        acceptance = payload.get("calibration_acceptance")
        if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
            raise ValueError("calibration policy is not marked accepted")
    return DecisionPolicy(
        event_threshold=float(payload["event_threshold"]),
        min_confidence=float(payload["min_confidence"]),
        temperature=float(payload["temperature"]),
    )


def filter_tiles(
    model_path: str | Path,
    data_root: str | Path,
    policy_path: str | Path,
    *,
    downlink_root: str | Path,
    log_path: str | Path,
) -> dict:
    runner = OnnxRunner(model_path)
    policy = load_policy(policy_path, runner)
    data_root = Path(data_root)
    files = discover_tile_files(data_root)
    if not files:
        raise ValueError(f"no supported tiles found under {data_root}")

    downlink_root = Path(downlink_root)
    if downlink_root.exists():
        shutil.rmtree(downlink_root)
    downlink_root.mkdir(parents=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    kept_bytes = 0
    kept_count = 0
    fallback_count = 0
    failure_count = 0
    with log_path.open("w", encoding="utf-8") as handle:
        for path in files:
            record = runner.evaluate_file(path, policy)
            record["file"] = str(path.relative_to(data_root))
            total_bytes += int(record["size_bytes"])
            if record["decision"].endswith("fallback"):
                fallback_count += 1
            if not record["inference_ok"]:
                failure_count += 1
            if record["kept"]:
                relative = path.relative_to(data_root)
                destination = downlink_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                kept_count += 1
                kept_bytes += int(record["size_bytes"])
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    saved_fraction = 1.0 - kept_bytes / total_bytes if total_bytes else 0.0
    summary = {
        "schema_version": 2,
        "model_sha256": runner.model_sha256,
        "tiles_total": len(files),
        "tiles_kept": kept_count,
        "fallback_tiles": fallback_count,
        "inference_failures": failure_count,
        "bytes_total": total_bytes,
        "bytes_kept": kept_bytes,
        "bandwidth_saved_fraction": saved_fraction,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--downlink-out", default="downlink")
    parser.add_argument("--log", default="logs/downlink.jsonl")
    args = parser.parse_args()
    filter_tiles(
        args.onnx,
        args.data,
        args.policy,
        downlink_root=args.downlink_out,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
