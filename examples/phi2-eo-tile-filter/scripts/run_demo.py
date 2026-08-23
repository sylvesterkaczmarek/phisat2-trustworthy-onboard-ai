from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete trustworthy tile-filter demonstration.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--bands", type=int, default=3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parents[1]
    repo_root = example_root.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from assurance.model_store import resolve_bundle

    output_root = args.output_root.resolve()
    tiles = output_root / "tiles"
    runs = output_root / "runs"
    models = output_root / "models"
    logs = output_root / "logs"
    reports = output_root / "reports"
    downlink = output_root / "downlink"
    calibration = output_root / "calibration.json"
    candidate_bundle = models / "candidate_bundle"
    bundle_store = models / "bundles"
    deployment_state = models / "deployment_state.json"
    python = sys.executable

    run(
        [
            python,
            "-m",
            "phi2_tile_filter.synth",
            "--out",
            str(tiles),
            "--n",
            str(args.n),
            "--bands",
            str(args.bands),
            "--size",
            str(args.size),
            "--seed",
            str(args.seed),
            "--overwrite",
        ],
        cwd=example_root,
    )
    run(
        [
            python,
            "-m",
            "phi2_tile_filter.train",
            "--data",
            str(tiles),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--out",
            str(runs / "tinycnn.pt"),
        ],
        cwd=example_root,
    )
    run(
        [
            python,
            "-m",
            "phi2_tile_filter.export_onnx",
            "--weights",
            str(runs / "tinycnn.pt"),
            "--out",
            str(models / "tinycnn_fp32.onnx"),
        ],
        cwd=example_root,
    )
    run(
        [
            python,
            "-m",
            "phi2_tile_filter.quantize_ptq",
            "--onnx",
            str(models / "tinycnn_fp32.onnx"),
            "--calib",
            str(tiles / "calib"),
            "--out",
            str(models / "tinycnn_int8.onnx"),
        ],
        cwd=example_root,
    )
    run(
        [
            python,
            "-m",
            "phi2_tile_filter.validate_models",
            "--fp32",
            str(models / "tinycnn_fp32.onnx"),
            "--int8",
            str(models / "tinycnn_int8.onnx"),
            "--data",
            str(tiles / "test"),
            "--max-accuracy-drop",
            "0.05",
            "--min-argmax-agreement",
            "0.95",
            "--out",
            str(reports / "model_validation.json"),
        ],
        cwd=example_root,
    )
    run(
        [
            python,
            "-m",
            "phi2_tile_filter.calibrate_threshold",
            "--onnx",
            str(models / "tinycnn_int8.onnx"),
            "--data",
            str(tiles / "calib"),
            "--target-recall",
            "0.95",
            "--min-confidence",
            "0.60",
            "--out",
            str(calibration),
        ],
        cwd=example_root,
    )
    run(
        [
            python,
            str(repo_root / "assurance" / "model_store.py"),
            "build",
            "--model",
            str(models / "tinycnn_int8.onnx"),
            "--policy",
            str(calibration),
            "--validation",
            str(reports / "model_validation.json"),
            "--out",
            str(candidate_bundle),
        ],
        cwd=repo_root,
    )
    run(
        [
            python,
            str(repo_root / "assurance" / "model_store.py"),
            "promote",
            "--candidate-bundle",
            str(candidate_bundle),
            "--store",
            str(bundle_store),
            "--state",
            str(deployment_state),
        ],
        cwd=repo_root,
    )

    active = resolve_bundle(bundle_store, deployment_state, slot="active")
    active_model = Path(active["model"])
    active_policy = Path(active["policy"])

    run(
        [
            python,
            str(repo_root / "assurance" / "telemetry_log.py"),
            "--onnx",
            str(active_model),
            "--data",
            str(tiles / "test"),
            "--policy",
            str(active_policy),
            "--out",
            str(logs / "test.jsonl"),
        ],
        cwd=repo_root,
    )
    run(
        [
            python,
            "-m",
            "phi2_tile_filter.bandwidth_filter",
            "--onnx",
            str(active_model),
            "--data",
            str(tiles / "test"),
            "--policy",
            str(active_policy),
            "--downlink-out",
            str(downlink),
            "--log",
            str(logs / "downlink.jsonl"),
        ],
        cwd=example_root,
    )
    run(
        [
            python,
            str(repo_root / "assurance" / "summarize.py"),
            "--test-log",
            str(logs / "test.jsonl"),
            "--downlink-log",
            str(logs / "downlink.jsonl"),
            "--calib",
            str(active_policy),
            "--out-dir",
            str(reports),
        ],
        cwd=repo_root,
    )

    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    metrics["active_bundle_id"] = active["bundle_id"]
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
