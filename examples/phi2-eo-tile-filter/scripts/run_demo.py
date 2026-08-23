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
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calib-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--calibration-confidence-level", type=float, default=0.95)
    parser.add_argument("--min-calibration-recall-lower-bound", type=float, default=None)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.95)
    parser.add_argument("--max-event-recall-drop", type=float, default=0.05)
    parser.add_argument("--max-event-fnr-increase", type=float, default=0.05)
    parser.add_argument("--max-pr-auc-drop", type=float, default=0.05)
    parser.add_argument("--min-policy-decision-agreement", type=float, default=0.95)
    parser.add_argument("--max-event-retention-recall-drop", type=float, default=0.05)
    parser.add_argument("--max-event-score-drift", type=float, default=0.05)
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
            "--train-fraction",
            str(args.train_fraction),
            "--calib-fraction",
            str(args.calib_fraction),
            "--validation-fraction",
            str(args.validation_fraction),
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

    calibration_command = [
        python,
        "-m",
        "phi2_tile_filter.calibrate_threshold",
        "--onnx",
        str(models / "tinycnn_int8.onnx"),
        "--data",
        str(tiles / "calib"),
        "--target-recall",
        str(args.target_recall),
        "--min-confidence",
        str(args.min_confidence),
        "--confidence-level",
        str(args.calibration_confidence_level),
        "--out",
        str(calibration),
    ]
    if args.min_calibration_recall_lower_bound is not None:
        calibration_command.extend(
            ["--min-event-recall-lower-bound", str(args.min_calibration_recall_lower_bound)]
        )
    run(calibration_command, cwd=example_root)

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
            str(tiles / "validation"),
            "--policy",
            str(calibration),
            "--max-accuracy-drop",
            str(args.max_accuracy_drop),
            "--min-argmax-agreement",
            str(args.min_argmax_agreement),
            "--max-event-recall-drop",
            str(args.max_event_recall_drop),
            "--max-event-fnr-increase",
            str(args.max_event_fnr_increase),
            "--max-pr-auc-drop",
            str(args.max_pr_auc_drop),
            "--min-policy-decision-agreement",
            str(args.min_policy_decision_agreement),
            "--max-event-retention-recall-drop",
            str(args.max_event_retention_recall_drop),
            "--max-event-score-drift",
            str(args.max_event_score_drift),
            "--out",
            str(reports / "model_validation.json"),
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
    active_bundle_id = str(active["bundle_id"])

    # The final test split is touched only after the candidate has passed calibration,
    # validation, bundle verification, and promotion.
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
            "--bundle-id",
            active_bundle_id,
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
            "--bundle-id",
            active_bundle_id,
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

    manifest = json.loads((tiles / "manifest.json").read_text(encoding="utf-8"))
    calibration_result = json.loads(active_policy.read_text(encoding="utf-8"))
    validation_result = json.loads((reports / "model_validation.json").read_text(encoding="utf-8"))
    final_test_metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    result = {
        "active_bundle_id": active_bundle_id,
        "split_counts": manifest["split_counts"],
        "split_roles": manifest["split_roles"],
        "calibration_statistics": calibration_result["calibration_statistics"],
        "calibration_acceptance": calibration_result["calibration_acceptance"],
        "validation_acceptance": {
            "accepted": validation_result["accepted"],
            "acceptance_checks": validation_result["acceptance_checks"],
            "classification_quantization_regression": validation_result["classification_metrics"][
                "quantization_regression"
            ],
            "policy_quantization_regression": validation_result["policy_metrics"][
                "quantization_regression"
            ],
            "score_drift_metrics": validation_result["score_drift_metrics"],
        },
        "final_test_metrics": final_test_metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
