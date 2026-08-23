from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SRC = REPO_ROOT / "examples" / "phi2-eo-tile-filter" / "src"
if str(EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SRC))

from phi2_tile_filter.telemetry import (  # noqa: E402
    DOWNLINK_RECORD_KIND,
    FINAL_TEST_RECORD_KIND,
    validate_telemetry_record,
)
from phi2_tile_filter.utils import sha256_file  # noqa: E402


IDENTITY_KEYS = (
    "deployment_bundle_id",
    "model_sha256",
    "policy_sha256",
    "input_schema_sha256",
    "input_schema_file_sha256",
    "preprocessing_sha256",
)


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number} of {path}") from exc
            records.append(record)
    return records


def _one_value(records: list[dict], key: str):
    values = {record.get(key) for record in records}
    if len(values) != 1:
        raise ValueError(f"records contain inconsistent {key} values")
    return next(iter(values))


def _records_by_file(records: list[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        file_name = str(record["file"])
        if file_name in indexed:
            raise ValueError(f"{label} telemetry contains duplicate file record: {file_name}")
        indexed[file_name] = record
    return indexed


def _calibration_metadata(calibration: dict) -> dict:
    if calibration.get("schema_version") in (3, 4, 5):
        stats = calibration.get("calibration_statistics", {})
        acceptance = calibration.get("calibration_acceptance", {})
        guard = calibration.get("input_quality_guard")
        return {
            "event_threshold": calibration.get("event_threshold"),
            "min_confidence": calibration.get("min_confidence"),
            "temperature": calibration.get("temperature"),
            "temperature_fitted": calibration.get("temperature_fitted"),
            "input_schema_sha256": calibration.get("input_schema_sha256"),
            "input_band_ids": calibration.get("input_band_ids"),
            "preprocessing_version": calibration.get("preprocessing_version"),
            "input_quality_guard": guard,
            "calibration_samples_total": stats.get("samples_total"),
            "calibration_event_samples": stats.get("event_samples"),
            "calibration_target_event_recall_for_threshold_selection": stats.get(
                "target_event_recall_for_threshold_selection"
            ),
            "calibration_empirical_event_recall": stats.get("empirical_event_recall"),
            "calibration_event_recall_lower_bound": stats.get("event_recall_lower_bound"),
            "calibration_event_recall_confidence_level": stats.get("event_recall_confidence_level"),
            "calibration_event_recall_bound_method": stats.get("event_recall_bound_method"),
            "calibration_required_min_event_recall_lower_bound": acceptance.get(
                "required_min_event_recall_lower_bound"
            ),
        }
    return {
        "event_threshold": calibration.get("event_threshold"),
        "min_confidence": calibration.get("min_confidence"),
        "temperature": calibration.get("temperature"),
        "calibration_target_event_recall_for_threshold_selection": calibration.get("target_event_recall"),
        "calibration_empirical_event_recall": calibration.get("achieved_event_recall"),
    }


def _verify_telemetry_integrity(
    test: list[dict],
    downlink: list[dict],
    calibration: dict,
    calibration_path: str | Path,
) -> tuple[dict[str, dict], dict[str, dict], dict]:
    declared_test_schema = _one_value(test, "input_schema_sha256")
    declared_down_schema = _one_value(downlink, "input_schema_sha256")
    if declared_test_schema != declared_down_schema:
        raise ValueError("final-test/downlink input schema hash mismatch")
    if calibration.get("input_schema_sha256") != declared_test_schema:
        raise ValueError("calibration input/preprocessing schema differs from final-test runtime")

    for record in test:
        validate_telemetry_record(
            record,
            expected_kind=FINAL_TEST_RECORD_KIND,
            require_artifact_identity=True,
        )
    for record in downlink:
        validate_telemetry_record(
            record,
            expected_kind=DOWNLINK_RECORD_KIND,
            require_artifact_identity=True,
        )

    test_by_file = _records_by_file(test, label="final-test")
    down_by_file = _records_by_file(downlink, label="downlink")
    if set(test_by_file) != set(down_by_file):
        raise ValueError("final-test and downlink logs do not cover the same files")

    identity: dict[str, object] = {}
    for key in IDENTITY_KEYS:
        test_value = _one_value(test, key)
        down_value = _one_value(downlink, key)
        if test_value != down_value:
            raise ValueError(f"final-test/downlink {key} mismatch")
        identity[key] = test_value

    test_record_version = _one_value(test, "schema_version")
    down_record_version = _one_value(downlink, "schema_version")
    if test_record_version != down_record_version:
        raise ValueError("final-test/downlink telemetry schema version mismatch")

    test_bundle_verified = _one_value(test, "deployment_bundle_verified")
    down_bundle_verified = _one_value(downlink, "deployment_bundle_verified")
    if test_bundle_verified != down_bundle_verified:
        raise ValueError("final-test/downlink bundle verification state mismatch")
    if identity["deployment_bundle_id"] is not None and test_bundle_verified is not True:
        raise ValueError("telemetry names a deployment bundle that was not verified")
    identity["deployment_bundle_verified"] = bool(test_bundle_verified)

    calibration_hash = sha256_file(calibration_path)
    if identity["policy_sha256"] != calibration_hash:
        raise ValueError("telemetry policy hash does not match the calibration policy artifact")
    if calibration.get("model_sha256") != identity["model_sha256"]:
        raise ValueError("calibration belongs to a different model")

    for file_name in sorted(test_by_file):
        test_record = test_by_file[file_name]
        down_record = down_by_file[file_name]
        if not test_record["input_observation_ok"] or not down_record["input_observation_ok"]:
            raise ValueError(f"cannot verify input identity for {file_name}: file observation failed")
        test_hash = test_record.get("input_sha256")
        down_hash = down_record.get("input_sha256")
        if test_hash != down_hash:
            raise ValueError(f"final-test/downlink input hash mismatch for {file_name}")
        if test_record.get("size_bytes") != down_record.get("size_bytes"):
            raise ValueError(f"final-test/downlink input size mismatch for {file_name}")

    identity["policy_artifact_sha256"] = calibration_hash
    identity["input_hashes_verified"] = True
    identity["telemetry_record_schema_version"] = test_record_version
    return test_by_file, down_by_file, identity


def _timing_summary(records: list[dict], key: str) -> dict:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return {"samples": 0, "avg": None, "p50": None, "p95": None, "p99": None}
    data = np.asarray(values, dtype=float)
    return {
        "samples": len(values),
        "avg": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def summarize(test_log: str | Path, downlink_log: str | Path, calibration_path: str | Path) -> dict:
    test = read_jsonl(test_log)
    downlink = read_jsonl(downlink_log)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if not test or not downlink:
        raise ValueError("final-test and downlink logs must be non-empty")

    _, down_by_file, identity = _verify_telemetry_integrity(
        test,
        downlink,
        calibration,
        calibration_path,
    )

    successful = [record for record in test if record.get("inference_ok")]
    y_true = np.array([int(record["true_class"]) for record in successful], dtype=int)
    y_pred = np.array([int(record["pred_class"]) for record in successful], dtype=int)
    scores = np.array([float(record["prob_event"]) for record in successful], dtype=float)
    if successful:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average=None, zero_division=0
        )
        auc = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else None
        pr_auc = float(average_precision_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else None
    else:
        precision = recall = f1 = np.array([0.0])
        auc = None
        pr_auc = None

    event_records = [record for record in test if int(record["true_class"]) == 1]
    background_records = [record for record in test if int(record["true_class"]) == 0]
    event_retained = sum(
        bool(down_by_file[record["file"]]["retained_for_downlink"]) for record in event_records
    )
    background_discarded = sum(
        not bool(down_by_file[record["file"]]["retained_for_downlink"])
        for record in background_records
    )
    event_retention_recall = event_retained / len(event_records) if event_records else 0.0
    background_rejection_rate = (
        background_discarded / len(background_records) if background_records else 0.0
    )

    source_bytes_total = sum(int(record["size_bytes"]) for record in downlink)
    source_bytes_retained = sum(
        int(record["size_bytes"])
        for record in downlink
        if record["retained_for_downlink"]
    )
    materialization_failures = sum(
        bool(record["retention_requested"]) and not bool(record["retained_for_downlink"])
        for record in downlink
    )
    fallback_count = sum(str(record["decision"]).endswith("fallback") for record in downlink)
    quality_fallback_count = sum(record["decision"] == "input_quality_fallback" for record in downlink)
    quality_guard_enabled = bool(_one_value(downlink, "input_quality_guard_enabled"))

    record_version = identity["telemetry_record_schema_version"]
    execution_provider = _one_value(test, "execution_provider") if int(record_version) >= 6 else None
    timing_scope = _one_value(test, "timing_scope") if int(record_version) >= 6 else None

    return {
        "schema_version": 7,
        "split_role": "final_test",
        "deployment_bundle_id": identity["deployment_bundle_id"],
        "model_sha256": identity["model_sha256"],
        "policy_sha256": identity["policy_sha256"],
        "input_schema_sha256": identity["input_schema_sha256"],
        "telemetry_integrity": identity,
        "final_test_sample_counts": {
            "samples_total": len(test),
            "event_samples": len(event_records),
            "background_samples": len(background_records),
            "successful_inferences": len(successful),
            "inference_failures": len(test) - len(successful),
        },
        "final_test_model_classification_metrics": {
            "evaluated_successful_inferences": len(successful),
            "accuracy": float(accuracy_score(y_true, y_pred)) if successful else 0.0,
            "event_precision": float(precision[0]),
            "event_recall": float(recall[0]),
            "event_false_negative_rate": float(1.0 - recall[0]),
            "event_f1": float(f1[0]),
            "roc_auc": auc,
            "pr_auc_average_precision": pr_auc,
        },
        "calibration_policy_metadata": _calibration_metadata(calibration),
        "final_test_input_quality_metrics": {
            "guard_enabled": quality_guard_enabled,
            "input_quality_fallback_tiles": int(quality_fallback_count),
            "input_quality_fallback_rate": float(quality_fallback_count / len(downlink)),
        },
        "final_test_downlink_retention_metrics": {
            "event_retention_recall": float(event_retention_recall),
            "background_rejection_rate": float(background_rejection_rate),
            "tiles_retained": sum(bool(record["retained_for_downlink"]) for record in downlink),
            "tiles_total": len(downlink),
            "fallback_tiles": int(fallback_count),
            "fallback_rate": float(fallback_count / len(downlink)),
            "downlink_materialization_failures": int(materialization_failures),
            "source_bytes_total": source_bytes_total,
            "source_bytes_retained": source_bytes_retained,
            "source_bytes_reduction_pct": (
                100.0 * (1.0 - source_bytes_retained / source_bytes_total)
                if source_bytes_total
                else 0.0
            ),
            "operational_link_bandwidth_measured": False,
        },
        "final_test_runtime_metrics": {
            "timing_scope": timing_scope,
            "execution_provider": execution_provider,
            "spacecraft_timing_measured": False,
            "input_observation_latency_ms": _timing_summary(test, "input_observation_latency_ms"),
            "preprocessing_latency_ms": _timing_summary(test, "preprocessing_latency_ms"),
            "input_quality_latency_ms": _timing_summary(test, "input_quality_latency_ms"),
            "onnx_inference_latency_ms": _timing_summary(test, "onnx_inference_latency_ms"),
            "policy_latency_ms": _timing_summary(test, "policy_latency_ms"),
            "end_to_end_tile_latency_ms": _timing_summary(test, "end_to_end_latency_ms"),
        },
    }


def _append_mapping(lines: list[str], payload: dict, *, level: int = 2) -> None:
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.extend(["", f"{'#' * level} {key}"])
            _append_mapping(lines, value, level=min(level + 1, 6))
        else:
            lines.append(f"- **{key}**: {value}")


def write_summary(metrics: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# Run summary", ""]
    _append_mapping(lines, metrics)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-log", required=True)
    parser.add_argument("--downlink-log", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    metrics = summarize(args.test_log, args.downlink_log, args.calib)
    write_summary(metrics, args.out_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
