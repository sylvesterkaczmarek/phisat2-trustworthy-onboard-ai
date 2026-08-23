from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score


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


def _calibration_metadata(calibration: dict) -> dict:
    if calibration.get("schema_version") == 3:
        stats = calibration.get("calibration_statistics", {})
        acceptance = calibration.get("calibration_acceptance", {})
        return {
            "event_threshold": calibration.get("event_threshold"),
            "min_confidence": calibration.get("min_confidence"),
            "temperature": calibration.get("temperature"),
            "temperature_fitted": calibration.get("temperature_fitted"),
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


def summarize(test_log: str | Path, downlink_log: str | Path, calibration_path: str | Path) -> dict:
    test = read_jsonl(test_log)
    downlink = read_jsonl(downlink_log)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if not test or not downlink:
        raise ValueError("final-test and downlink logs must be non-empty")
    if {r["file"] for r in test} != {r["file"] for r in downlink}:
        raise ValueError("final-test and downlink logs do not cover the same files")

    model_hash = _one_value(test, "model_sha256")
    if _one_value(downlink, "model_sha256") != model_hash:
        raise ValueError("final-test/downlink model hash mismatch")
    if calibration.get("model_sha256") != model_hash:
        raise ValueError("calibration belongs to a different model")

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

    down_by_file = {record["file"]: record for record in downlink}
    event_records = [record for record in test if int(record["true_class"]) == 1]
    background_records = [record for record in test if int(record["true_class"]) == 0]
    event_kept = sum(bool(down_by_file[record["file"]]["kept"]) for record in event_records)
    background_discarded = sum(not bool(down_by_file[record["file"]]["kept"]) for record in background_records)
    event_retention_recall = event_kept / len(event_records) if event_records else 0.0
    background_rejection_rate = background_discarded / len(background_records) if background_records else 0.0

    total_bytes = sum(int(record["size_bytes"]) for record in downlink)
    kept_bytes = sum(int(record["size_bytes"]) for record in downlink if record["kept"])
    latencies = [float(record["latency_ms"]) for record in successful if record.get("latency_ms") is not None]

    return {
        "schema_version": 3,
        "split_role": "final_test",
        "model_sha256": model_hash,
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
        "final_test_downlink_retention_metrics": {
            "event_retention_recall": float(event_retention_recall),
            "background_rejection_rate": float(background_rejection_rate),
            "tiles_kept": sum(bool(record["kept"]) for record in downlink),
            "tiles_total": len(downlink),
            "fallback_tiles": sum(str(record["decision"]).endswith("fallback") for record in downlink),
            "bytes_total": total_bytes,
            "bytes_kept": kept_bytes,
            "downlink_bytes_saved_pct": 100.0 * (1.0 - kept_bytes / total_bytes) if total_bytes else 0.0,
        },
        "final_test_runtime_metrics": {
            "avg_inference_latency_ms": float(np.mean(latencies)) if latencies else None,
            "p95_inference_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
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
