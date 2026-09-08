from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .filesystem import assert_file_outputs_disjoint, staged_text_file
from .input_schema import find_dataset_input_schema, find_model_input_schema
from .policy import softmax
from .runtime import OnnxRunner
from .utils import discover_labeled_tiles, load_tile_numpy


def evaluate(model: str | Path, data: str | Path, *, temperature: float = 1.0) -> dict:
    runner = OnnxRunner(model)
    runner.assert_data_schema(data)
    items = discover_labeled_tiles(data)
    if not items:
        raise ValueError(f"no labeled tiles found under {data}")
    y_true: list[int] = []
    y_pred: list[int] = []
    event_scores: list[float] = []
    onnx_latencies: list[float] = []
    for path, cls in items:
        array = load_tile_numpy(path, input_schema=runner.input_schema)
        logits, latency = runner.logits_for_array(array)
        probs = softmax(logits, temperature=temperature)[0]
        y_true.append(cls)
        y_pred.append(int(probs.argmax()))
        event_scores.append(float(probs[1]))
        onnx_latencies.append(latency)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    auc = float(roc_auc_score(y_true, event_scores)) if len(set(y_true)) == 2 else None
    result = {
        "schema_version": 3,
        "model_sha256": runner.model_sha256,
        "input_schema_sha256": runner.input_schema_sha256,
        "input_band_ids": list(runner.band_ids),
        "preprocessing_version": runner.input_schema["preprocessing"]["version"],
        "samples": len(items),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "event_precision": float(precision[0]),
        "event_recall": float(recall[0]),
        "event_f1": float(f1[0]),
        "auc_roc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "timing_scope": "onnxruntime_session_run_only",
        "execution_provider": runner.selected_execution_provider,
        "host_measurement_only": True,
        "spacecraft_timing_measured": False,
        "onnx_inference_latency_ms": {
            "avg": float(np.mean(onnx_latencies)),
            "p95": float(np.percentile(onnx_latencies, 95)),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classification metrics; latency is ONNX Runtime session.run host timing only."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is not None:
        bundle_roots = [
            parent
            for parent in {Path(args.onnx).parent.resolve(), Path(args.onnx).resolve().parent}
            if (parent / "bundle.json").is_file()
        ]
        assert_file_outputs_disjoint(
            [args.out],
            protected_paths=[
                args.onnx,
                args.data,
                find_model_input_schema(args.onnx),
                find_dataset_input_schema(args.data),
                *bundle_roots,
            ],
        )
    result = evaluate(args.onnx, args.data, temperature=args.temperature)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        with staged_text_file(args.out) as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
