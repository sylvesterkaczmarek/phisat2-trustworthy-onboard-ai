from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .policy import softmax
from .runtime import OnnxRunner
from .utils import discover_labeled_tiles, load_tile_numpy


def evaluate(model: str | Path, data: str | Path, *, temperature: float = 1.0) -> dict:
    runner = OnnxRunner(model)
    items = discover_labeled_tiles(data)
    if not items:
        raise ValueError(f"no labeled tiles found under {data}")
    y_true: list[int] = []
    y_pred: list[int] = []
    event_scores: list[float] = []
    latencies: list[float] = []
    for path, cls in items:
        array = load_tile_numpy(path, bands=runner.spec.bands, size=runner.spec.size)
        logits, latency = runner.logits_for_array(array)
        probs = softmax(logits, temperature=temperature)[0]
        y_true.append(cls)
        y_pred.append(int(probs.argmax()))
        event_scores.append(float(probs[1]))
        latencies.append(latency)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    auc = float(roc_auc_score(y_true, event_scores)) if len(set(y_true)) == 2 else None
    result = {
        "schema_version": 1,
        "model_sha256": runner.model_sha256,
        "samples": len(items),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "event_precision": float(precision[0]),
        "event_recall": float(recall[0]),
        "event_f1": float(f1[0]),
        "auc_roc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args.onnx, args.data, temperature=args.temperature)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
