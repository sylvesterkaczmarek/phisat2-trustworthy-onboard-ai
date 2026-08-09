from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from .policy import softmax
from .runtime import OnnxRunner
from .utils import discover_labeled_tiles, load_tile_numpy


def validate_models(
    fp32_path: str | Path,
    int8_path: str | Path,
    data_root: str | Path,
    *,
    max_accuracy_drop: float = 0.02,
    min_argmax_agreement: float = 0.98,
) -> dict:
    if max_accuracy_drop < 0 or not (0.0 <= min_argmax_agreement <= 1.0):
        raise ValueError("invalid validation tolerances")
    fp32 = OnnxRunner(fp32_path)
    int8 = OnnxRunner(int8_path)
    if fp32.spec != int8.spec:
        raise ValueError("FP32 and INT8 model input specifications differ")
    items = discover_labeled_tiles(data_root)
    if not items:
        raise ValueError("test set is empty")

    y_true: list[int] = []
    fp32_pred: list[int] = []
    int8_pred: list[int] = []
    probability_drift: list[float] = []
    for path, cls in items:
        array = load_tile_numpy(path, bands=fp32.spec.bands, size=fp32.spec.size)
        fp32_logits, _ = fp32.logits_for_array(array)
        int8_logits, _ = int8.logits_for_array(array)
        fp32_probs = softmax(fp32_logits)[0]
        int8_probs = softmax(int8_logits)[0]
        y_true.append(cls)
        fp32_pred.append(int(fp32_probs.argmax()))
        int8_pred.append(int(int8_probs.argmax()))
        probability_drift.append(float(np.max(np.abs(fp32_probs - int8_probs))))

    fp32_accuracy = float(accuracy_score(y_true, fp32_pred))
    int8_accuracy = float(accuracy_score(y_true, int8_pred))
    drop = fp32_accuracy - int8_accuracy
    agreement = float(np.mean(np.asarray(fp32_pred) == np.asarray(int8_pred)))
    result = {
        "schema_version": 1,
        "samples": len(items),
        "fp32_sha256": fp32.model_sha256,
        "int8_sha256": int8.model_sha256,
        "fp32_accuracy": fp32_accuracy,
        "int8_accuracy": int8_accuracy,
        "accuracy_drop": drop,
        "argmax_agreement": agreement,
        "max_probability_drift": float(np.max(probability_drift)),
        "mean_probability_drift": float(np.mean(probability_drift)),
        "max_accuracy_drop_allowed": max_accuracy_drop,
        "min_argmax_agreement_required": min_argmax_agreement,
    }
    if drop > max_accuracy_drop:
        raise RuntimeError(f"INT8 accuracy drop {drop:.4f} exceeds {max_accuracy_drop:.4f}")
    if agreement < min_argmax_agreement:
        raise RuntimeError(f"FP32/INT8 agreement {agreement:.4f} is below {min_argmax_agreement:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.98)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = validate_models(
        args.fp32,
        args.int8,
        args.data,
        max_accuracy_drop=args.max_accuracy_drop,
        min_argmax_agreement=args.min_argmax_agreement,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
