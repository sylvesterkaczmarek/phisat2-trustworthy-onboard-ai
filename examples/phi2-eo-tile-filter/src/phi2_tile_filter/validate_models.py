from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .bandwidth_filter import load_policy
from .policy import DecisionPolicy, softmax
from .runtime import OnnxRunner
from .utils import discover_labeled_tiles, load_tile_numpy


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, event_scores: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    event_recall = float(recall[0])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "event_precision": float(precision[0]),
        "event_recall": event_recall,
        "event_false_negative_rate": float(1.0 - event_recall),
        "event_f1": float(f1[0]),
        "roc_auc": float(roc_auc_score(y_true, event_scores)),
        "pr_auc_average_precision": float(average_precision_score(y_true, event_scores)),
    }


def _policy_metrics(y_true: np.ndarray, kept: np.ndarray) -> dict:
    event_mask = y_true == 1
    background_mask = y_true == 0
    return {
        "event_retention_recall": float(np.mean(kept[event_mask])),
        "background_rejection_rate": float(np.mean(~kept[background_mask])),
        "retained_fraction": float(np.mean(kept)),
    }


def _policy_kept(policy: DecisionPolicy, probabilities: np.ndarray) -> np.ndarray:
    decisions = [
        policy.decide(
            prob_event=float(row[1]),
            max_prob=float(np.max(row)),
            inference_ok=True,
        )[0]
        for row in probabilities
    ]
    return np.asarray(decisions, dtype=bool)


def validate_models(
    fp32_path: str | Path,
    int8_path: str | Path,
    data_root: str | Path,
    policy_path: str | Path,
    *,
    max_accuracy_drop: float = 0.02,
    min_argmax_agreement: float = 0.98,
    max_event_recall_drop: float = 0.02,
    max_event_fnr_increase: float = 0.02,
    max_pr_auc_drop: float = 0.02,
    min_policy_decision_agreement: float = 0.98,
    max_event_retention_recall_drop: float = 0.02,
    max_event_score_drift: float = 0.05,
) -> dict:
    nonnegative = {
        "max_accuracy_drop": max_accuracy_drop,
        "max_event_recall_drop": max_event_recall_drop,
        "max_event_fnr_increase": max_event_fnr_increase,
        "max_pr_auc_drop": max_pr_auc_drop,
        "max_event_retention_recall_drop": max_event_retention_recall_drop,
        "max_event_score_drift": max_event_score_drift,
    }
    if any(value < 0.0 for value in nonnegative.values()):
        raise ValueError("validation drop/drift tolerances must be non-negative")
    for name, value in {
        "min_argmax_agreement": min_argmax_agreement,
        "min_policy_decision_agreement": min_policy_decision_agreement,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    fp32 = OnnxRunner(fp32_path)
    int8 = OnnxRunner(int8_path)
    if fp32.spec != int8.spec:
        raise ValueError("FP32 and INT8 model input specifications differ")
    policy = load_policy(policy_path, int8)

    items = discover_labeled_tiles(data_root)
    if not items:
        raise ValueError("validation set is empty")

    y_true: list[int] = []
    fp32_logits_list: list[np.ndarray] = []
    int8_logits_list: list[np.ndarray] = []
    for path, cls in items:
        array = load_tile_numpy(path, bands=fp32.spec.bands, size=fp32.spec.size)
        fp32_logits, _ = fp32.logits_for_array(array)
        int8_logits, _ = int8.logits_for_array(array)
        y_true.append(cls)
        fp32_logits_list.append(fp32_logits[0])
        int8_logits_list.append(int8_logits[0])

    y = np.asarray(y_true, dtype=np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("validation set must contain both background and event examples")
    fp32_logits = np.stack(fp32_logits_list)
    int8_logits = np.stack(int8_logits_list)
    fp32_probs = softmax(fp32_logits, temperature=policy.temperature)
    int8_probs = softmax(int8_logits, temperature=policy.temperature)
    fp32_pred = fp32_probs.argmax(axis=1).astype(np.int64)
    int8_pred = int8_probs.argmax(axis=1).astype(np.int64)
    fp32_scores = fp32_probs[:, 1]
    int8_scores = int8_probs[:, 1]

    fp32_classification = _classification_metrics(y, fp32_pred, fp32_scores)
    int8_classification = _classification_metrics(y, int8_pred, int8_scores)
    fp32_kept = _policy_kept(policy, fp32_probs)
    int8_kept = _policy_kept(policy, int8_probs)
    fp32_policy = _policy_metrics(y, fp32_kept)
    int8_policy = _policy_metrics(y, int8_kept)

    score_drift = np.abs(fp32_scores - int8_scores)
    classification_regression = {
        "accuracy_drop": float(fp32_classification["accuracy"] - int8_classification["accuracy"]),
        "event_recall_drop": float(fp32_classification["event_recall"] - int8_classification["event_recall"]),
        "event_false_negative_rate_increase": float(
            int8_classification["event_false_negative_rate"]
            - fp32_classification["event_false_negative_rate"]
        ),
        "event_f1_drop": float(fp32_classification["event_f1"] - int8_classification["event_f1"]),
        "roc_auc_drop": float(fp32_classification["roc_auc"] - int8_classification["roc_auc"]),
        "pr_auc_drop": float(
            fp32_classification["pr_auc_average_precision"]
            - int8_classification["pr_auc_average_precision"]
        ),
        "argmax_agreement": float(np.mean(fp32_pred == int8_pred)),
    }
    policy_regression = {
        "retention_decision_agreement": float(np.mean(fp32_kept == int8_kept)),
        "event_retention_recall_drop": float(
            fp32_policy["event_retention_recall"] - int8_policy["event_retention_recall"]
        ),
        "retained_fraction_change": float(int8_policy["retained_fraction"] - fp32_policy["retained_fraction"]),
    }
    score_drift_metrics = {
        "mean_absolute_event_score_drift": float(np.mean(score_drift)),
        "p95_absolute_event_score_drift": float(np.percentile(score_drift, 95)),
        "max_absolute_event_score_drift": float(np.max(score_drift)),
    }

    criteria = {
        "max_classification_accuracy_drop": float(max_accuracy_drop),
        "min_classification_argmax_agreement": float(min_argmax_agreement),
        "max_classification_event_recall_drop": float(max_event_recall_drop),
        "max_classification_event_false_negative_rate_increase": float(max_event_fnr_increase),
        "max_classification_pr_auc_drop": float(max_pr_auc_drop),
        "min_policy_retention_decision_agreement": float(min_policy_decision_agreement),
        "max_policy_event_retention_recall_drop": float(max_event_retention_recall_drop),
        "max_event_score_drift": float(max_event_score_drift),
    }
    checks = {
        "classification_accuracy_drop": classification_regression["accuracy_drop"] <= max_accuracy_drop,
        "classification_argmax_agreement": classification_regression["argmax_agreement"] >= min_argmax_agreement,
        "classification_event_recall_drop": classification_regression["event_recall_drop"] <= max_event_recall_drop,
        "classification_event_false_negative_rate_increase": (
            classification_regression["event_false_negative_rate_increase"] <= max_event_fnr_increase
        ),
        "classification_pr_auc_drop": classification_regression["pr_auc_drop"] <= max_pr_auc_drop,
        "policy_retention_decision_agreement": (
            policy_regression["retention_decision_agreement"] >= min_policy_decision_agreement
        ),
        "policy_event_retention_recall_drop": (
            policy_regression["event_retention_recall_drop"] <= max_event_retention_recall_drop
        ),
        "event_score_drift": score_drift_metrics["max_absolute_event_score_drift"] <= max_event_score_drift,
    }
    accepted = all(checks.values())

    return {
        "schema_version": 2,
        "split_role": "validation",
        "validation_samples": len(items),
        "validation_event_samples": int(np.sum(y == 1)),
        "validation_background_samples": int(np.sum(y == 0)),
        "fp32_sha256": fp32.model_sha256,
        "int8_sha256": int8.model_sha256,
        "policy_model_sha256": int8.model_sha256,
        "policy_temperature": float(policy.temperature),
        "classification_metrics": {
            "fp32": fp32_classification,
            "int8": int8_classification,
            "quantization_regression": classification_regression,
        },
        "policy_metrics": {
            "fp32": fp32_policy,
            "int8": int8_policy,
            "quantization_regression": policy_regression,
        },
        "score_drift_metrics": score_drift_metrics,
        "acceptance_criteria": criteria,
        "acceptance_checks": checks,
        "accepted": bool(accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--data", required=True, help="Validation split directory")
    parser.add_argument("--policy", required=True, help="Calibration policy bound to the INT8 model")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.98)
    parser.add_argument("--max-event-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-event-fnr-increase", type=float, default=0.02)
    parser.add_argument("--max-pr-auc-drop", type=float, default=0.02)
    parser.add_argument("--min-policy-decision-agreement", type=float, default=0.98)
    parser.add_argument("--max-event-retention-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-event-score-drift", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = validate_models(
        args.fp32,
        args.int8,
        args.data,
        args.policy,
        max_accuracy_drop=args.max_accuracy_drop,
        min_argmax_agreement=args.min_argmax_agreement,
        max_event_recall_drop=args.max_event_recall_drop,
        max_event_fnr_increase=args.max_event_fnr_increase,
        max_pr_auc_drop=args.max_pr_auc_drop,
        min_policy_decision_agreement=args.min_policy_decision_agreement,
        max_event_retention_recall_drop=args.max_event_retention_recall_drop,
        max_event_score_drift=args.max_event_score_drift,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if result["accepted"] is not True:
        failed = [name for name, passed in result["acceptance_checks"].items() if not passed]
        raise RuntimeError("INT8 validation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
