from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assurance.model_store import (
    build_bundle,
    promote_bundle,
    resolve_bundle,
    rollback,
    sha256_file,
    verify_bundle,
)
from assurance.summarize import summarize
from assurance.watchdog import run_watchdog


def _write_test_model(path: Path, marker: str, *, bands: int = 3, size: int = 8) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    input_value = helper.make_tensor_value_info(
        "input", tensor_proto.FLOAT, ["batch", bands, size, size]
    )
    output_value = helper.make_tensor_value_info(
        "output", tensor_proto.FLOAT, ["batch", bands, size, size]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        f"bundle-test-{marker}",
        [input_value],
        [output_value],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    helper.set_model_props(model, {"marker": marker})
    onnx.save(model, path)


def _valid_validation_payload(model_hash: str) -> dict:
    checks = {
        "classification_accuracy_drop": True,
        "classification_argmax_agreement": True,
        "classification_event_recall_drop": True,
        "classification_event_false_negative_rate_increase": True,
        "classification_pr_auc_drop": True,
        "policy_retention_decision_agreement": True,
        "policy_event_retention_recall_drop": True,
        "event_score_drift": True,
    }
    return {
        "schema_version": 2,
        "split_role": "validation",
        "validation_samples": 20,
        "validation_event_samples": 10,
        "validation_background_samples": 10,
        "fp32_sha256": "b" * 64,
        "int8_sha256": model_hash,
        "classification_metrics": {
            "quantization_regression": {
                "accuracy_drop": 0.0,
                "event_recall_drop": 0.0,
                "event_false_negative_rate_increase": 0.0,
                "event_f1_drop": 0.0,
                "roc_auc_drop": 0.0,
                "pr_auc_drop": 0.0,
                "argmax_agreement": 1.0,
            }
        },
        "policy_metrics": {
            "quantization_regression": {
                "retention_decision_agreement": 1.0,
                "event_retention_recall_drop": 0.0,
                "retained_fraction_change": 0.0,
            }
        },
        "score_drift_metrics": {
            "mean_absolute_event_score_drift": 0.0,
            "p95_absolute_event_score_drift": 0.0,
            "max_absolute_event_score_drift": 0.0,
        },
        "acceptance_criteria": {
            "max_classification_accuracy_drop": 0.02,
            "min_classification_argmax_agreement": 0.98,
            "max_classification_event_recall_drop": 0.02,
            "max_classification_event_false_negative_rate_increase": 0.02,
            "max_classification_pr_auc_drop": 0.02,
            "min_policy_retention_decision_agreement": 0.98,
            "max_policy_event_retention_recall_drop": 0.02,
            "max_event_score_drift": 0.05,
        },
        "acceptance_checks": checks,
        "accepted": True,
    }


def _write_candidate_artifacts(root: Path, marker: str) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model.onnx"
    _write_test_model(model, marker)
    model_hash = sha256_file(model)
    policy = root / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "split_role": "calibration",
                "model_sha256": model_hash,
                "bands": 3,
                "size": 8,
                "event_threshold": 0.8,
                "min_confidence": 0.6,
                "temperature": 1.0,
                "temperature_fitted": True,
                "calibration_statistics": {
                    "samples_total": 20,
                    "event_samples": 10,
                    "background_samples": 10,
                    "target_event_recall_for_threshold_selection": 0.95,
                    "empirical_event_recall": 1.0,
                    "event_captures": 10,
                    "event_precision_at_threshold": 1.0,
                    "roc_auc": 1.0,
                    "event_recall_lower_bound": 0.74,
                    "event_recall_confidence_level": 0.95,
                    "event_recall_bound_method": "clopper-pearson-one-sided-exact",
                },
                "calibration_acceptance": {
                    "required_min_event_recall_lower_bound": None,
                    "accepted": True,
                },
            }
        )
    )
    validation = root / "validation.json"
    validation.write_text(json.dumps(_valid_validation_payload(model_hash)))
    return model, policy, validation


def test_bundle_promotion_and_rollback_are_coherent(tmp_path: Path) -> None:
    store = tmp_path / "store"
    state = tmp_path / "deployment_state.json"

    model_a, policy_a, validation_a = _write_candidate_artifacts(tmp_path / "a", "a")
    bundle_a = tmp_path / "bundle-a"
    manifest_a = build_bundle(model_a, policy_a, validation_a, bundle_a)
    first_state = promote_bundle(bundle_a, store, state)
    assert first_state["active_bundle_id"] == manifest_a["bundle_id"]
    assert first_state["previous_bundle_id"] is None
    resolved_a = resolve_bundle(store, state)
    assert sha256_file(resolved_a["model"]) == manifest_a["model_sha256"]
    assert sha256_file(resolved_a["policy"]) == manifest_a["policy_sha256"]

    model_b, policy_b, validation_b = _write_candidate_artifacts(tmp_path / "b", "b")
    bundle_b = tmp_path / "bundle-b"
    manifest_b = build_bundle(model_b, policy_b, validation_b, bundle_b)
    second_state = promote_bundle(bundle_b, store, state)
    assert second_state["active_bundle_id"] == manifest_b["bundle_id"]
    assert second_state["previous_bundle_id"] == manifest_a["bundle_id"]

    rolled_back = rollback(store, state)
    assert rolled_back["active_bundle_id"] == manifest_a["bundle_id"]
    assert rolled_back["previous_bundle_id"] == manifest_b["bundle_id"]
    resolved_after = resolve_bundle(store, state)
    assert sha256_file(resolved_after["model"]) == manifest_a["model_sha256"]
    assert sha256_file(resolved_after["policy"]) == manifest_a["policy_sha256"]


def test_bundle_rejects_model_policy_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "mismatch-policy")
    payload = json.loads(policy.read_text())
    payload["model_sha256"] = "f" * 64
    policy.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different model"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_rejects_validation_report_hash_mismatch(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "mismatch-validation")
    payload = json.loads(validation.read_text())
    payload["int8_sha256"] = "e" * 64
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not cover"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_rejects_test_set_as_acceptance_evidence(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "wrong-split")
    payload = json.loads(validation.read_text())
    payload["split_role"] = "final_test"
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="validation split"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_bundle_recomputes_scientific_validation_checks(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "tampered-validation")
    payload = json.loads(validation.read_text())
    payload["classification_metrics"]["quantization_regression"]["event_recall_drop"] = 0.5
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="acceptance checks do not match"):
        build_bundle(model, policy, validation, tmp_path / "bundle")


def test_failed_calibration_cannot_be_promoted(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "failed-calibration")
    payload = json.loads(policy.read_text())
    payload["calibration_acceptance"]["required_min_event_recall_lower_bound"] = 0.9
    payload["calibration_acceptance"]["accepted"] = False
    policy.write_text(json.dumps(payload))
    bundle = tmp_path / "bundle"
    state = tmp_path / "deployment_state.json"
    with pytest.raises(ValueError, match="not marked accepted"):
        build_bundle(model, policy, validation, bundle)
    assert not bundle.exists()
    assert not state.exists()


def test_incomplete_or_corrupt_bundle_is_rejected(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "corrupt")
    bundle = tmp_path / "bundle"
    build_bundle(model, policy, validation, bundle)
    (bundle / "input_schema.json").unlink()
    with pytest.raises(FileNotFoundError, match="input_schema"):
        verify_bundle(bundle)


def test_orphaned_partial_state_file_does_not_change_active_bundle(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "partial-state")
    bundle = tmp_path / "bundle"
    manifest = build_bundle(model, policy, validation, bundle)
    store = tmp_path / "store"
    state = tmp_path / "deployment_state.json"
    promote_bundle(bundle, store, state)

    orphan = state.with_name(f".{state.name}.tmp-interrupted")
    orphan.write_text("{partial")
    resolved = resolve_bundle(store, state)
    assert resolved["bundle_id"] == manifest["bundle_id"]


def test_promoted_bundle_is_immutable_from_candidate_changes(tmp_path: Path) -> None:
    model, policy, validation = _write_candidate_artifacts(tmp_path / "candidate", "immutable")
    bundle = tmp_path / "bundle"
    manifest = build_bundle(model, policy, validation, bundle)
    store = tmp_path / "store"
    state = tmp_path / "deployment_state.json"
    promote_bundle(bundle, store, state)
    (bundle / "policy.json").write_text("{}")
    resolved = resolve_bundle(store, state)
    assert sha256_file(resolved["policy"]) == manifest["policy_sha256"]


def test_watchdog_does_not_need_shell(tmp_path: Path) -> None:
    rc = run_watchdog([sys.executable, "-c", "import sys; sys.exit(0)"], restarts=0, sleep_s=0)
    assert rc == 0


def test_summarizer_separates_final_test_model_and_downlink_metrics(tmp_path: Path) -> None:
    model_hash = "a" * 64
    test_records = [
        {"file": "background/0.npy", "model_sha256": model_hash, "true_class": 0, "pred_class": 0, "prob_event": 0.1, "inference_ok": True, "latency_ms": 1.0},
        {"file": "event/0.npy", "model_sha256": model_hash, "true_class": 1, "pred_class": 1, "prob_event": 0.9, "inference_ok": True, "latency_ms": 2.0},
        {"file": "event/1.npy", "model_sha256": model_hash, "true_class": 1, "pred_class": 0, "prob_event": 0.4, "inference_ok": True, "latency_ms": 3.0},
    ]
    down_records = [
        {"file": "background/0.npy", "model_sha256": model_hash, "kept": False, "decision": "confident_background", "size_bytes": 100},
        {"file": "event/0.npy", "model_sha256": model_hash, "kept": True, "decision": "event", "size_bytes": 200},
        {"file": "event/1.npy", "model_sha256": model_hash, "kept": True, "decision": "low_confidence_fallback", "size_bytes": 300},
    ]
    test_log = tmp_path / "test.jsonl"
    down_log = tmp_path / "down.jsonl"
    test_log.write_text("".join(json.dumps(r) + "\n" for r in test_records))
    down_log.write_text("".join(json.dumps(r) + "\n" for r in down_records))
    calib = tmp_path / "calib.json"
    calib.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "model_sha256": model_hash,
                "event_threshold": 0.8,
                "min_confidence": 0.6,
                "temperature": 1.0,
                "calibration_statistics": {
                    "samples_total": 20,
                    "event_samples": 10,
                    "target_event_recall_for_threshold_selection": 0.95,
                    "empirical_event_recall": 1.0,
                    "event_recall_lower_bound": 0.74,
                    "event_recall_confidence_level": 0.95,
                    "event_recall_bound_method": "clopper-pearson-one-sided-exact",
                },
                "calibration_acceptance": {
                    "required_min_event_recall_lower_bound": None,
                    "accepted": True,
                },
            }
        )
    )
    metrics = summarize(test_log, down_log, calib)
    assert metrics["split_role"] == "final_test"
    downlink = metrics["final_test_downlink_retention_metrics"]
    model = metrics["final_test_model_classification_metrics"]
    assert downlink["tiles_kept"] == 2
    assert downlink["bytes_kept"] == 500
    assert downlink["downlink_bytes_saved_pct"] == pytest.approx(100.0 / 6.0)
    assert downlink["event_retention_recall"] == 1.0
    assert model["event_recall"] == 0.5
    assert metrics["calibration_policy_metadata"]["calibration_event_recall_lower_bound"] == 0.74
