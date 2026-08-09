from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assurance.model_store import promote, rollback, sha256_file
from assurance.summarize import summarize
from assurance.watchdog import run_watchdog


def test_model_store_promote_and_rollback(tmp_path: Path) -> None:
    candidate_a = tmp_path / "a.onnx"
    candidate_b = tmp_path / "b.onnx"
    candidate_a.write_bytes(b"model-a")
    candidate_b.write_bytes(b"model-b")
    active = tmp_path / "active.onnx"
    previous = tmp_path / "previous.onnx"
    manifest = tmp_path / "state.json"
    promote(candidate_a, active, previous, manifest)
    promote(candidate_b, active, previous, manifest)
    assert active.read_bytes() == b"model-b"
    assert previous.read_bytes() == b"model-a"
    rollback(active, previous, manifest)
    assert active.read_bytes() == b"model-a"
    assert previous.read_bytes() == b"model-b"
    state = json.loads(manifest.read_text())
    assert state["active_sha256"] == sha256_file(active)


def test_watchdog_does_not_need_shell(tmp_path: Path) -> None:
    rc = run_watchdog([sys.executable, "-c", "import sys; sys.exit(0)"], restarts=0, sleep_s=0)
    assert rc == 0


def test_summarizer_uses_kept_and_bytes(tmp_path: Path) -> None:
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
    calib.write_text(json.dumps({"model_sha256": model_hash, "target_event_recall": 0.95, "event_threshold": 0.8, "min_confidence": 0.6, "temperature": 1.0}))
    metrics = summarize(test_log, down_log, calib)
    assert metrics["tiles_kept"] == 2
    assert metrics["bytes_kept"] == 500
    assert metrics["bandwidth_saved_pct"] == pytest.approx(100.0 / 6.0)
    assert metrics["downlink_event_recall"] == 1.0
    assert metrics["fallback_tiles"] == 1
