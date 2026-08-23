from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort

from .input_schema import (
    assert_dataset_schema_compatible,
    band_ids,
    validate_model_input_schema_binding,
)
from .policy import DecisionPolicy, softmax
from .telemetry import DOWNLINK_RECORD_KIND, TELEMETRY_RECORD_SCHEMA_VERSION
from .utils import load_tile_numpy, sha256_file


@dataclass(frozen=True)
class ModelInputSpec:
    input_name: str
    bands: int
    size: int


def input_spec_from_session(session: ort.InferenceSession) -> ModelInputSpec:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"expected one model input, found {len(inputs)}")
    shape = inputs[0].shape
    if len(shape) != 4:
        raise ValueError(f"expected NCHW input, found shape {shape}")
    bands = shape[1]
    height = shape[2]
    width = shape[3]
    if not isinstance(bands, int) or not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("model must have static channel and spatial dimensions")
    if height != width:
        raise ValueError("demo expects square input tiles")
    return ModelInputSpec(inputs[0].name, bands, height)


def _stat_fingerprint(stat_result: Any) -> tuple[int, int, int, int]:
    return (
        int(getattr(stat_result, "st_dev", 0)),
        int(getattr(stat_result, "st_ino", 0)),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


class OnnxRunner:
    def __init__(
        self,
        model_path: str | Path,
        *,
        intra_op_threads: int | None = None,
        input_schema_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.input_schema, self.input_schema_sha256, self.input_schema_path = (
            validate_model_input_schema_binding(self.model_path, input_schema_path)
        )
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            if intra_op_threads <= 0:
                raise ValueError("intra_op_threads must be positive")
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.spec = input_spec_from_session(self.session)
        self.model_sha256 = sha256_file(self.model_path)
        self.input_schema_file_sha256 = sha256_file(self.input_schema_path)
        self.preprocessing_sha256 = str(self.input_schema["preprocessing_sha256"])
        tensor = self.input_schema["tensor"]
        schema_bands = len(tensor["bands"])
        if schema_bands != self.spec.bands:
            raise ValueError("model channel count does not match input schema band count")
        if int(tensor["height"]) != self.spec.size or int(tensor["width"]) != self.spec.size:
            raise ValueError("model spatial dimensions do not match input schema")
        self.band_ids = band_ids(self.input_schema)

    def assert_data_schema(self, data_root: str | Path) -> None:
        assert_dataset_schema_compatible(data_root, self.input_schema)

    def logits_for_array(self, array_chw: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.asarray(array_chw)
        if x.dtype != np.float32:
            raise ValueError(f"expected float32 model input, got {x.dtype}")
        expected = (self.spec.bands, self.spec.size, self.spec.size)
        if x.shape != expected:
            raise ValueError(f"expected tile shape {expected}, got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("model input contains non-finite values")
        started = perf_counter()
        outputs = self.session.run(None, {self.spec.input_name: x[None, ...]})
        latency_ms = (perf_counter() - started) * 1000.0
        if len(outputs) != 1:
            raise ValueError("expected one model output")
        logits = np.asarray(outputs[0])
        if logits.shape != (1, 2):
            raise ValueError(f"expected logits shape (1, 2), got {logits.shape}")
        return logits, latency_ms

    def evaluate_file(
        self,
        path: str | Path,
        policy: DecisionPolicy,
        *,
        policy_sha256: str | None = None,
        deployment_bundle_id: str | None = None,
        deployment_bundle_verified: bool = False,
        record_kind: str = DOWNLINK_RECORD_KIND,
    ) -> dict:
        """Evaluate one input without allowing file/I/O failures to escape.

        File stat, hashing, preprocessing, inference, and decision stages are all
        inside the conservative fallback boundary. The record distinguishes a
        requested fallback retention from later downlink materialization.
        """
        path = Path(path)
        record: dict[str, Any] = {
            "schema_version": TELEMETRY_RECORD_SCHEMA_VERSION,
            "record_kind": record_kind,
            "file": str(path),
            "deployment_bundle_id": deployment_bundle_id,
            "deployment_bundle_verified": bool(deployment_bundle_verified),
            "model_sha256": self.model_sha256,
            "policy_sha256": policy_sha256,
            "input_schema_sha256": self.input_schema_sha256,
            "input_schema_file_sha256": self.input_schema_file_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "input_band_ids": list(self.band_ids),
            "preprocessing_version": self.input_schema["preprocessing"]["version"],
            "input_sha256": None,
            "size_bytes": None,
            "input_mtime_ns": None,
            "input_observation_ok": False,
            "event_threshold": policy.event_threshold,
            "min_confidence": policy.min_confidence,
            "temperature": policy.temperature,
            "inference_ok": False,
            "error": None,
            "error_type": None,
            "failure_stage": None,
            "pred_class": None,
            "prob_event": None,
            "max_prob": None,
            "retention_requested": True,
            "kept": True,
            "decision": "inference_failure_fallback",
            "latency_ms": None,
        }

        failure_stage = "stat_input"
        try:
            before = path.stat()
            before_fingerprint = _stat_fingerprint(before)
            record["size_bytes"] = int(before.st_size)
            record["input_mtime_ns"] = int(before.st_mtime_ns)

            failure_stage = "hash_input"
            record["input_sha256"] = sha256_file(path)
            after_hash = path.stat()
            if _stat_fingerprint(after_hash) != before_fingerprint:
                raise RuntimeError("input file changed while it was being hashed")
            record["input_observation_ok"] = True

            failure_stage = "preprocess_input"
            array = load_tile_numpy(path, input_schema=self.input_schema)
            after_preprocess = path.stat()
            if _stat_fingerprint(after_preprocess) != before_fingerprint:
                record["input_observation_ok"] = False
                raise RuntimeError("input file changed while it was being preprocessed")

            failure_stage = "onnx_inference"
            logits, latency_ms = self.logits_for_array(array)
            probabilities = softmax(logits, temperature=policy.temperature)[0]
            prob_event = float(probabilities[1])
            max_prob = float(probabilities.max())

            failure_stage = "decision_policy"
            retain, decision = policy.decide(
                prob_event=prob_event,
                max_prob=max_prob,
                inference_ok=True,
            )
            record.update(
                {
                    "inference_ok": True,
                    "error": None,
                    "error_type": None,
                    "failure_stage": None,
                    "pred_class": int(probabilities.argmax()),
                    "prob_event": prob_event,
                    "max_prob": max_prob,
                    "retention_requested": bool(retain),
                    "kept": bool(retain),
                    "decision": decision,
                    "latency_ms": float(latency_ms),
                }
            )
        except Exception as exc:
            retain, decision = policy.decide(
                prob_event=float("nan"),
                max_prob=float("nan"),
                inference_ok=False,
            )
            record.update(
                {
                    "inference_ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_type": type(exc).__name__,
                    "failure_stage": failure_stage,
                    "pred_class": None,
                    "prob_event": None,
                    "max_prob": None,
                    "retention_requested": bool(retain),
                    "kept": bool(retain),
                    "decision": decision,
                    "latency_ms": None,
                }
            )
        return record
