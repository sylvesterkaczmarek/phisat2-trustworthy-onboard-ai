from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any

from .utils import sha256_file

TELEMETRY_RECORD_SCHEMA_VERSION = 5
FINAL_TEST_RECORD_KIND = "final_test_inference"
DOWNLINK_RECORD_KIND = "downlink_decision"

_HEX = set(string.hexdigits.lower())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def resolve_artifact_identity(
    model_path: str | Path,
    policy_path: str | Path,
    *,
    model_sha256: str,
    input_schema_sha256: str,
    input_schema_path: str | Path,
    preprocessing_sha256: str,
    explicit_bundle_id: str | None = None,
) -> dict[str, Any]:
    model = Path(model_path).resolve(strict=False)
    policy = Path(policy_path).resolve(strict=False)
    schema = Path(input_schema_path).resolve(strict=False)
    policy_sha = sha256_file(policy)
    schema_file_sha = sha256_file(schema)

    bundle_id = explicit_bundle_id
    bundle_verified = False
    if bundle_id is not None and not _is_sha256(bundle_id):
        raise ValueError("deployment bundle id must be a 64-character SHA-256 identifier")

    manifest_path = model.parent / "bundle.json"
    if manifest_path.is_file() and policy.parent == model.parent:
        manifest = _read_json_object(manifest_path)
        manifest_id = manifest.get("bundle_id")
        if not _is_sha256(manifest_id):
            raise ValueError("deployment bundle manifest has an invalid bundle_id")
        if bundle_id is not None and bundle_id != manifest_id:
            raise ValueError("explicit deployment bundle id does not match bundle manifest")
        bundle_id = str(manifest_id)
        if manifest.get("model_sha256") != model_sha256:
            raise ValueError("deployment bundle manifest model hash does not match runtime model")
        if manifest.get("policy_sha256") != policy_sha:
            raise ValueError("deployment bundle manifest policy hash does not match runtime policy")
        if manifest.get("input_contract_sha256") != input_schema_sha256:
            raise ValueError("deployment bundle manifest input contract does not match runtime schema")
        if manifest.get("input_schema_file_sha256") != schema_file_sha:
            raise ValueError("deployment bundle manifest input-schema file hash does not match runtime schema")
        components = manifest.get("components")
        if not isinstance(components, dict):
            raise ValueError("deployment bundle manifest is missing component descriptors")
        expected_paths = {"model": model, "policy": policy, "input_schema": schema}
        for name, actual in expected_paths.items():
            descriptor = components.get(name)
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
                raise ValueError(f"deployment bundle is missing {name} component metadata")
            declared = (model.parent / descriptor["path"]).resolve(strict=False)
            if declared != actual:
                raise ValueError(f"runtime {name} path does not match deployment bundle component")
        bundle_verified = True

    return {
        "deployment_bundle_id": bundle_id,
        "deployment_bundle_verified": bundle_verified,
        "model_sha256": model_sha256,
        "policy_sha256": policy_sha,
        "input_schema_sha256": input_schema_sha256,
        "input_schema_file_sha256": schema_file_sha,
        "preprocessing_sha256": preprocessing_sha256,
    }


def _validate_quality_evidence(record: dict[str, Any]) -> None:
    enabled = record.get("input_quality_guard_enabled")
    if not isinstance(enabled, bool):
        raise ValueError("telemetry record is missing input quality guard state")
    method = record.get("input_quality_method")
    score = record.get("input_quality_score")
    threshold = record.get("input_quality_threshold")
    quality_ok = record.get("input_quality_ok")
    if enabled:
        if not isinstance(method, str) or not method:
            raise ValueError("enabled input quality guard requires a method")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or float(threshold) < 0.0:
            raise ValueError("enabled input quality guard requires a finite non-negative threshold")
        if score is not None and (
            not isinstance(score, (int, float)) or not math.isfinite(float(score)) or float(score) < 0.0
        ):
            raise ValueError("input quality score must be null or finite and non-negative")
        if quality_ok is not None and not isinstance(quality_ok, bool):
            raise ValueError("input_quality_ok must be null or boolean")
        if record.get("inference_ok") is True and (score is None or not isinstance(quality_ok, bool)):
            raise ValueError("successful inference with quality guard requires quality score and decision")
    else:
        if method is not None or score is not None or threshold is not None:
            raise ValueError("disabled input quality guard must not emit guard metrics")
        if record.get("inference_ok") is True and quality_ok is not True:
            raise ValueError("successful inference without quality guard must mark input_quality_ok true")
        if quality_ok not in (None, True):
            raise ValueError("disabled input quality guard has invalid input_quality_ok state")


def validate_telemetry_record(
    record: dict[str, Any],
    *,
    expected_kind: str | None = None,
    require_artifact_identity: bool = True,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("telemetry record must be an object")
    schema_version = record.get("schema_version")
    if schema_version not in (4, TELEMETRY_RECORD_SCHEMA_VERSION):
        raise ValueError("unsupported telemetry record schema version")
    kind = record.get("record_kind")
    if kind not in {FINAL_TEST_RECORD_KIND, DOWNLINK_RECORD_KIND}:
        raise ValueError("unsupported telemetry record kind")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"expected telemetry record kind {expected_kind}, found {kind}")
    if not isinstance(record.get("file"), str) or not record["file"]:
        raise ValueError("telemetry record is missing file identity")

    for key in ("model_sha256", "input_schema_sha256", "input_schema_file_sha256", "preprocessing_sha256"):
        if not _is_sha256(record.get(key)):
            raise ValueError(f"telemetry record has invalid {key}")
    policy_hash = record.get("policy_sha256")
    if require_artifact_identity and not _is_sha256(policy_hash):
        raise ValueError("telemetry record has invalid policy_sha256")
    if policy_hash is not None and not _is_sha256(policy_hash):
        raise ValueError("telemetry record has invalid policy_sha256")
    bundle_id = record.get("deployment_bundle_id")
    if bundle_id is not None and not _is_sha256(bundle_id):
        raise ValueError("telemetry record has invalid deployment_bundle_id")
    if not isinstance(record.get("deployment_bundle_verified"), bool):
        raise ValueError("telemetry record is missing deployment bundle verification state")
    if not isinstance(record.get("inference_ok"), bool):
        raise ValueError("telemetry record is missing inference status")
    if not isinstance(record.get("retention_requested"), bool):
        raise ValueError("telemetry record is missing retention request")
    if not isinstance(record.get("decision"), str) or not record["decision"]:
        raise ValueError("telemetry record is missing decision")
    if schema_version == TELEMETRY_RECORD_SCHEMA_VERSION:
        _validate_quality_evidence(record)

    input_hash = record.get("input_sha256")
    size_bytes = record.get("size_bytes")
    observation_ok = record.get("input_observation_ok")
    if not isinstance(observation_ok, bool):
        raise ValueError("telemetry record is missing input observation status")
    if observation_ok:
        if not _is_sha256(input_hash):
            raise ValueError("verified input observation requires input_sha256")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("verified input observation requires non-negative size_bytes")
    else:
        if input_hash is not None and not _is_sha256(input_hash):
            raise ValueError("telemetry input_sha256 must be null or valid SHA-256")
        if size_bytes is not None and (not isinstance(size_bytes, int) or size_bytes < 0):
            raise ValueError("telemetry size_bytes must be null or non-negative")

    if kind == FINAL_TEST_RECORD_KIND:
        if record.get("true_class") not in (0, 1):
            raise ValueError("final-test telemetry record is missing true_class")
    else:
        if not isinstance(record.get("downlink_materialized"), bool):
            raise ValueError("downlink telemetry record is missing materialization status")
        if not isinstance(record.get("retained_for_downlink"), bool):
            raise ValueError("downlink telemetry record is missing retained_for_downlink")
        error = record.get("downlink_error")
        if error is not None and not isinstance(error, str):
            raise ValueError("downlink_error must be null or a string")
        if record["retained_for_downlink"] and not record["retention_requested"]:
            raise ValueError("downlink record cannot retain data that the policy did not request")
    return record
