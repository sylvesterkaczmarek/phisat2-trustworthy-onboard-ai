from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .filesystem import (
    assert_paths_disjoint,
    assert_safe_tree_target,
    remove_stage,
    replace_tree_from_stage,
    sibling_stage_path,
)
from .policy import DecisionPolicy
from .runtime import OnnxRunner
from .telemetry import (
    DOWNLINK_RECORD_KIND,
    resolve_artifact_identity,
    validate_telemetry_record,
)
from .utils import discover_tile_files, sha256_file


def load_policy_artifact(
    path: str | Path,
    runner: OnnxRunner,
) -> tuple[DecisionPolicy, dict, str]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 4:
        raise ValueError(
            "unsupported calibration policy schema; policies must include an explicit input/preprocessing contract"
        )
    if payload.get("model_sha256") != runner.model_sha256:
        raise ValueError("calibration policy belongs to a different model")
    if payload.get("input_schema_sha256") != runner.input_schema_sha256:
        raise ValueError("calibration policy input/preprocessing schema does not match the model")
    if tuple(payload.get("input_band_ids", [])) != runner.band_ids:
        raise ValueError("calibration policy band ordering does not match the model input schema")
    if int(payload.get("preprocessing_version", -1)) != int(
        runner.input_schema["preprocessing"]["version"]
    ):
        raise ValueError("calibration policy preprocessing version does not match the model")
    if int(payload.get("bands", -1)) != runner.spec.bands or int(
        payload.get("size", -1)
    ) != runner.spec.size:
        raise ValueError("calibration policy input shape does not match model")
    acceptance = payload.get("calibration_acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise ValueError("calibration policy is not marked accepted")
    policy = DecisionPolicy(
        event_threshold=float(payload["event_threshold"]),
        min_confidence=float(payload["min_confidence"]),
        temperature=float(payload["temperature"]),
    )
    return policy, payload, sha256_file(path)


def load_policy(path: str | Path, runner: OnnxRunner) -> DecisionPolicy:
    policy, _, _ = load_policy_artifact(path, runner)
    return policy


def filter_tiles(
    model_path: str | Path,
    data_root: str | Path,
    policy_path: str | Path,
    *,
    downlink_root: str | Path,
    log_path: str | Path,
    deployment_bundle_id: str | None = None,
) -> dict:
    runner = OnnxRunner(model_path)
    policy, _, policy_sha = load_policy_artifact(policy_path, runner)
    data_root = Path(data_root).resolve(strict=False)
    runner.assert_data_schema(data_root)
    files = discover_tile_files(data_root)
    if not files:
        raise ValueError(f"no supported tiles found under {data_root}")

    downlink_root = assert_safe_tree_target(
        downlink_root,
        protected_paths=[data_root],
        operation="downlink tree replacement",
    )
    log_path = Path(log_path).resolve(strict=False)
    assert_paths_disjoint(log_path, data_root, description="downlink log/input paths")
    assert_paths_disjoint(log_path, downlink_root, description="downlink log/output paths")

    identity = resolve_artifact_identity(
        model_path,
        policy_path,
        model_sha256=runner.model_sha256,
        input_schema_sha256=runner.input_schema_sha256,
        input_schema_path=runner.input_schema_path,
        preprocessing_sha256=runner.preprocessing_sha256,
        explicit_bundle_id=deployment_bundle_id,
    )
    if identity["policy_sha256"] != policy_sha:
        raise ValueError("resolved policy identity changed while loading policy")

    stage_root = sibling_stage_path(downlink_root, label="stage")
    log_stage = sibling_stage_path(log_path, label="tmp")
    stage_root.mkdir(parents=True, exist_ok=False)
    log_stage.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    kept_bytes = 0
    kept_count = 0
    fallback_count = 0
    failure_count = 0
    copy_failure_count = 0
    unknown_size_files = 0

    try:
        with log_stage.open("w", encoding="utf-8") as handle:
            for path in files:
                record = runner.evaluate_file(
                    path,
                    policy,
                    policy_sha256=str(identity["policy_sha256"]),
                    deployment_bundle_id=identity["deployment_bundle_id"],
                    deployment_bundle_verified=bool(identity["deployment_bundle_verified"]),
                    record_kind=DOWNLINK_RECORD_KIND,
                )
                relative = path.relative_to(data_root)
                record["file"] = str(relative)

                size_bytes = record.get("size_bytes")
                if isinstance(size_bytes, int):
                    total_bytes += size_bytes
                else:
                    unknown_size_files += 1

                if str(record["decision"]).endswith("fallback"):
                    fallback_count += 1
                if not record["inference_ok"]:
                    failure_count += 1

                record["downlink_materialized"] = False
                record["retained_for_downlink"] = False
                record["downlink_error"] = None
                record["downlink_copy_sha256"] = None

                if record["retention_requested"]:
                    destination = stage_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(path, destination)
                        copied_hash = sha256_file(destination)
                        record["downlink_copy_sha256"] = copied_hash
                        expected_hash = record.get("input_sha256")
                        if expected_hash is not None and copied_hash != expected_hash:
                            destination.unlink(missing_ok=True)
                            raise RuntimeError(
                                "source changed between inference observation and downlink materialization"
                            )
                        record["downlink_materialized"] = True
                        record["retained_for_downlink"] = True
                        kept_count += 1
                        if isinstance(size_bytes, int):
                            kept_bytes += size_bytes
                    except Exception as exc:
                        copy_failure_count += 1
                        record["downlink_error"] = f"{type(exc).__name__}: {exc}"
                        record["downlink_materialized"] = False
                        record["retained_for_downlink"] = False

                validate_telemetry_record(
                    record,
                    expected_kind=DOWNLINK_RECORD_KIND,
                    require_artifact_identity=True,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

        replace_tree_from_stage(stage_root, downlink_root)
        os.replace(log_stage, log_path)
    finally:
        remove_stage(stage_root)
        remove_stage(log_stage)

    saved_fraction = 1.0 - kept_bytes / total_bytes if total_bytes else 0.0
    summary = {
        "schema_version": 4,
        "deployment_bundle_id": identity["deployment_bundle_id"],
        "deployment_bundle_verified": identity["deployment_bundle_verified"],
        "model_sha256": runner.model_sha256,
        "policy_sha256": identity["policy_sha256"],
        "input_schema_sha256": runner.input_schema_sha256,
        "input_schema_file_sha256": runner.input_schema_file_sha256,
        "preprocessing_sha256": runner.preprocessing_sha256,
        "tiles_total": len(files),
        "tiles_kept": kept_count,
        "fallback_tiles": fallback_count,
        "inference_failures": failure_count,
        "downlink_materialization_failures": copy_failure_count,
        "bytes_total": total_bytes,
        "bytes_kept": kept_bytes,
        "bytes_accounting_complete": unknown_size_files == 0,
        "unknown_size_files": unknown_size_files,
        "bandwidth_saved_fraction": saved_fraction,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--downlink-out", default="downlink")
    parser.add_argument("--log", default="logs/downlink.jsonl")
    args = parser.parse_args()
    filter_tiles(
        args.onnx,
        args.data,
        args.policy,
        downlink_root=args.downlink_out,
        log_path=args.log,
        deployment_bundle_id=args.bundle_id,
    )


if __name__ == "__main__":
    main()
