from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "phi2-eo-tile-filter"
EXAMPLE_SRC = EXAMPLE_ROOT / "src"
if str(EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_SRC))

from phi2_tile_filter.bandwidth_filter import load_policy_artifact  # noqa: E402
from phi2_tile_filter.filesystem import assert_paths_disjoint, staged_text_file  # noqa: E402
from phi2_tile_filter.runtime import OnnxRunner  # noqa: E402
from phi2_tile_filter.telemetry import (  # noqa: E402
    FINAL_TEST_RECORD_KIND,
    TELEMETRY_RECORD_SCHEMA_VERSION,
    resolve_artifact_identity,
    validate_telemetry_record,
)
from phi2_tile_filter.utils import CLASS_NAMES, discover_labeled_tiles  # noqa: E402


def emit_telemetry(
    model_path: str | Path,
    data_root: str | Path,
    policy_path: str | Path,
    output: str | Path,
    *,
    deployment_bundle_id: str | None = None,
) -> dict:
    runner = OnnxRunner(model_path)
    policy, _, policy_sha = load_policy_artifact(policy_path, runner)
    data_root = Path(data_root).resolve(strict=False)
    runner.assert_data_schema(data_root)
    items = discover_labeled_tiles(data_root)
    if not items:
        raise ValueError(f"no labeled tiles found under {data_root}")

    output = Path(output).resolve(strict=False)
    assert_paths_disjoint(output, data_root, description="telemetry output/input paths")
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

    failures = 0
    quality_fallbacks = 0
    with staged_text_file(output) as handle:
        for path, true_class in items:
            record = runner.evaluate_file(
                path,
                policy,
                policy_sha256=str(identity["policy_sha256"]),
                deployment_bundle_id=identity["deployment_bundle_id"],
                deployment_bundle_verified=bool(identity["deployment_bundle_verified"]),
                record_kind=FINAL_TEST_RECORD_KIND,
            )
            record["file"] = str(path.relative_to(data_root))
            record["true_class"] = int(true_class)
            record["true_class_name"] = CLASS_NAMES[true_class]
            if not record["inference_ok"]:
                failures += 1
            if record["decision"] == "input_quality_fallback":
                quality_fallbacks += 1
            validate_telemetry_record(
                record,
                expected_kind=FINAL_TEST_RECORD_KIND,
                require_artifact_identity=True,
            )
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    result = {
        "schema_version": 5,
        "record_schema_version": TELEMETRY_RECORD_SCHEMA_VERSION,
        "samples": len(items),
        "inference_failures": failures,
        "input_quality_guard_enabled": policy.input_quality_guard is not None,
        "input_quality_fallbacks": quality_fallbacks,
        "deployment_bundle_id": identity["deployment_bundle_id"],
        "deployment_bundle_verified": identity["deployment_bundle_verified"],
        "model_sha256": runner.model_sha256,
        "policy_sha256": identity["policy_sha256"],
        "input_schema_sha256": runner.input_schema_sha256,
        "input_schema_file_sha256": runner.input_schema_file_sha256,
        "preprocessing_sha256": runner.preprocessing_sha256,
        "input_band_ids": list(runner.band_ids),
        "preprocessing_version": runner.input_schema["preprocessing"]["version"],
        "output": str(output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--out", default="logs/test.jsonl")
    args = parser.parse_args()
    emit_telemetry(
        args.onnx,
        args.data,
        args.policy,
        args.out,
        deployment_bundle_id=args.bundle_id,
    )


if __name__ == "__main__":
    main()
