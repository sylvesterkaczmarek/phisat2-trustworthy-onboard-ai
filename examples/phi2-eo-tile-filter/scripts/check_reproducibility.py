from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _inspect_run(root: Path) -> dict:
    required = (
        "tiles/input_schema.json",
        "tiles/manifest.json",
        "calibration.json",
        "models/deployment_state.json",
        "logs/test.jsonl",
        "logs/downlink.jsonl",
        "reports/model_validation.json",
        "reports/metrics.json",
        "reports/summary.md",
        "reports/run_environment.json",
    )
    for relative in required:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty demo artifact: {path}")

    example_root = Path(__file__).resolve().parents[1]
    repo_root = example_root.parents[1]
    example_src = example_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(example_src) not in sys.path:
        sys.path.insert(0, str(example_src))

    from assurance.model_store import resolve_bundle, verify_bundle

    active = resolve_bundle(
        root / "models" / "bundles",
        root / "models" / "deployment_state.json",
        slot="active",
    )
    manifest = verify_bundle(active["bundle_dir"])
    metrics = _load_json(root / "reports" / "metrics.json")
    if metrics.get("deployment_bundle_id") != active["bundle_id"]:
        raise ValueError("final-test metrics are not bound to the active deployment bundle")
    if metrics.get("split_role") != "final_test":
        raise ValueError("metrics do not identify the final test split")
    if metrics.get("telemetry_integrity", {}).get("input_hashes_verified") is not True:
        raise ValueError("final-test input hashes were not verified")

    stable_metrics = dict(metrics)
    stable_metrics.pop("final_test_runtime_metrics", None)

    environment = _load_json(root / "reports" / "run_environment.json")
    if not environment.get("dependency_fingerprint_sha256"):
        raise ValueError("run environment is missing the dependency fingerprint")
    if not environment.get("environment_fingerprint_sha256"):
        raise ValueError("run environment is missing the environment fingerprint")

    return {
        "bundle_id": active["bundle_id"],
        "bundle_manifest": manifest,
        "stable_metrics": stable_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two seeded demo runs and require deterministic scientific outputs."
    )
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()

    first = _inspect_run(args.first.resolve())
    second = _inspect_run(args.second.resolve())

    if first["bundle_id"] != second["bundle_id"]:
        raise SystemExit("reproducibility failure: deployment bundle IDs differ")
    if first["bundle_manifest"] != second["bundle_manifest"]:
        raise SystemExit("reproducibility failure: deployment bundle manifests differ")
    if first["stable_metrics"] != second["stable_metrics"]:
        raise SystemExit("reproducibility failure: deterministic final-test metrics differ")

    print(
        json.dumps(
            {
                "reproducible": True,
                "bundle_id": first["bundle_id"],
                "runtime_metrics_excluded": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
