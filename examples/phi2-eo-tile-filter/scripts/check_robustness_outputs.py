from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify robustness benchmark smoke artifacts.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--samples-per-category", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = root / "robustness_benchmark" / "benchmark_manifest.json"
    report_path = root / "reports" / "robustness_benchmark.json"
    environment_path = root / "reports" / "robustness_environment.json"
    log_path = root / "logs" / "robustness_downlink.jsonl"

    for path in (manifest_path, report_path, environment_path, log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty robustness artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_categories = {"nominal", "degraded", "corrupted", "ood"}
    expected_samples = args.samples_per_category * len(expected_categories)
    if len(manifest.get("samples", [])) != expected_samples:
        raise SystemExit(
            f"expected {expected_samples} robustness samples, got {len(manifest.get('samples', []))}"
        )
    if set(report.get("categories", {})) != expected_categories:
        raise SystemExit("robustness report is missing one or more benchmark categories")
    if not report.get("run_environment", {}).get("dependency_fingerprint_sha256"):
        raise SystemExit("robustness report is missing its dependency fingerprint")

    print(
        json.dumps(
            {
                "robustness_artifacts_verified": True,
                "categories": sorted(expected_categories),
                "samples": expected_samples,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
