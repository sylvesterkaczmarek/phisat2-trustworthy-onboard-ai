from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic EO robustness benchmark against an active demo deployment."
    )
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--samples-per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--event-prevalences", default="0.01,0.05,0.10")
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--brightness-shift", type=float, default=0.12)
    parser.add_argument("--band-gain-drift", type=float, default=0.25)
    parser.add_argument("--band-offset-drift", type=float, default=0.08)
    parser.add_argument("--cloud-opacity", type=float, default=0.70)
    parser.add_argument("--spatial-shift-px", type=int, default=3)
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parents[1]
    repo_root = example_root.parents[1]
    example_src = example_root / "src"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(example_src) not in sys.path:
        sys.path.insert(0, str(example_src))

    from assurance.model_store import resolve_bundle
    from phi2_tile_filter.bandwidth_filter import filter_tiles
    from phi2_tile_filter.filesystem import assert_safe_workspace_root
    from phi2_tile_filter.provenance import collect_run_environment, write_run_environment
    from phi2_tile_filter.robustness_benchmark import generate_benchmark, summarize_benchmark
    from phi2_tile_filter.runtime import OnnxRunner

    root = assert_safe_workspace_root(args.output_root.resolve(), operation="robustness benchmark output")
    models = root / "models"
    logs = root / "logs"
    reports = root / "reports"
    benchmark_root = root / "robustness_benchmark"
    benchmark_downlink = root / "robustness_downlink"
    benchmark_log = logs / "robustness_downlink.jsonl"
    report_path = reports / "robustness_benchmark.json"
    environment_path = reports / "robustness_environment.json"

    active = resolve_bundle(
        models / "bundles",
        models / "deployment_state.json",
        slot="active",
    )
    manifest = generate_benchmark(
        benchmark_root,
        active["input_schema"],
        samples_per_category=args.samples_per_category,
        seed=args.seed,
        noise_std=args.noise_std,
        brightness_shift=args.brightness_shift,
        band_gain_drift=args.band_gain_drift,
        band_offset_drift=args.band_offset_drift,
        cloud_opacity=args.cloud_opacity,
        spatial_shift_px=args.spatial_shift_px,
        overwrite=True,
    )
    downlink_summary = filter_tiles(
        active["model"],
        benchmark_root,
        active["policy"],
        downlink_root=benchmark_downlink,
        log_path=benchmark_log,
        deployment_bundle_id=active["bundle_id"],
    )
    prevalences = tuple(
        float(value.strip())
        for value in args.event_prevalences.split(",")
        if value.strip()
    )
    report = summarize_benchmark(
        benchmark_root / "benchmark_manifest.json",
        benchmark_log,
        event_prevalences=prevalences,
    )

    provider = OnnxRunner(active["model"]).selected_execution_provider
    environment = collect_run_environment(
        repo_root,
        seed=args.seed,
        selected_execution_provider=provider,
        run_parameters=vars(args),
    )
    write_run_environment(environment_path, environment)
    report["run_environment"] = {
        "path": str(environment_path),
        "git_commit_sha": environment["git"]["commit_sha"],
        "git_dirty": environment["git"]["dirty"],
        "dependency_fingerprint_sha256": environment["dependency_fingerprint_sha256"],
        "environment_fingerprint_sha256": environment["environment_fingerprint_sha256"],
        "reference_requirements_sha256": environment["reference_environment"]["sha256"],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "benchmark_manifest": str(benchmark_root / "benchmark_manifest.json"),
        "benchmark_samples": len(manifest["samples"]),
        "downlink_summary": downlink_summary,
        "report": str(report_path),
        "environment": str(environment_path),
        "category_metrics": report["categories"],
        "prevalence_simulation": report["prevalence_simulation"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
