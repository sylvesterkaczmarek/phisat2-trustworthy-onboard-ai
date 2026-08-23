from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil

from .provenance import collect_run_environment
from .runtime import OnnxRunner


def benchmark(
    model_path: str | Path,
    *,
    iterations: int = 200,
    warmup: int = 20,
    intra_op_threads: int | None = None,
    seed: int = 0,
) -> dict:
    """Benchmark ONNX Runtime session.run only on a synthetic in-memory tensor."""
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    runner = OnnxRunner(model_path, intra_op_threads=intra_op_threads)
    rng = np.random.default_rng(seed)
    x = rng.random(
        (runner.spec.bands, runner.spec.size, runner.spec.size), dtype=np.float32
    )
    for _ in range(warmup):
        runner.logits_for_array(x)

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    latencies: list[float] = []
    started = time.perf_counter()
    for _ in range(iterations):
        _, latency_ms = runner.logits_for_array(x)
        latencies.append(latency_ms)
        peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed = time.perf_counter() - started
    data = np.asarray(latencies, dtype=float)
    repo_root = Path(__file__).resolve().parents[4]
    environment = collect_run_environment(
        repo_root,
        seed=seed,
        selected_execution_provider=runner.selected_execution_provider,
        run_parameters={
            "iterations": iterations,
            "warmup": warmup,
            "intra_op_threads": intra_op_threads,
            "model_path": str(model_path),
        },
    )
    return {
        "schema_version": 2,
        "benchmark_scope": "onnxruntime_session_run_only",
        "timing_source": "time.perf_counter",
        "host_measurement_only": True,
        "spacecraft_timing_measured": False,
        "execution_provider": runner.selected_execution_provider,
        "model_sha256": runner.model_sha256,
        "iterations": iterations,
        "warmup": warmup,
        "onnx_inference_latency_ms": {
            "avg": float(np.mean(data)),
            "p50": float(np.percentile(data, 50)),
            "p90": float(np.percentile(data, 90)),
            "p99": float(np.percentile(data, 99)),
        },
        "wall_clock_throughput_tiles_per_s": float(iterations / elapsed),
        "process_memory_mb": {
            "baseline_rss": baseline_rss / 1e6,
            "peak_rss": peak_rss / 1e6,
            "peak_delta": (peak_rss - baseline_rss) / 1e6,
        },
        "intra_op_threads": intra_op_threads,
        "seed": seed,
        "environment": environment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime session.run on the host CPU; this is not spacecraft timing."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--intra-op-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = benchmark(
        args.onnx,
        iterations=args.iters,
        warmup=args.warmup,
        intra_op_threads=args.intra_op_threads,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
