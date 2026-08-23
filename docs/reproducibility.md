# Reproducibility

The repository separates **supported dependency ranges** from a **pinned reference environment**:

- `examples/phi2-eo-tile-filter/pyproject.toml` is the canonical source of supported package ranges.
- `examples/phi2-eo-tile-filter/requirements-reference.txt` pins one concrete direct-dependency environment for repeatable reference runs.

The broad `requirements.txt` duplicate was removed so dependency declarations cannot silently drift in two places.

## Supported and reference installs

For normal development with supported ranges:

```bash
cd examples/phi2-eo-tile-filter
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For the pinned reference environment:

```bash
cd examples/phi2-eo-tile-filter
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reference.txt
python -m pip install --no-deps -e .
```

The reference file pins direct project/test dependencies rather than every platform-specific transitive wheel. Every complete run therefore also records the **full installed-distribution set and its SHA-256 fingerprint**, which is the authoritative record of what was actually present at execution time.

## Automatic run provenance

`run_demo.py` writes `reports/run_environment.json`. `run_robustness_benchmark.py` writes `reports/robustness_environment.json`.

Each provenance artifact records:

- Git commit SHA and branch when Git is available;
- whether the work tree is dirty when that can be determined;
- Python version, implementation, and executable;
- operating system, release, architecture, and platform string;
- CPU model and logical CPU count where available;
- CUDA availability/runtime and visible GPU names where available;
- selected and available ONNX Runtime execution providers;
- versions of NumPy, PyTorch, ONNX, ONNX Runtime, ONNX Script, Pillow, scikit-learn, SciPy, psutil, pytest, and Ruff;
- the complete installed Python distribution map;
- the SHA-256 of `requirements-reference.txt`;
- a dependency fingerprint and a broader environment fingerprint;
- the run seed and command parameters.

Environment variables are deliberately not dumped into the artifact because they may contain credentials or unrelated private configuration.

## Deterministic data lifecycle

The main generator creates independent `train`, `calib`, `validation`, and `test` splits from separate child RNG streams and records their roles in `manifest.json`.

- `train` fits model parameters.
- `calib` performs INT8 calibration, policy calibration, and input-quality-guard calibration.
- `validation` controls candidate acceptance.
- `test` produces final headline metrics only after acceptance and promotion.

The optional robustness benchmark is generated only after deployment and never participates in candidate selection. Its seed controls nominal, degraded, corrupted, and OOD sample generation, and its manifest records the perturbation configuration and each sample's category/recipe.

## Reproducing a complete run

```bash
python scripts/run_demo.py \
  --n 160 --bands 7 --size 32 --epochs 2 --seed 0 \
  --output-root /tmp/phi2-7band

python scripts/run_robustness_benchmark.py \
  --output-root /tmp/phi2-7band \
  --samples-per-category 20 \
  --seed 101 \
  --event-prevalences 0.01,0.05,0.10
```

Archive the generated `reports/run_environment.json`, `reports/robustness_environment.json`, manifests, deployment bundle, telemetry, and reports together. A separate manual `pip freeze` is no longer required to identify the installed Python environment, although it can still be useful for external archival workflows.

## CI coverage

CI tests Python 3.11 and Python 3.12. The Python 3.11 job installs from the supported ranges; the Python 3.12 job installs the pinned reference environment and also runs the standalone smoke demo. Both jobs compile the code and run the complete pytest suite. A lightweight Ruff check targets high-confidence syntax/name errors without imposing an unrelated repository-wide style rewrite.

## Timing reproducibility

Runtime telemetry separates:

- input observation/hash latency;
- preprocessing latency;
- input-quality-guard latency;
- ONNX Runtime `session.run` latency;
- probability/policy evaluation latency;
- end-to-end per-tile host wall-clock latency.

The standalone ONNX benchmark measures only `session.run` on an in-memory tensor. Reports record the execution provider and explicitly state that host timing is **not spacecraft timing**. Hardware qualification, WCET, thermal behaviour, accelerator scheduling, I/O contention, and flight-computer timing require target-hardware measurement.

## Limits

Deterministic training is intended for reproducible testing on the same software and hardware class. Bit-for-bit equality across different BLAS, accelerator, kernel, compiler, or hardware stacks is not claimed. Statistical confidence bounds, synthetic perturbations, and deterministic OOD patterns do not make the benchmark representative of an operational EO distribution or a physically validated sensor model.
