# Reproducibility

The synthetic dataset and training path use explicit seeds. The main generator creates independent `train`, `calib`, `validation`, and `test` splits from separate child RNG streams and records their roles in `manifest.json`.

The lifecycle is deliberately separated:

- `train` fits model parameters;
- `calib` performs INT8 calibration, policy calibration, and input-quality-guard calibration;
- `validation` controls candidate acceptance;
- `test` produces final headline metrics only after acceptance and promotion.

The optional robustness benchmark is generated only after deployment and never participates in candidate selection.

## Clean run

```bash
cd examples/phi2-eo-tile-filter
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_demo.py --n 200 --bands 3 --size 64 --epochs 3 --seed 0
python -m pytest -q
```

For a seven-band run plus robustness evaluation:

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

The benchmark seed controls nominal, degraded, corrupted, and OOD sample generation. Its manifest records the perturbation configuration and each sample's category/recipe. Running it again with the same code, input contract, seed, and configuration is intended to reproduce the same synthetic benchmark files.

Generated artifacts record model/bundle hashes, the calibrated input-quality guard, validation evidence, deployment-bound telemetry, final-test source-byte retention metrics, and the robustness report. The robustness prevalence simulation records the supplied event prevalences and its assumptions explicitly.

For archival work, save the Git commit and `python -m pip freeze` next to the generated run artifacts.

Deterministic training is intended for reproducible testing on the same software and hardware class. Bit-for-bit equality across different accelerator stacks is not claimed. Statistical confidence bounds, synthetic perturbations, and deterministic OOD patterns do not make the benchmark representative of an operational EO distribution or a physically validated sensor model.
