# Reproducibility

The synthetic dataset and training path use explicit seeds. The generator creates independent `train`, `calib`, `validation`, and `test` splits from separate child RNG streams and records their roles in `manifest.json`.

The lifecycle is deliberately separated:

- `train` fits model parameters;
- `calib` performs INT8 calibration and policy calibration;
- `validation` controls candidate acceptance;
- `test` produces final headline metrics only after acceptance and promotion.

Tests check that generated tile contents do not overlap across these four splits.

## Clean run

```bash
cd examples/phi2-eo-tile-filter
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_demo.py --n 200 --bands 3 --size 64 --epochs 3 --seed 0
python -m pytest -q
```

For a multispectral smoke run:

```bash
python scripts/run_demo.py --n 160 --bands 7 --size 32 --epochs 2 --seed 0 --output-root /tmp/phi2-7band
```

Generated artifacts record the four-way dataset manifest, model hashes, calibration policy and recall-bound evidence, validation-only quantization acceptance metrics, final-test metrics, exact byte-level downlink accounting, and an immutable deployment bundle. `models/deployment_state.json` identifies the active and previous bundle by content-derived bundle ID. The active bundle contains the exact model, policy, preprocessing metadata, and validation report used by runtime filtering.

For archival work, also save `python -m pip freeze` next to the run artifacts.

Deterministic training is intended for reproducible testing on the same software and hardware class. Bit-for-bit equality across different accelerator stacks is not claimed. Statistical confidence bounds do not make the synthetic data representative of an operational EO distribution.
