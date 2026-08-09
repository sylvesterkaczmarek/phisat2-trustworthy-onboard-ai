# Reproducibility

The synthetic dataset and training path use explicit seeds. The generator creates separate `train`, `calib`, and `test` splits, so threshold selection is not evaluated on the same samples used for the headline test metrics.

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

Generated artifacts record model hashes, input hashes, calibration policy, test metrics, FP32/INT8 comparison, and exact byte-level downlink savings. For archival work, also save `python -m pip freeze` next to the run artifacts.

Deterministic training is intended for reproducible testing on the same software and hardware class. Bit-for-bit equality across different accelerator stacks is not claimed.
