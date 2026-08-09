# PhiSat-2 Trustworthy Onboard AI

![PhiSat-2 Trustworthy Onboard AI](assets/social/github-social-card-phisat2-onboard-ai.png)

[![CI](https://github.com/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17567181.svg)](https://doi.org/10.5281/zenodo.17567181)

Research demonstrator for deterministic Earth-observation tile triage, PyTorch to ONNX deployment, static INT8 quantization, conservative downlink fallback, telemetry, and model rollback. The workflow is inspired by onboard EO processing such as PhiSat-2, but it is independent software and is not ESA or PhiSat-2 flight code.

## At a glance

```mermaid
flowchart LR
    A[Train split] --> B[TinyCNN]
    B --> C[FP32 ONNX]
    C --> D[INT8 QDQ]
    E[Calibration split] --> F[Event threshold and temperature]
    D --> F
    D --> G[Held-out model validation]
    F --> H[Downlink policy]
    I[Test split] --> G
    I --> H
    H --> J[Telemetry and report]
```

The design separates model training, policy calibration, and final evaluation. A calibration policy is cryptographically bound to the model SHA-256 and cannot silently be reused with a different model.

## What is implemented

- deterministic synthetic EO data generation with independent `train`, `calib`, and `test` splits
- arbitrary multispectral band counts through NumPy tile stacks
- deterministic TinyCNN training with architecture metadata stored in the checkpoint
- PyTorch to FP32 ONNX export with ONNX validation and numerical equivalence check
- static QDQ INT8 quantization with calibration data kept separate from final test data
- held-out FP32 versus INT8 accuracy and prediction-agreement regression checks
- calibrated event threshold targeting a requested event recall
- optional deterministic temperature scaling on the calibration split
- conservative fallback that downlinks low-confidence tiles and inference failures
- model/policy hash binding
- per-tile input and model SHA-256 telemetry
- exact byte-level bandwidth accounting
- atomic known-good model promotion and rollback
- bounded watchdog without `shell=True`
- CI that runs the full multispectral pipeline

## Decision policy

A tile is retained when it is predicted to contain the target event, when confidence is below the configured minimum, or when preprocessing/inference fails. Only confidently classified background data are discarded.

This is intentionally conservative about science-data loss. See [docs/assurance.md](docs/assurance.md).

## Quick start

```bash
git clone https://github.com/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai.git
cd phisat2-trustworthy-onboard-ai/examples/phi2-eo-tile-filter
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_demo.py --n 200 --bands 3 --size 64 --epochs 3 --seed 0
```

A multispectral run uses the same pipeline:

```bash
python scripts/run_demo.py --n 160 --bands 7 --size 32 --epochs 2 --seed 0 --output-root /tmp/phi2-7band
```

The synthetic generator uses `.npy` arrays so band counts greater than RGB are preserved without pretending that PNG is a multispectral format.

## Generated outputs

A complete run produces:

```text
calibration.json
logs/test.jsonl
logs/downlink.jsonl
models/active.onnx
models/model_state.json
reports/model_validation.json
reports/metrics.json
reports/summary.md
```

The report uses held-out test telemetry for precision, recall, AUC, latency, fallback count, event capture, and exact byte-level bandwidth savings. Calibration metrics are kept separate from test metrics.

## Validation gates

The pipeline fails rather than silently continuing when:

- the checkpoint architecture metadata do not match the exporter
- PyTorch and FP32 ONNX outputs exceed the configured numerical tolerance
- ONNX structural validation fails
- the quantized artifact lacks the expected QDQ operators
- held-out INT8 accuracy degrades beyond the allowed threshold
- FP32 and INT8 prediction agreement falls below the required level
- a calibration file belongs to a different model hash or input shape

## Known-good model handling

After validation, the INT8 candidate can be promoted atomically:

```bash
python ../../assurance/model_store.py promote \
  --candidate models/tinycnn_int8.onnx \
  --active models/active.onnx \
  --previous models/previous.onnx \
  --manifest models/model_state.json
```

Rollback is location-independent:

```bash
bash ../../assurance/rollback.sh
```

## Watchdog

The watchdog takes an explicit argv list and does not execute a shell command string:

```bash
python ../../assurance/watchdog.py --restarts 3 --sleep-s 2 --cwd . -- \
  python -m phi2_tile_filter.infer_onnx --onnx models/active.onnx --data tiles/test
```

## Data interface

The demonstration accepts image files for 1, 3, or 4 bands and `.npy` arrays for arbitrary multispectral input. Floating-point NumPy tiles must already be scaled to `[0, 1]`. Real mission data should have a documented preprocessing and radiometric normalization pipeline rather than relying on this toy loader.

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md). Training seeds Python, NumPy, PyTorch, and DataLoader shuffling. CI performs the end-to-end pipeline using a seven-band synthetic dataset so multispectral support cannot regress unnoticed.

## What this repository does not claim

This repository is a software and assurance demonstrator. The synthetic benchmark is deliberately simple. It does not establish PhiSat-2 performance, flight readiness, radiation tolerance, worst-case execution time, hardware qualification, operational EO accuracy, formal safety, or mission-level fault tolerance.

The watchdog and rollback helpers are process-level examples rather than flight-qualified FDIR. Real onboard deployment would also require representative sensor data, hardware-in-the-loop validation, resource and thermal limits, persistent storage guarantees, signed model/update handling, fault injection, and mission-specific acceptance criteria.

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── assurance/
│   ├── model_store.py
│   ├── rollback.sh
│   ├── summarize.py
│   ├── telemetry_log.py
│   └── watchdog.py
├── docs/
│   ├── assurance.md
│   └── reproducibility.md
├── examples/phi2-eo-tile-filter/
│   ├── data/
│   ├── scripts/run_demo.py
│   ├── src/phi2_tile_filter/
│   ├── tests/
│   ├── pyproject.toml
│   └── requirements.txt
├── CITATION.cff
├── LICENSE
├── Makefile
└── README.md
```

## Requirements

- Python 3.11 or newer
- PyTorch 2.x
- ONNX and ONNX Runtime for export, quantization, and inference
- no accelerator is required for the CI-scale CPU demonstration

Exact dependency bounds are defined in [`examples/phi2-eo-tile-filter/pyproject.toml`](examples/phi2-eo-tile-filter/pyproject.toml) and mirrored in [`requirements.txt`](examples/phi2-eo-tile-filter/requirements.txt).

## Cite this repository

If you use or adapt this repository, please cite:

> Kaczmarek, S. (2025). *PhiSat-2 Trustworthy Onboard AI*. Zenodo. https://doi.org/10.5281/zenodo.17567181

```bibtex
@software{Kaczmarek_2025_PhiSat2_Trustworthy_Onboard_AI,
  author    = {Sylvester Kaczmarek},
  title     = {{PhiSat-2 Trustworthy Onboard AI}},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17567181},
  url       = {https://doi.org/10.5281/zenodo.17567181}
}
```

## License

MIT. See [LICENSE](LICENSE).

© **Sylvester Kaczmarek** · [https://www.sylvesterkaczmarek.com](https://www.sylvesterkaczmarek.com)
