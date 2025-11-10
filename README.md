# PhiSat-2 Trustworthy Onboard AI
[![CI](https://github.com/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17567181.svg)](https://doi.org/10.5281/zenodo.17567181)
[![Discussions](https://img.shields.io/github/discussions/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai)](https://github.com/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai/discussions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![PhiSat-2 Trustworthy Onboard AI](assets/social/github-card.png)

Compact PyTorch → ONNX → INT8 pipeline for onboard inference on EO small-sat/CubeSat platforms. Includes an Earth Observation tile triage example that mirrors a PhiSat-2 style onboard selection step. Runs locally on a laptop or a small SBC. See the runnable demo in `examples/phi2-eo-tile-filter`.

## Project overview

- Train a tiny CNN on small tiles.
- Export FP32 ONNX and quantize to static INT8 (QDQ) with ONNX Runtime.
- Benchmark latency and memory on the ORT CPU Execution Provider.
- Calibrate a confidence threshold to hit a target recall.
- Filter tiles for event-triggered downlink to save bandwidth.
- Ship assurance hooks: watchdog, telemetry log, rollback, and a run summary.

## Why this is useful
- End-to-end path from training to a compact INT8 artifact suitable for onboard execution (PyTorch → ONNX → ORT INT8).
- Calibrated confidence gate to meet a target recall, letting you trade downlink volume against science yield.
- Assurance hooks: watchdog, fallback-to-downlink on low confidence, rollback to last-known-good, and JSONL telemetry (latency + hashes) for audit.
- Deterministic, CI-checked demo with fixed seeds and small scripts; easy to port to HIL or ARM/aarch64 ORT.
- Produces measurable stats (accuracy, confusion, latency, bandwidth saved) to support requirement verification.

## Features

- Minimal dependencies (PyTorch, ONNX, ONNX Runtime; a few small extras).
- Clear scripts: `train`, `export_onnx`, `quantize_ptq`, `infer_onnx`, `bench_onnxruntime`, `bandwidth_filter`.
- Self-contained synthetic dataset for quick runs; swap to Sentinel-2 crops and recalibrate.
- CI workflow that smoke-tests the pipeline on push and PR.
- Seeded runs and JSONL logs for reproducibility and audit.
- Laptop or small SBC ready (ORT CPU EP) with a path to ARM/aarch64.
- Assurance utilities in `assurance/` (watchdog, telemetry, summarizer, rollback).

## ESA PhiSat-2 context

This repository reflects workflows used in missions with onboard processing, such as ESA’s PhiSat-2 CubeSat for AI-enabled EO. The example filters tiles onboard and prioritizes downlink of useful data. Mission page: https://earth.esa.int/eogateway/missions/phisat-2

## Quick start

```bash
cd examples/phi2-eo-tile-filter
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt onnxscript==0.1.0 scikit-learn

# data → train → export → quantize → eval
python -m data.synth --out ./tiles --n 200 --bands 3 --size 64
python -m src.train --data ./tiles --epochs 12 --base 32 --lr 0.003
python -m src.export_onnx --weights runs/tinycnn.pt --out models/tinycnn_fp32.onnx --bands 3 --size 64 --base 32
python -m src.quantize_ptq --onnx models/tinycnn_fp32.onnx --calib ./tiles/val --out models/tinycnn_int8.onnx --size 64 --bands 3
python -m src.infer_onnx --onnx models/tinycnn_int8.onnx --data ./tiles/val --size 64 --bands 3

# calibrate → filter → telemetry → report
python -m src.calibrate_threshold --onnx models/tinycnn_int8.onnx --data ./tiles/val --target_recall 0.95 --out calibration.json
mkdir -p logs
python -m src.bandwidth_filter --onnx models/tinycnn_int8.onnx --data ./tiles/val --calibration calibration.json --downlink_out downlink --log logs/downlink.jsonl
THR=$(python -c "import json; print(json.load(open('calibration.json'))['threshold'])")
python ../../assurance/telemetry_log.py --onnx models/tinycnn_int8.onnx --data ./tiles/val --out logs/val.jsonl --threshold "$THR"
python ../../assurance/summarize.py --val_log logs/val.jsonl --downlink_log logs/downlink.jsonl --val_dir tiles/val --calib calibration.json --out_dir reports
cat reports/summary.md
```

## Outputs

- `logs/downlink.jsonl` decisions used by the bandwidth filter
- `logs/val.jsonl` per tile telemetry with probabilities and latency
- `reports/summary.md` and `reports/metrics.json` summary for quick review
- `models/tinycnn_fp32.onnx` and `models/tinycnn_int8.onnx` artifacts

## Results snapshot

Numbers from the latest synthetic run in `examples/phi2-eo-tile-filter`.
| Metric | Value |
|---|---|
| Threshold | 0.678 |
| Recall | 0.95 |
| Precision | 1.00 |
| AUC | 1.00 |
| Avg latency (ms) | 0.459 |
| Tiles kept | 19 / 40 |
| Bandwidth saved | 54.9% |

## Assurance hooks

See `assurance/`.
- `watchdog.py` restarts a failing inference command a few times
- `telemetry_log.py` emits one JSON line per tile with hashes and timings
- `summarize.py` converts logs into a small Markdown and JSON metrics report
- `rollback.sh` swaps back to the last good FP32 model

## File layout

```text
.
├─ assurance/                   # watchdog, rollback, telemetry, summary
├─ examples/
│  └─ phi2-eo-tile-filter/
│     ├─ data/                  # synthetic tiles
│     ├─ src/                   # train, export, quantize, infer, bench, filter
│     ├─ logs/ and reports/     # created by the quick start
│     └─ models/                # TinyCNN
└─ .github/workflows/ci.yml     # smoke test pipeline
```

## Extending

- Swap synthetic tiles for Sentinel-2 crops and recalibrate the gate for the same recall target.
- Add OpenVINO or TensorRT export paths for specific hardware.
- Log exact downlink bytes and confidence histograms for fuller telemetry.
- Replace `TinyCNN` with a stronger model and keep the PyTorch → ONNX → INT8 interface stable.

## Requirements

- Python 3.12
- torch ≥ 2.2, onnx 1.19.1, onnxruntime 1.23.2
- numpy, Pillow, scikit-learn, psutil, pytest

## Cite this demo

If you use or adapt this repository, please cite

> Kaczmarek, S. (2025). *PhiSat-2 Trustworthy Onboard AI*. Zenodo. https://doi.org/10.5281/zenodo.17567181

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17567181.svg)](https://doi.org/10.5281/zenodo.17567181)

**BibTeX**
```bibtex
@software{Kaczmarek_2025_PhiSat2_Onboard_AI,
  author    = {Sylvester Kaczmarek},
  title     = {{PhiSat-2 Trustworthy Onboard AI}},
  year      = {2025},
  publisher = {Zenodo},
  url       = {https://github.com/sylvesterkaczmarek/phisat2-trustworthy-onboard-ai},
  doi       = {10.5281/zenodo.17567181}
}
```

## License

MIT. See [LICENSE](LICENSE).

© Sylvester Kaczmarek · https://www.sylvesterkaczmarek.com
