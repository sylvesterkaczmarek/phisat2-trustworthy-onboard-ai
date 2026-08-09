# Data

The demo generates deterministic synthetic tiles into independent `train`, `calib`, and `test` splits.

- Training uses only `train`.
- INT8 calibration and threshold/temperature selection use only `calib`.
- Final model and downlink metrics use only `test`.

Synthetic tiles are stored as float32 `.npy` arrays in HWC layout and scaled to `[0, 1]`. This preserves arbitrary multispectral band counts without pretending that RGB image formats represent multispectral data.

Real EO data should use a documented, mission-specific preprocessing and radiometric-normalization path before being fed into this demonstrator.
