# Data

The demo generates deterministic synthetic tiles into four independent lifecycle splits:

- `train` is used only to fit model parameters.
- `calib` is used for INT8 calibration and for decision-threshold / temperature calibration.
- `validation` is used for model and quantization acceptance gates.
- `test` is reserved for final reporting after the candidate bundle has already been accepted and promoted.

The generator uses independent child RNG streams for the four splits and records their roles and counts in `manifest.json`. Tests also check that generated tile contents do not overlap between splits.

Synthetic tiles are stored as float32 `.npy` arrays in HWC layout and scaled to `[0, 1]`. This preserves arbitrary multispectral band counts without pretending that RGB image formats represent multispectral data.

The requested calibration recall is a threshold-selection target on the calibration sample. It is not presented as a population guarantee. The calibration artifact reports empirical recall together with a one-sided exact Clopper-Pearson lower confidence bound and the number of positive calibration examples.

Real EO data should use a documented, mission-specific preprocessing and radiometric-normalization path before being fed into this demonstrator.
