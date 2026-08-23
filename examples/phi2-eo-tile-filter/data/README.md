# Data

The demo generates deterministic synthetic tiles into four independent lifecycle splits:

- `train` is used only to fit model parameters.
- `calib` is used for INT8 calibration and decision-threshold / temperature calibration.
- `validation` is used for model and quantization acceptance gates.
- `test` is reserved for final reporting after the candidate bundle has already been accepted and promoted.

The generator uses independent child RNG streams for the four splits and records their roles and counts in `manifest.json`. Tests also check that generated tile contents do not overlap between splits.

## Input contract

Every generated dataset also contains `input_schema.json`. Its canonical SHA-256 is recorded in `manifest.json`, embedded in the training checkpoint, carried into FP32 and INT8 ONNX metadata, recorded by calibration and validation evidence, and bound into the deployment bundle.

The schema describes the scientific meaning of the tensor interface rather than only its shape. It can record:

- ordered band identifiers and names;
- optional centre wavelength or wavelength range metadata;
- source and model tensor layouts;
- tile height and width;
- source and model dtypes;
- expected radiometric range;
- normalization procedure and version;
- nodata handling;
- preprocessing implementation and version;
- resize behaviour and channel-conversion policy.

The synthetic dataset uses float32 `.npy` tiles in HWC layout, values already scaled to `[0, 1]`, identity normalization, and a reject-nodata policy. The model receives float32 NCHW tensors.

A seven-band synthetic contract therefore looks conceptually like this:

```json
{
  "schema_version": 2,
  "contract_type": "eo-input-preprocessing",
  "preprocessing_sha256": "<canonical preprocessing fingerprint>",
  "tensor": {
    "model_layout": "NCHW",
    "source_layout": "HWC",
    "height": 32,
    "width": 32,
    "dtype": "float32",
    "bands": [
      {"index": 0, "id": "band_01", "name": "synthetic_band_01", "wavelength_nm": null},
      {"index": 1, "id": "band_02", "name": "synthetic_band_02", "wavelength_nm": null},
      {"index": 2, "id": "band_03", "name": "synthetic_band_03", "wavelength_nm": null},
      {"index": 3, "id": "band_04", "name": "synthetic_band_04", "wavelength_nm": null},
      {"index": 4, "id": "band_05", "name": "synthetic_band_05", "wavelength_nm": null},
      {"index": 5, "id": "band_06", "name": "synthetic_band_06", "wavelength_nm": null},
      {"index": 6, "id": "band_07", "name": "synthetic_band_07", "wavelength_nm": null}
    ]
  },
  "source": {
    "format": "npy",
    "dtype": "float32",
    "value_range": [0.0, 1.0]
  },
  "normalization": {
    "name": "identity_unit_interval",
    "version": 1,
    "parameters": {}
  },
  "nodata": {
    "policy": "reject",
    "non_finite": "reject",
    "values": []
  },
  "preprocessing": {
    "name": "phi2_tile_filter.utils.load_tile_numpy",
    "version": 2,
    "channel_policy": "exact-no-implicit-conversion",
    "tiff_policy": "reject-use-npy-or-mission-specific-loader"
  }
}
```

The `preprocessing_sha256` value fingerprints the normalization, nodata and preprocessing sections; the full `input_schema_sha256` additionally covers band order and all other contract fields.

The band names above are intentionally generic. They are **not** claimed to be PhiSat-2 spectral bands or wavelengths. A real dataset should replace them with validated sensor-specific identifiers, ordering, wavelengths, radiometric units/scaling, nodata semantics, and preprocessing metadata.

## File-format behaviour

For strict model execution, the source format must match the model's input schema. The synthetic model contract is `.npy` only.

The generic utility loader can read PNG or JPEG only when the channel count is already exact. It never converts RGB to RGBA, replicates grayscale into RGB, drops alpha, or otherwise invents/discards channels. Palette, CMYK and other modes require an explicit upstream conversion and corresponding contract.

TIFF and GeoTIFF are deliberately rejected by the generic loader. Scientific TIFF can carry high-bit-depth samples, multiple bands, georeferencing, nodata, scale/offset and other metadata that an 8-bit image conversion could silently destroy. Use validated `.npy` arrays under an explicit schema or add a mission-specific EO loader.

The requested calibration recall remains a threshold-selection target on the calibration sample. It is not presented as a population guarantee; the calibration artifact reports empirical recall and a one-sided exact Clopper-Pearson lower confidence bound.
