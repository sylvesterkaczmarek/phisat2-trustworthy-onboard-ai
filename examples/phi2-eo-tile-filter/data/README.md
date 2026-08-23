# Data

The main demo generates deterministic synthetic tiles into four independent lifecycle splits:

- `train` fits model parameters only.
- `calib` performs INT8, threshold, temperature, and input-quality-guard calibration.
- `validation` controls model and quantization acceptance.
- `test` is reserved for final reporting after the candidate bundle is accepted and promoted.

Independent child RNG streams are recorded in `manifest.json`, and tests check that generated tile contents do not overlap across lifecycle splits.

## Input contract

Every generated dataset contains `input_schema.json`. Its canonical SHA-256 is propagated through the dataset manifest, training checkpoint, FP32/INT8 ONNX metadata, calibration policy, validation evidence, telemetry, and deployment bundle.

The schema can record ordered band IDs/names, optional wavelength metadata, source/model layouts, dimensions, dtypes, radiometric range, normalization, nodata policy, preprocessing version, resize behaviour, and channel policy.

The synthetic dataset uses float32 HWC `.npy` tiles already scaled to `[0, 1]`, identity normalization, and reject-nodata semantics. The model receives float32 NCHW tensors.

A seven-band synthetic contract uses generic bands such as:

```json
{
  "schema_version": 2,
  "contract_type": "eo-input-preprocessing",
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
  "source": {"format": "npy", "dtype": "float32", "value_range": [0.0, 1.0]},
  "normalization": {"name": "identity_unit_interval", "version": 1, "parameters": {}},
  "nodata": {"policy": "reject", "non_finite": "reject", "values": []}
}
```

These names are intentionally generic. They are not PhiSat-2 band definitions or validated sensor wavelengths.

## Robustness benchmark data

The optional robustness benchmark is a separate post-deployment dataset. It does not enter training, calibration, validation, or model acceptance.

It creates four deterministic categories:

- `nominal`: the same lightweight square-event distribution used by the demo;
- `degraded`: sensor-noise-like perturbations, illumination shifts, per-band gain/offset drift, blur, cloud-like occlusion, spatial shift, and spectral distribution shift;
- `corrupted`: missing-band zero fill, non-finite corrupt bands, saturated regions, and dead-pixel/stripe patterns;
- `ood`: unknown checkerboard, sinusoidal, radial, and striped backgrounds with altered spectral structure.

The benchmark writes `benchmark_manifest.json` with every sample's category and perturbation recipe. All random generation is seed-controlled.

These perturbations are deliberately synthetic stressors. They are **not** physically calibrated models of PhiSat-2 or any other EO sensor, atmosphere, detector, optics, compression chain, or cloud process. Their purpose is to test whether the decision and fallback logic behaves conservatively when input statistics move away from the nominal synthetic distribution.

Example after running the main demo:

```bash
python scripts/run_robustness_benchmark.py \
  --output-root /tmp/phi2-7band \
  --samples-per-category 20 \
  --seed 101 \
  --event-prevalences 0.01,0.05,0.10
```

Perturbation magnitudes are configurable through command-line options so experiments can be repeated at fixed stress levels.

## File-format behaviour

For strict model execution, source format must match the model input schema. The synthetic model contract is `.npy`.

PNG/JPEG are accepted by the generic utility loader only with an already exact channel structure. No implicit channel manufacturing or dropping is performed. TIFF/GeoTIFF are deliberately rejected because scientific TIFF can carry high-bit-depth, multiband, geospatial, scale/offset, and nodata semantics requiring an EO-aware ingest path.

## Interpretation

The calibrated event-recall target is not a population guarantee; the policy records finite-sample Clopper-Pearson evidence. Likewise, the calibrated input-quality guard is a lightweight statistical-distance heuristic rather than a guaranteed OOD detector.

Robustness reports describe source-file byte retention/reduction. They do not measure spacecraft link bandwidth or account for packetisation, coding, retransmission, framing, or contact geometry.
