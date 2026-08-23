# Assurance

Operational hooks for the demonstrator include a conservative decision policy, statistically explicit calibration evidence, validation-only quantization acceptance, structured telemetry, bounded process restart, and immutable deployment bundles that bind the ONNX model to its calibration policy, preprocessing metadata, and validation evidence. Promotion and rollback use an atomic deployment-state pointer so the model and policy move together.

See [`docs/assurance.md`](../docs/assurance.md) for the data lifecycle, statistical semantics, deployment behaviour, and limitations.
