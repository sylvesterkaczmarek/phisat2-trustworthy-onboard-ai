# Assurance

Operational hooks for the demonstrator include conservative runtime fallback, statistically explicit calibration evidence, validation-only quantization acceptance, deployment-bound telemetry, staged filesystem replacement, bounded process restart with hang detection, and immutable deployment bundles that bind the ONNX model to its calibration policy, preprocessing metadata, and validation evidence.

The final summarizer reconciles bundle, model, policy, preprocessing, and per-input hashes before producing final metrics. The watchdog supports wall-clock timeout, optional heartbeat timeout, graceful termination, kill escalation, and structured restart telemetry.

See [`docs/assurance.md`](../docs/assurance.md) for the data lifecycle, runtime failure semantics, telemetry checks, filesystem safeguards, deployment behaviour, and limitations.
