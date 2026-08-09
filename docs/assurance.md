# Assurance model

The repository treats the model as one component in a bounded decision pipeline.

## Decision policy

A tile is kept for downlink when any of the following holds:

1. the event probability meets the calibrated event threshold;
2. maximum class confidence falls below the configured minimum confidence;
3. preprocessing or inference fails.

Only a confidently classified background tile is discarded. This makes the fallback conservative with respect to science-data loss.

## Binding calibration to a model

`calibration.json` stores the SHA-256 of the model used to derive the event threshold. Runtime filtering refuses to use the policy with a different model or input shape.

## Deployment checks

Before model promotion, the workflow checks:

- PyTorch versus FP32 ONNX numerical agreement;
- ONNX structural validity;
- QDQ INT8 model validity;
- held-out FP32 versus INT8 accuracy drop and prediction agreement;
- model SHA-256 identity.

The known-good model helper uses atomic copies and keeps a previous artifact for rollback.

## Scope

The watchdog, rollback helper, telemetry, and threshold policy are software assurance demonstrations. They are not flight-qualified safety mechanisms and do not establish mission safety by themselves.
