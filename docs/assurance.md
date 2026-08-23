# Assurance model

The repository treats the neural network, calibrated decision policy, preprocessing contract, validation evidence, and deployment state as one bounded decision system.

## Decision policy

A tile is kept for downlink when any of the following holds:

1. the event probability meets the calibrated event threshold;
2. maximum class confidence falls below the configured minimum confidence;
3. preprocessing or inference fails.

Only a confidently classified background tile is discarded. This makes the fallback conservative with respect to science-data loss.

## Deployment bundles

A deployable candidate is an immutable directory containing:

- `model.onnx`
- `policy.json`
- `input_schema.json`
- `validation.json`
- `bundle.json`

`bundle.json` has an explicit schema and bundle format version, SHA-256 hashes for every component, the model SHA-256, and a deterministic `bundle_id` derived from the manifest contents.

Bundle creation verifies the ONNX structure, checks that the calibration policy belongs to the exact model and input shape, checks that the validation report covers the exact INT8 model, re-evaluates the recorded FP32/INT8 acceptance thresholds, and validates the generated preprocessing/input metadata. A candidate that fails any of these checks is not promotable.

## Promotion and rollback

Validated bundles are copied into a content-addressed store under their `bundle_id`. Stored bundles are immutable. Deployment state is held in a small `deployment_state.json` pointer with `active_bundle_id`, `previous_bundle_id`, and a monotonically increasing generation number.

Promotion writes the candidate bundle completely and verifies it before atomically replacing the deployment-state pointer. If execution stops before the state replacement, the previous active bundle remains selected. A fully written but unreferenced bundle may remain in the store and is harmless.

Rollback atomically swaps the active and previous bundle identifiers. Because the policy, preprocessing metadata, validation evidence, and model live in the same immutable bundle, rollback cannot intentionally select an old model with a new calibration policy.

The active bundle is resolved and re-verified before use by the demonstration pipeline.

## Pipeline ordering

The demonstration now follows this order:

1. train the PyTorch model;
2. export and validate FP32 ONNX;
3. quantize to INT8;
4. run FP32/INT8 acceptance checks;
5. calibrate the decision policy against the exact INT8 model;
6. build and verify the deployment bundle;
7. promote the complete bundle;
8. run telemetry and downlink filtering from the resolved active bundle.

The active deployment therefore does not change if validation, calibration, or bundle construction fails.

## Scope

These mechanisms improve software integrity and reproducibility for the demonstrator. They do not provide flight qualification, formal safety guarantees, radiation tolerance, authenticated update distribution, power-loss-safe spacecraft storage semantics, or mission-level fault tolerance. A flight implementation would need platform-specific persistent-storage guarantees, signed update handling, hardware-in-the-loop testing, fault injection, resource limits, and mission-specific acceptance criteria.
