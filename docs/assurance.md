# Assurance model

The repository treats the neural network, calibrated decision policy, preprocessing contract, validation evidence, and deployment state as one bounded decision system.

## Data lifecycle

The demonstration uses four independent data roles:

1. `train` fits model parameters only.
2. `calib` supplies static INT8 calibration data and selects the decision threshold and temperature.
3. `validation` is the only split used to accept or reject FP32-to-INT8 quantization and calibrated-policy behaviour.
4. `test` is reserved for final reporting after the candidate has passed calibration, validation, bundle verification, and promotion.

The synthetic generator creates the four splits from independent deterministic child RNG streams and records their roles in the dataset manifest. The final test split is not an acceptance gate.

## Decision policy

A tile is kept for downlink when any of the following holds:

1. the event probability meets the calibrated event threshold;
2. maximum class confidence falls below the configured minimum confidence;
3. preprocessing or inference fails.

Only a confidently classified background tile is discarded. This makes the fallback conservative with respect to science-data loss.

## Calibration uncertainty

The requested event recall is used to select a threshold on the calibration sample. It is not treated as a population-level guarantee.

The calibration artifact records:

- positive and background calibration sample counts;
- the requested threshold-selection recall;
- achieved empirical event recall;
- event precision at the selected threshold;
- ROC-AUC;
- a one-sided exact Clopper-Pearson lower confidence bound for event recall;
- the confidence level used for that bound.

An optional minimum lower-bound requirement can reject a calibration before bundle construction. This is a statistical sampling criterion for the demonstrator. It does not address dataset shift, sensor shift, dependence between samples, or mission-specific hazard analysis.

## Quantization validation

FP32 and INT8 acceptance is evaluated only on the validation split, using the calibration policy selected independently on `calib`.

The validation report separates:

- model classification metrics for FP32 and INT8;
- quantization regressions in accuracy, event recall, false-negative rate, F1, ROC-AUC and PR-AUC;
- event-score drift;
- calibrated downlink-policy metrics such as event retention and background rejection;
- FP32/INT8 agreement on the actual retain/discard decision.

Configurable gates cover accuracy drop, event-recall drop, false-negative-rate increase, PR-AUC drop, argmax agreement, retain/discard agreement, event-retention-recall drop, and event-score drift. The bundle verifier recomputes those acceptance checks from the report's recorded metrics and criteria before accepting the report as deployment evidence.

## Deployment bundles

A deployable candidate is an immutable directory containing:

- `model.onnx`
- `policy.json`
- `input_schema.json`
- `validation.json`
- `bundle.json`

`bundle.json` has an explicit schema and bundle format version, SHA-256 hashes for every component, the model SHA-256, and a deterministic `bundle_id` derived from the manifest contents.

Bundle creation verifies the ONNX structure, checks that the calibration policy belongs to the exact model and input shape, checks that accepted statistical calibration metadata is present, verifies that the validation report covers the exact INT8 model and explicitly represents the validation split, re-evaluates the recorded scientific acceptance thresholds, and validates the generated preprocessing/input metadata.

## Promotion and rollback

Validated bundles are copied into a content-addressed store under their `bundle_id`. Stored bundles are immutable. Deployment state is held in a small `deployment_state.json` pointer with `active_bundle_id`, `previous_bundle_id`, and a monotonically increasing generation number.

Promotion writes the candidate bundle completely and verifies it before atomically replacing the deployment-state pointer. If execution stops before the state replacement, the previous active bundle remains selected. A fully written but unreferenced bundle may remain in the store and is harmless.

Rollback atomically swaps the active and previous bundle identifiers. Because the policy, preprocessing metadata, validation evidence, and model live in the same immutable bundle, rollback cannot intentionally select an old model with a new calibration policy.

The active bundle is resolved and re-verified before use by the demonstration pipeline.

## Pipeline ordering

The demonstration follows this order:

1. generate independent train, calibration, validation and final-test splits;
2. fit the PyTorch model on `train`;
3. export and validate FP32 ONNX;
4. quantize to INT8 using calibration data;
5. calibrate threshold, temperature, and recall-bound evidence on `calib`;
6. evaluate FP32/INT8 and policy regressions on `validation` only;
7. build and verify the deployment bundle;
8. promote the complete bundle;
9. evaluate final classification and downlink retention on `test` only.

The active deployment therefore does not change if calibration, validation, or bundle construction fails, and final test results cannot influence model acceptance in the same run.

## Scope

These mechanisms improve software integrity, experimental hygiene, and statistical reporting for the demonstrator. Clopper-Pearson bounds quantify finite-sample binomial uncertainty under their assumptions; they do not establish performance under operational distribution shift. The repository does not provide flight qualification, formal safety guarantees, radiation tolerance, authenticated update distribution, power-loss-safe spacecraft storage semantics, or mission-level fault tolerance. A flight implementation would need representative mission data, platform-specific persistent-storage guarantees, signed update handling, hardware-in-the-loop testing, fault injection, resource limits, and mission-specific acceptance criteria.
