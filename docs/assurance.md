# Assurance model

The repository treats the neural network, calibrated decision policy, preprocessing contract, validation evidence, deployment state, runtime telemetry, and retained data as one bounded decision system.

## Data lifecycle

The demonstration uses four independent data roles:

1. `train` fits model parameters only.
2. `calib` supplies static INT8 calibration data and selects the decision threshold and temperature.
3. `validation` is the only split used to accept or reject FP32-to-INT8 quantization and calibrated-policy behaviour.
4. `test` is reserved for final reporting after the candidate has passed calibration, validation, bundle verification, and promotion.

The synthetic generator creates the four splits from independent deterministic child RNG streams and records their roles in the dataset manifest. The final test split is not an acceptance gate.

## Decision policy

A tile is requested for downlink when any of the following holds:

1. the event probability meets the calibrated event threshold;
2. maximum class confidence falls below the configured minimum confidence;
3. input observation, preprocessing, or inference fails.

Only a confidently classified background tile is intentionally discarded. Runtime telemetry distinguishes this policy retention request from successful materialisation into the downlink output. If a fallback requests retention but the source can no longer be copied, that failure is recorded explicitly rather than counted as successful retention.

## Runtime failure boundary

`OnnxRunner.evaluate_file()` keeps file observation inside the conservative failure boundary. Stat, SHA-256 hashing, preprocessing, ONNX execution, and policy failures produce a structured fallback record rather than unexpectedly terminating a batch where possible.

The runtime records the failure stage and preserves any metadata that was successfully observed. A stable corrupt input can therefore have a valid input SHA-256 while still reporting a preprocessing failure. A missing or unreadable file can report a null input hash and size. The runtime compares file identity metadata before and after hashing/preprocessing so a file that changes while being evaluated is not treated as a stable observation.

## Telemetry integrity

Runtime and downlink JSONL records use an explicit telemetry schema version. Each record carries:

- deployment bundle ID and whether the local bundle manifest was verified;
- ONNX model SHA-256;
- calibration-policy file SHA-256;
- semantic input/preprocessing contract SHA-256;
- exact input-schema file SHA-256;
- preprocessing fingerprint;
- per-input file SHA-256 and observed size;
- inference status, failure stage, policy decision, and retention request.

Downlink records additionally state whether retention was actually materialised and the SHA-256 of the copied file. A copied file is rejected if its hash differs from the input hash observed during evaluation.

The final summarizer validates every record against the telemetry schema, rejects duplicate file entries, and requires final-test and downlink logs to agree on file set, per-file input hash and size, bundle, model, policy, input schema, and preprocessing identity. The SHA-256 of the calibration-policy file supplied to the summarizer must equal the policy hash recorded in both logs. If an input could not be stably hashed, the runtime can still emit conservative failure telemetry, but the scientific summarizer refuses to claim fully reconciled final metrics for that run.

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

Bundle creation verifies the ONNX structure, checks that the calibration policy belongs to the exact model and input shape, checks that accepted statistical calibration metadata is present, verifies that the validation report covers the exact INT8 model and explicitly represents the validation split, re-evaluates the recorded scientific acceptance thresholds, and validates the preprocessing/input contract.

## Promotion and rollback

Validated bundles are copied into a content-addressed store under their `bundle_id`. Stored bundles are immutable. Deployment state is held in a small `deployment_state.json` pointer with `active_bundle_id`, `previous_bundle_id`, and a monotonically increasing generation number.

Promotion writes the candidate bundle completely and verifies it before atomically replacing the deployment-state pointer. If execution stops before the state replacement, the previous active bundle remains selected. A fully written but unreferenced bundle may remain in the store and is harmless.

Rollback atomically swaps the active and previous bundle identifiers. Because the policy, preprocessing metadata, validation evidence, and model live in the same immutable bundle, rollback cannot intentionally select an old model with a new calibration policy.

The active bundle is resolved and re-verified before use by the demonstration pipeline.

## Filesystem safety

Commands that replace directory trees validate their destinations before recursive mutation. The exact filesystem root, home directory, current working directory, and system temporary-directory root are rejected as recursive replacement targets. Input and output trees must be fully disjoint, so equal paths and ancestor/descendant relationships are refused.

The synthetic generator builds a complete dataset in a sibling staging directory and only replaces the destination after generation succeeds. The downlink filter likewise writes retained data and JSONL telemetry to staging paths first. If processing fails before the swap, an existing destination tree is left intact. Directory replacement uses sibling renames and restores the previous tree if the staged rename itself fails.

## Watchdog

The watchdog uses `subprocess.Popen` without `shell=True`. In addition to bounded restart on non-zero exit, it supports an optional wall-clock timeout and an optional heartbeat-file timeout.

On timeout it first requests graceful termination, waits for the configured grace period, and escalates to a kill only when necessary. Each attempt can be written as structured JSONL telemetry containing outcome, child return code, watchdog return code, timeout configuration, termination action, heartbeat updates, whether another restart is scheduled, and the restart reason.

Heartbeat monitoring is intentionally simple. When configured, the child process or wrapper is expected to update the heartbeat file periodically. This is a process-level research pattern, not spacecraft FDIR.

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
9. emit final-test telemetry from the verified active bundle;
10. materialise policy-requested downlink data using staged output;
11. reconcile the final-test and downlink logs before reporting final metrics.

The active deployment therefore does not change if calibration, validation, or bundle construction fails, and final test results cannot influence model acceptance in the same run.

## Scope

These mechanisms improve software integrity, experimental hygiene, failure auditability, and statistical reporting for the demonstrator. They do not provide flight qualification, formal safety guarantees, radiation tolerance, authenticated telemetry, Byzantine-tamper resistance, power-loss-safe spacecraft storage semantics, or mission-level fault tolerance. Clopper-Pearson bounds quantify finite-sample binomial uncertainty under their assumptions and do not establish performance under operational distribution shift. A flight implementation would need representative mission data, platform-specific persistent-storage guarantees, signed update and telemetry handling, hardware-in-the-loop testing, fault injection, resource limits, and mission-specific acceptance criteria.
