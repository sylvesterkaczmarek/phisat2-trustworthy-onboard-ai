# Assurance model

The repository treats the neural network, calibrated decision policy, preprocessing contract, validation evidence, deployment state, runtime telemetry, and retained data as one bounded decision system.

## Data lifecycle

The demonstration uses four independent data roles:

1. `train` fits model parameters only.
2. `calib` supplies static INT8 calibration data and selects the decision threshold, temperature, and input-quality guard.
3. `validation` is the only split used to accept or reject FP32-to-INT8 quantization and calibrated-policy behaviour.
4. `test` is reserved for final reporting after the candidate has passed calibration, validation, bundle verification, and promotion.

The synthetic generator creates the four splits from independent deterministic child RNG streams and records their roles in the dataset manifest. The final test split is not an acceptance gate.

The separate robustness benchmark is evaluation-only. Its nominal, degraded, corrupted, and OOD samples are generated after deployment and do not participate in training, calibration, validation, model selection, or promotion.

## Decision policy

A tile is requested for downlink when any of the following holds:

1. the event probability meets the calibrated event threshold;
2. maximum class confidence falls below the configured minimum confidence;
3. the calibrated input-quality guard marks the input outside its nominal operating region;
4. input observation, preprocessing, or inference fails.

Only a confidently classified background tile that also passes the input-quality guard is intentionally discarded. Runtime telemetry distinguishes this policy retention request from successful materialisation into the downlink output. If a fallback requests retention but the source can no longer be copied, that failure is recorded explicitly rather than counted as successful retention.

## Input-quality guard

Policy schema version 5 contains a lightweight input-quality guard calibrated on the nominal `calib` split. The guard computes per-band mean and standard deviation together with low/high saturation fractions and horizontal/vertical total variation. These features are standardized using calibration-set means and diagonal scales, then reduced to a root-mean-square standardized distance.

The threshold is a configurable calibration-score quantile multiplied by a configurable margin. The default is deliberately conservative and deterministic. The guard is architecture-independent and inexpensive, but it is not a universal OOD detector. It does not prove semantic novelty, sensor health, or physical distribution equivalence. It is intended to surface obvious radiometric, spectral, saturation, missing-band, and texture shifts that maximum softmax confidence can miss.

A guard trigger does not reject the tile. It causes `input_quality_fallback`, so the tile is requested for retention for later inspection or ground processing.

## Robustness benchmark

`robustness_benchmark.py` provides a deterministic stress benchmark in four categories:

- `nominal`: the existing square-event synthetic distribution;
- `degraded`: valid numeric inputs with noise, illumination shifts, per-band gain/offset drift, blur, cloud-like occlusion, spatial shifts, and spectral distribution shifts;
- `corrupted`: missing-band zero fill, non-finite corrupt bands, saturated regions, and dead-pixel/stripe patterns;
- `ood`: synthetic unknown checkerboard, sinusoidal, radial, and striped backgrounds with altered spectral structure.

The perturbations are simulation tools only. They are not calibrated models of PhiSat-2, any particular EO sensor, atmospheric radiative transfer, detector electronics, cloud microphysics, or spacecraft image formation. The benchmark therefore measures robustness of the software decision logic under controlled synthetic stress, not operational EO accuracy.

The benchmark runs the already deployed model and policy without retraining or recalibration. It reports each condition separately, including event retention, background rejection, retained fraction, fallback rate, input-quality guard trigger rate, preprocessing/inference failure rate, degradation/OOD detection rate, and source-file bytes retained/reduced.

Before reporting, the robustness summarizer requires every telemetry record to use the same deployment bundle, model, policy, input contract, schema file, preprocessing fingerprint, and telemetry schema version. It also verifies the benchmark manifest's input contract and re-hashes each benchmark source file against the recorded input SHA-256 and size. Mixed or changed evidence is rejected rather than combined.

## Source-byte and prevalence reporting

Byte accounting refers to bytes in the source tile files handled by the demonstrator. Metrics use names such as `source_bytes_total`, `source_bytes_retained`, and `source_bytes_reduction_fraction` or `_pct`.

These values are not spacecraft link bandwidth. The repository does not model packetisation, framing, forward-error correction, retransmission, compression outside the source file, protocol overhead, contact windows, adaptive coding/modulation, or other link-layer effects. Reports explicitly set `operational_link_bandwidth_measured` to false.

The robustness report can simulate configurable operational event prevalence using class-conditional retention observed on nominal synthetic samples. It reports `expected_retained_fraction` and `expected_source_bytes_reduction_fraction`. This is a prevalence-weighted source-byte estimate under stated assumptions, not a measured mission traffic or link-budget result.

## Runtime failure boundary

`OnnxRunner.evaluate_file()` keeps file observation inside the conservative failure boundary. Stat, SHA-256 hashing, preprocessing, input-quality assessment, ONNX execution, and policy failures produce a structured fallback record rather than unexpectedly terminating a batch where possible.

The runtime records the failure stage and preserves any metadata that was successfully observed. A stable corrupt input can therefore have a valid input SHA-256 while still reporting a preprocessing failure. A missing or unreadable file can report a null input hash and size. The runtime compares file identity metadata before and after hashing/preprocessing so a file that changes while being evaluated is not treated as a stable observation.

## Telemetry integrity

Runtime and downlink JSONL records use an explicit telemetry schema version. Current schema version 6 records include input-quality evidence and host timing evidence as well as:

- deployment bundle ID and whether the complete local bundle manifest/components were verified;
- ONNX model SHA-256;
- calibration-policy file SHA-256;
- semantic input/preprocessing contract SHA-256;
- exact input-schema file SHA-256;
- preprocessing fingerprint;
- per-input file SHA-256 and observed size;
- inference status, failure stage, policy decision, and retention request;
- input-quality guard method, score, threshold, and in/out-of-region decision when enabled;
- execution provider and separate observation, preprocessing, input-quality, ONNX, policy, and end-to-end host wall-clock timing.

Downlink records additionally state whether retention was actually materialised and the SHA-256 of the copied file. A copied file is rejected if its hash differs from the input hash observed during evaluation.

A runtime record only marks `deployment_bundle_verified` true after checking the complete local bundle manifest identity and all four component hashes, including validation evidence. The final summarizer validates every record against the telemetry schema, rejects duplicate file entries, and requires final-test and downlink logs to agree on file set, per-file input hash and size, bundle, model, policy, input schema, and preprocessing identity. The SHA-256 of the calibration-policy file supplied to the summarizer must equal the policy hash recorded in both logs. If an input could not be stably hashed, the runtime can still emit conservative failure telemetry, but the scientific summarizer refuses to claim fully reconciled final metrics for that run.

Telemetry schema version 6 is emitted by the current runtime. The validator retains read compatibility with versions 4 and 5 for archived runs; version 4 predates input-quality evidence and version 5 predates the explicit timing breakdown.

## Calibration uncertainty

The requested event recall is used to select a threshold on the calibration sample. It is not treated as a population-level guarantee.

The calibration artifact records positive/background sample counts, requested threshold-selection recall, achieved empirical recall, precision, ROC-AUC, and a one-sided exact Clopper-Pearson lower confidence bound. An optional minimum lower-bound requirement can reject a calibration before bundle construction.

The input-quality guard is also calibrated on this split. Its quantile/margin threshold should be interpreted as a deterministic demonstrator setting, not a statistically guaranteed OOD detection rate.

## Quantization validation

FP32 and INT8 acceptance is evaluated only on the validation split, using the policy selected independently on `calib`.

The validation report separates model classification metrics, quantization regressions, event-score drift, calibrated retention metrics, and FP32/INT8 agreement on the actual retain/discard decision. It also records how often the input-quality guard flags nominal validation samples.

Configurable gates cover accuracy drop, event-recall drop, false-negative-rate increase, PR-AUC drop, argmax agreement, retain/discard agreement, event-retention-recall drop, and event-score drift. Validation evidence records the SHA-256 of the exact calibration-policy artifact used to calculate policy behaviour. The bundle verifier recomputes the acceptance checks and refuses a validation report whose policy hash differs from the policy being bundled.

## Deployment bundles

A deployable candidate is an immutable directory containing `model.onnx`, `policy.json`, `input_schema.json`, `validation.json`, and `bundle.json`.

The policy file is hashed as a bundle component, so the calibrated input-quality guard and its threshold are cryptographically bound to the deployed model and input contract. Validation evidence is separately bound to the exact policy SHA-256, preventing accepted evidence from one threshold/confidence/quality-guard configuration from being reused with another policy for the same model.

Bundle creation validates either legacy policy schema 4 or current schema 5; schema 5 must contain a structurally valid quality guard. Bundle build output is subject to the same destructive-path safety rules as other tree-replacement workflows and cannot overlap the model, policy, schema, or validation inputs.

## Promotion and rollback

Validated bundles are copied into a content-addressed store under their `bundle_id`. Stored bundles are immutable. Deployment state is held in `deployment_state.json` with active/previous bundle IDs and a monotonically increasing generation number.

Deployment-state bundle identifiers must be valid SHA-256 hex strings and must exactly match the selected bundle manifest. Promotion verifies both the new candidate and the currently active stored bundle before changing the state pointer, so a corrupted current deployment is not silently preserved as the claimed rollback target. Rollback verifies both bundles before swapping complete identifiers, so model, policy, quality guard, preprocessing metadata, and validation evidence move together.

## Filesystem safety

Commands that replace directory trees validate their destinations before recursive mutation. The exact filesystem root, home directory, current working directory, and system temporary-directory root are rejected as recursive replacement targets. Input and output trees must be fully disjoint.

The synthetic generator, robustness benchmark, downlink filter, and deployment-bundle builder use guarded destinations; the generated data/downlink workflows build complete outputs in sibling staging directories and only replace destinations after successful work. Existing output is preserved when generation or processing fails before the final swap.

## Watchdog

The watchdog uses `subprocess.Popen` without `shell=True`. It supports bounded restart, optional wall-clock timeout, optional heartbeat-file timeout, graceful terminate-first behaviour, kill escalation, and structured per-attempt JSONL telemetry. This remains a process-level research pattern rather than spacecraft FDIR.

## Pipeline ordering

The main demonstration follows this order:

1. generate train, calibration, validation and final-test splits;
2. fit the PyTorch model on `train`;
3. export and validate FP32 ONNX;
4. quantize to INT8 using calibration data;
5. calibrate threshold, temperature, recall evidence, and input-quality guard on `calib`;
6. evaluate FP32/INT8 and policy regressions on `validation` only, binding the report to the exact policy artifact;
7. build and verify the deployment bundle;
8. promote the complete bundle;
9. emit final-test telemetry from the verified active bundle;
10. materialise policy-requested data using staged output;
11. reconcile final-test/downlink telemetry and report final metrics.

The optional robustness benchmark runs only after this deployment lifecycle and does not alter the active model or policy.

## Scope

These mechanisms improve software integrity, experimental hygiene, failure auditability, and robustness evaluation for the demonstrator. They do not provide flight qualification, formal safety guarantees, radiation tolerance, authenticated telemetry, power-loss-safe spacecraft storage, operational EO accuracy, physical sensor fidelity, validated OOD guarantees, or mission-level fault tolerance. A flight implementation would require representative mission data, sensor-specific degradation models, trusted provenance, hardware-in-the-loop testing, fault injection, resource limits, signed update/telemetry handling, and mission-specific acceptance criteria.
