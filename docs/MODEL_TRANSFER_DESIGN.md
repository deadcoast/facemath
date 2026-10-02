# Frozen model transfer

FACEMATH 0.14 evaluates an already fitted pooled joint error model against a new
prospectively frozen cohort. The unit remains an image-bound Observation and the
existing reference8 evaluator. No new alignment, fit, radius selection or query
admission occurs in transfer.

## Review and scope

The 0.13 HELEN study retains a complete development center, full16 scatter and
calibration radius. Its evaluation outcomes cannot select a replacement model.
The next roadmap gate is reuse on new data with explicit domain-shift limits.
The inherited actual extracted-package baseline is recorded under
`evidence/model-transfer-20261002/development/parent-baseline.json`; reuse of that
completed test execution is distinguished from a new run.

## Data contract

An exported model capsule retains the exact serialized joint calibration, donor
study lock, donor reference source identifiers and qualification manifest/report
digests. Export first verifies the complete donor bundle. Import checks the
model's center/scatter/radius/protocol bindings and its donor development and
calibration enrollment. The whole donor enrollment, including excluded and
evaluation cases, is forbidden in the target by decoded/encoded raster hashes,
declared group/lineage identifiers and retained original-source hashes.
Hashes establish consistency, not authenticity or person independence.

The target uses evaluation cases only, a matching observer recipe and reference
protocol, and the donor minimum-rater/alpha contract. Its named acquisition
domain remains its own: even an identical name does not establish exchangeability.
The capsule is embedded in the study lock before the first native attempt.
Ordinary qualification rejects these locks. Existing single-attempt acquisition,
failure receipts and stored-raster reference conversion are reused unchanged.

Finalization uses the existing raw evaluator without calibration rows, then
`joint_scored_rows`, `evaluate_coverage` and the shared relation-graph utility
report with the imported model. Reports retain every enrolled case, separate
prospective exclusions, count failed eligible cases in reference denominators,
and report complete-enrollment group delivery. No transfer report grants an
empirical query gate or claims donor nominal coverage in the target domain.

## Plan and validation

1. Export/import a typed model capsule; enforce full donor enrollment leakage.
2. Freeze, acquire, finalize, verify and replay evaluation-only studies with the
   existing runner and raw-coordinate evaluator. Add a dataset pipeline route.
3. Freeze a hash-selected IBUG original-image cohort before inference, keeping
   the existing HELEN fit intact. Retain published PTS provenance, all failures,
   no retry/replacement, unknown visibility/identity/native-training overlap.
4. Test fit prohibition, leakage across all splits/exclusions/source aliases,
   immutable model/observations, missingness, tampering and exact replay.
5. Run source/wheel/extracted-archive regression, milestone controls and actual
   native transfer. Preserve commands, environments, failures and raw logs;
   verify bytes, package, push and prove local/remote synchronization.

The initial supported model is the pooled full16 joint calibration. Stratified
models need a separate domain/routing transfer contract and are rejected rather
than flattened. New human annotation or acceptance is not required to execute
this research pipeline; published-label agreement is not anatomical truth.
