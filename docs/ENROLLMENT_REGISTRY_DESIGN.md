# Cross-study enrollment registry

## Review

The actual parent is `packages/facemath-0.15.0-source-window-20261003.zip`,
SHA-256 `d70ccf98be02e28494e55162393fc3df6eb7eecbac2dddfe5c96e8f332bf7fc2`.
The extracted-parent baseline passed 1,096 tests with one Windows directory-symlink
skip in 268.48 seconds. Its raw output is retained separately before source
changes. The previous root handoff remains historical evidence.

Source windows separate positions in one pinned universe. The donor capsule
guards its own complete enrollment, but neither mechanism registers unrelated
studies. NEXT-02 therefore supplies prospective source reuse checks across
explicitly registered study versions. NEXT-01 independent anatomical semantics
and visibility remains open; no registry substitutes for independent labels.

## Design

Export a verified frozen study as an immutable enrollment receipt containing the
exact study JSON, all reference JSON and their original manifest pins. Include
every split and every excluded case. Acquisition outcomes are unnecessary: an
enrolled source remains registered whether observation succeeds, fails or has
never run. The receipt preserves complete enrollment instead of a curator-picked
list of successful rows. Portable verification checks the manifest bindings and
rederives aliases; hashes establish consistency, not external authenticity.

Build a create-only registry with an explicit human-readable scope and a fixed
alias policy. An optional prior snapshot is retained in full and cannot lose
entries or change scope. Additions are deterministically ordered; exact duplicate
studies are rejected. Distinct studies may enroll the same source, and every
matching prior case remains visible. The scope covers supplied exports only;
completeness across all research or native-model training is not asserted.

Compare all image hashes in one namespace: stored-raster crop hash, originally
encoded input hash, and recorded original encoded/raster source hashes. Compare
group and lineage IDs in a second shared namespace, including cross-role aliases.
IDs are exact, case-sensitive declarations, not inferred person identities.
Curators must use stable source-qualified IDs; generic reused IDs conservatively
collide. Undeclared aliases, edited images, different person photographs and
unregistered studies cannot be inferred by this policy.

An explicit registry argument to study freezing produces a v16 study and manifest
with an embedded, pinned registry file and a recomputable full-denominator check.
Reject the entire prospective cohort if ANY enrolled case overlaps, including
prospectively excluded targets. Never automatically filter, backfill, change a
rank window or inspect outcomes. A read-only check reports every matching source.
Loading the lock, acquiring, verifying and replaying must recompute the check.
Registry-bound fitted-model exports also retain the exact registry bytes in a
v16 capsule and enforce its exclusions on transfer. Qualification snapshots keep
the registry and reject prior sources in complete query/context groups.
Explicit registry and prior-snapshot inputs must validate as registry objects:
JSON `null`, empty option paths and malformed values cannot disable the guard.
Legacy v5 studies without registry retain their existing semantics. Older readers
must reject v16 rather than silently omit its registry check.

Expose export/build/check CLI commands and registry arguments on study-freeze and
both native dataset pipelines. Do not change fitted centers, scatter, radii,
observations, coordinate conversion, empirical admission or qualification policy.

## Plan and validation

1. Verify/extract the actual parent and run its complete inherited suite.
2. Implement receipts, immutable snapshots, complete alias checks, frozen study
   binding and pipeline/CLI integration. Test all splits, exclusions, recorded
   acquisition failures, source aliases, cross-role IDs, format refusal,
   resealed tampering, append preservation and zero observer calls on rejection.
3. Export retained real LFPW, HELEN, IBUG, pilot and HELEN holdout enrollment;
   explicitly record the scope. Reject a previously enrolled holdout. Freeze a
   small next HELEN window before a distinct native run with the unchanged model.
   This is registry integration evidence, not a new model or utility acceptance.
4. Run full regression, milestone-lab, verify/replay, isolated installed-wheel
   checks, final extracted-archive validation and observed-array/model invariance.
   Retain raw commands, environment, failures, skips and package bytes.
5. Update the root handoff with completed implementation and remaining independent
   reference, sampling, utility, stratified-transfer and platform gates.

## Recorded integration

The prospective snapshot contains 243 cases from five retained source plans.
The prior 60-source HELEN holdout was rejected before any observer session was
created. The new window is fixed at 12 roots, `[120,132)`, with zero registered
aliases. The first Windows and Ubuntu executions observed all 12 and the unchanged
model covered all 12 recorded reference groups. This small integration was not
designed to authorize scientific acceptance; no utility threshold or fit changed.

`next-registry.json` is a separately declared, broader project-scope root that
retains all five earlier exports and the new full 12-case export: 255 enrolled
cases. Its construction record checks every prior row is retained. The original
five-plan scope and prospective snapshot are unchanged. Further snapshots can
extend the broader scope with `--prior`; scope changes themselves require an
explicit new root, not an alteration inside an immutable extension.

Completed platform/package results and observed-array/model-byte comparisons are
recorded in `evidence/enrollment-registry-20261003/VALIDATION.json`. Repeated
platform executions do not add independent source enrollment. The historical
mathematics audit, failed acquisitions, reference limits and external gates remain
retained separately.

Final review reproduced an initial explicit-JSON-null bypass before correcting
the input boundary. Its unguarded synthetic lock, command result and pre-fix code
are retained as superseded diagnostics. Final requalification uses the identical
12-source plan; those outcomes are already known and are not new unseen enrollment.
`prospective-freeze-03.json` records the final fingerprint and this distinction.
