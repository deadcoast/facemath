# Native datasets and machine-reference pipelines

The 2026-10-02 direction authorizes externally sourced images, automated testing,
and automated release decisions. Work does not depend on a new human annotation
campaign. The instrument still distinguishes numerical reproducibility,
agreement with machine labels, and anatomical reference accuracy.

This milestone builds on the actual 0.10.1 archive, SHA-256
`1aad162e54dacbdc0b66902697b30e36ef2d1584d137f178deff5251d8bf61ff`.
The inherited byte inventory and tests are executed before source changes.

## Dataset boundary

A pinned manifest declares local asset bytes, upstream URLs/revision, label
origin, coordinate origin, source-image groups, lineages and explicit splits.
The first adapter reads dlib's 68-point XML format. It requires every image and
every annotated face to be enrolled. No detector selects the sample. A fixed
box margin produces integer raster crops without resizing; the reference8
coordinates undergo only the recorded integer translation. All crops of one
source share its group and lineage. Insufficient reference support is recorded
as a prospective exclusion before observation, with its reason.

The public pilot uses the dlib example PASCAL VOC photographs and their published
shape-predictor outputs. These are `model_generated` references. Their absent
visibility labels and uncertain anatomical correspondence remain explicit.
Source-image grouping is not a claim of verified person-level independence.
The pilot is a pipeline/consistency benchmark, not a population accuracy study.

## Ownership and transfer

Dataset preparation verifies every pin and retains original encoded images,
annotation XML, the input manifest, conversion records and source-bound labels.
The dataset is immutable and independently verifiable. A pipeline run copies
that complete dataset, then invokes the existing study freezer, NativeExecutor,
finalizer and exact numerical replay. It does not add a geometric alignment,
another fitting implementation, or an alternative observation contract.

Development fits the existing joint error model, calibration chooses its radius,
and evaluation consumes that frozen model. Tiny calibration inventories may
produce an unbounded radius; they are reported without changing alpha or splits.
Machine references can train research models but cannot satisfy the independent
reference gate for empirical admission. Measured coordinates remain unchanged.

The transferable unit is the complete pipeline directory, including original
dataset assets, frozen protocol, native stdout/stderr and receipts, fitted model,
numerical arrays and replay comparison. Successful runs receive a manifest.
Failures retain their actual partial directory and journal without a success
manifest. Repeated native execution uses a new run; replay recomputes retained
observations and does not assert bitwise determinism of a fresh native process.

## Validation

Tests cover incorrect pins, malformed/duplicate parts, unsafe paths, source and
lineage split leakage, exact crop translation, unchanged pixels, missing support,
tampering after transfer, explicit machine-reference provenance and refusal of
machine labels by empirical admission. The real pilot executes the pinned model.
Source, installed-wheel and extracted-archive checks retain raw command records.
The inherited milestone checks continue to test observed-state invariance.
