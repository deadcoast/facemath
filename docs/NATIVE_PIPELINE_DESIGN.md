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

## Native portability repair

The next patch starts from the actual 0.11.0 archive, SHA-256
`bf181b9e8719b99f370bd1e0e17e1f4a187d053e397a7cce5413fb883791bfd4`.
Its hosted regression matrix passed all twelve OS/Python combinations. Native
Windows passed; Linux lacked `libEGL.so.1`, and macOS aborted in the native
constructor inside pytest. Those original outputs remain failure evidence.

Move live validation through the existing isolated worker and run a retained,
strict native execution gate before the full test suite. A fatal library error
must remain a failing gate with native stderr and return code, without aborting
the validation controller. Install the required Linux shared libraries explicitly.
Diagnose macOS from its isolated process output before selecting a remedy.
No numerical settings, labels, splits or evaluation outcomes change in this repair.

The inherited observer lock also installed two distributions sharing `cv2`:
`opencv-python==4.14.0.94` and `opencv-contrib-python==5.0.0.93`. Hosted native
Linux/macOS loaded 5.0.0 while the Windows pilot loaded 4.14.0. Replace both with
one `opencv-contrib-python>=4.9,<5` dependency. Native preflight and direct provider
construction require that single distribution and a matching loaded 4.x binding.
Environment records list all four OpenCV wheel variants so conflicts stay visible.
This fixes a demonstrated environment ambiguity; it does not establish the cause
of the macOS abort. The next hosted jobs were refused before execution because of
an account billing/spending-limit condition; retain that scheduler evidence too.
