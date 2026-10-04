# FACEMATH · reproducible facial geometry

Version **0.21.0** compares image edges around the eyes, mouth and jaw. Both
rasters are sampled on identical contour-normal rays in the existing canonical
alignment. `analyze` returns `comparison.edge_comparison`; `overview` and the
report show signed B-minus-A displacement, RMS, 90th absolute displacement and
paired support. A-anchored and B-anchored rays remain separate inventories.
Two annotated image views locate the accepted edge candidates and their measured
normal displacements, using the report's existing circle/cross/line legend.

Weak, ambiguous, truncated and source-padding rays retain their rejection reasons
and denominators. Missing edges are unavailable, never zero mismatch. Exported
response arrays and masks support inspection and replay. The within-tolerance
range accounts for unresolved rays; it is not a resemblance probability or
confidence interval. These image gradients can reflect texture or lighting and
are not independently labeled anatomy. See [the paired-edge contract](docs/EDGE_COMPARISON_DESIGN.md).

[Open the paired image-edge report](evidence/edge-comparison-20261004/pair-02/calibrated/report.html)
or [inspect its component data](evidence/edge-comparison-20261004/pair-02/edge-audit.json).
This reuses the first fixed native pair from the core-feature pass, without a
new cohort or fit. Its eight rows retain 5–15 paired supported rays out of 32;
the remaining rays stay unresolved. All 158 inherited observed arrays are
unchanged, and calibration preserves all 358 current observed arrays.

Version **0.20.0** repairs reference-coordinate input. Workbench clicks, markers,
zoom anchors and the magnifier now share a content-box transform with explicit
pixel-center coordinates. First/last pixels remain selectable at fractional
display scales. Alt-arrows provide keyboard panning alongside Alt-drag.

New drafts carry the capture version, image, protocol, reference kind and rater.
Restore is an explicit in-page action; legacy drafts remain in their original
storage namespace and are never silently shifted. Prepared synthetic workspaces
now emit synthetic references that their own importer accepts. Native workspaces
continue to request declared blind references. The unchanged reference format
retains capture metadata through import and study freeze.

The inherited browser exported up to 0.535 pixels of error on an indexed raster.
The repaired six-case browser matrix exported the exact expected coordinates at
0.001-pixel resolution. These are automated UI controls, not human annotations
or a claim of anatomical accuracy. See [the input repair contract](docs/ANNOTATION_COORDINATES_DESIGN.md).

`overview` also handles redirected Windows text encodings: readable ASCII
decoration replaces unsupported glyphs, and unrepresentable labels use reversible
escapes. Numeric results stay unchanged; UTF-8 output keeps the existing layout.

Version **0.19.0** adds core regional comparison functions for the eyes, nose,
mouth, jaw, central face and complete landmark set. Each region now separates
its measured B-minus-A shift from contour change after translation. The existing
triangle maps are decomposed into rotation, area expansion/compression and
anisotropic stretch/shear. No additional alignment is fitted.

`analyze` returns these values under `comparison.regional_comparison`; `overview`
prints the regional shifts, centered RMS, area ratios, stretch and support counts.
The existing report's measurement ledger includes the component measurements.
The full polar factors, logarithmic stretch tensors and support masks are exported
in `arrays.npz`. These describe observed projected geometry; the reference8
localization model does not supply confidence bounds for the full mesh.

A failed optional holdout registration now leaves the primary comparison usable,
with its own unavailable result and retained failure reason. Failed sensitivity
refits keep their requested slots and withhold intervals instead of dropping
failed draws. See [the core-function contract](docs/REGIONAL_COMPARISON_DESIGN.md).

[Native core-function comparison](evidence/core-features-20261004/pair-01/calibrated/report.html)
and [component data](evidence/core-features-20261004/pair-01/regional-audit.json)
use the first two sources in the frozen HELEN `[212,232)` evaluation window.
The production run observed 20/20 sources with the existing model unchanged.
For this pair, 898 supported triangle maps were decomposed and 158 observed-state
arrays remained unchanged when adding calibration. The next enrollment registry
is `evidence/core-features-20261004/next-registry.json` (355 cases).

Version **0.18.0** returns fourteen named structural measurements in the normal
image-pair analysis: eye widths and spacing, mouth width, eye/nose/mouth/chin
heights, lateral offsets, mouth-corner height difference and eye-width asymmetry.
The CLI JSON includes `comparison.structural_comparison`; `overview` and the
report show A, B, signed B-minus-A differences and available reference bounds.
Each measurement uses the existing observations and shared error graph.

```shell
python -m facemath analyze portrait-a.png portrait-b.png --model models/face_landmarker.task --out runs/my-comparison
python -m facemath overview runs/my-comparison
```

Open `runs/my-comparison/report.html` to inspect the named rows and image overlay.
Widths and heights use each image's eye-midpoint span; pose and expression affect
them. Without an attached research calibration, observed values are available
and reference bounds explicitly remain unavailable. Missing points retain their
rows. These are projected geometric measurements, not an identity score.

Published output directories now inherit the chosen parent's access permissions,
fixing Windows results that the execution account could validate but the user
could not read. Frozen transfer also carries decoded image dimensions into the
inherited enrollment-registry check. See [the implementation contract](docs/STRUCTURAL_COMPARISON_DESIGN.md).

[Open the native comparison](evidence/structural-comparison-20261004/pair-01/calibrated/report.html)
or [read its values](evidence/structural-comparison-20261004/pair-01/comparison.json).
Both are from the first two entries of the new frozen evaluation window, with
all 478 vertices, flow, edges and 96 sensitivity draws. Calibration preserves
146 observed-state arrays. The measured rows are available; all fourteen
reference comparisons remain unresolved under the broad frozen error model.

The new HELEN `[192,212)` window observed 20/20 sources without refitting. Against
published PTS labels, mouth width meets the existing 0.05 eye-span tolerance in
16/20 sources and eye-to-nose height in 14/20. The complete dependent measurement
inventory and errors are retained in [the label-agreement audit](evidence/structural-comparison-20261004/published-label-agreement.json).
For later source selection, use `evidence/structural-comparison-20261004/next-registry.json`
(335 registered cases). This does not close the independent anatomical and
visibility validation gate.

Version **0.17.0** versions the native landmark correspondence. New observations
use MediaPipe vertex **4** for `nose_tip`, selected from retained development
evidence. Historical observations and prepared datasets keep vertex 1. The
correspondence is bound into observation/study serialization and calibration
compatibility; incompatible mappings cannot be pooled, compared or transferred.
All 478 native coordinates and perturbation samples remain intact. This change
requires a new fit and calibration; it does not repair the old model in place.
See [the repair design](docs/LANDMARK_CORRESPONDENCE_DESIGN.md).

`examples/helen-correspondence-cohort.json` freezes a new 60-source HELEN window
`[132,192)` against the retained enrollment registry. Use the pinned assets from
the preceding intake:

```shell
python -m facemath dataset cohort examples/helen-correspondence-cohort.json --archive runs/source-cohort-20261002/ibug_300W_large_face_landmark_dataset.tar.gz --model evidence/enrollment-registry-20261003/dataset/assets/face_landmarker.task --out runs/correspondence-intake
python -m facemath dataset run runs/correspondence-intake/dataset --registry evidence/enrollment-registry-20261003/next-registry.json --out runs/correspondence-study
```

This measures published-label agreement. The chin discrepancy and independent
anatomical/visibility validation remain open. The original landmark diagnosis
and historical model evidence remain unchanged.

The new native study observed 60/60 sources. On its 20 evaluation groups, the
prespecified nose mapping reduced median error from 0.1203 to 0.0525 eye spans;
10/20 nose points met the fixed 0.05 threshold, compared with 2/20 using vertex 1.
The separately fitted joint model covered 19/20 groups. These results establish
improved recorded agreement, with substantial remaining localization error.
[Raw audit](evidence/landmark-correspondence-20261004/source-audit-final.json) ·
[Full native pair report](evidence/landmark-correspondence-20261004/native-pair-final/calibrated/report.html).
The pair retains all 478 vertices; flow, edges and 96 sensitivity draws execute,
and all 142 raw geometry arrays remain unchanged when adding the research model.
Use `evidence/landmark-correspondence-20261004/next-registry.json` for subsequent
source selection; it extends the same scope to all 315 enrolled cases.
Qualification separately reports native execution and declared reference kinds.
Published PTS labels do not set the blind-human validation flag. The first repair
run exposed that reporting defect and is retained as superseded evidence; final
validation repeats the same frozen plan without retuning or claiming new sources.

Version **0.16.0** adds immutable cross-study enrollment registries. Export a
verified study's complete enrollment, retain earlier snapshots when extending a
registry, and pass `--registry` to `study-freeze` or `dataset run`. Any registered
source/hash/group/lineage overlap refuses the entire cohort before inference,
including excluded cases and sources whose earlier observation failed. Bound
studies, model exports and qualification snapshots preserve the registry through
verification, transfer and query/context checks. See
[the registry design](docs/ENROLLMENT_REGISTRY_DESIGN.md).

```shell
python -m facemath enrollment export runs/prior/study --out runs/prior-enrollment.json
python -m facemath enrollment build --receipt runs/prior-enrollment.json --scope "Explicitly named prior studies" --out runs/registry.json
python -m facemath dataset run runs/new-dataset --registry runs/registry.json --frozen-model runs/model.json --out runs/new-evaluation
```

`enrollment check REGISTRY STUDY --out REPORT` checks an existing frozen study;
overlap writes the full report and exits 2. `enrollment build --prior REGISTRY`
extends the same scope with new receipts without dropping old enrollment. These
are source-reuse checks for the declared registry scope, not inferred person
identity, independent anatomical labels or empirical admission. The retained
12-source HELEN window `[120,132)` exercises this integration with the unchanged
model; its results belong to `evidence/enrollment-registry-20261003/VALIDATION.json`.
For future studies, start with `evidence/enrollment-registry-20261003/next-registry.json`:
it includes all 255 cases from the six retained source plans. The older
`registry.json` is the immutable five-plan snapshot frozen before this integration.

Version **0.15.0** adds frozen photograph-group sampling windows for new holdouts.
The supplied HELEN plan retains the donor seed and selects ranked roots 60–119
for evaluation with the unchanged saved model. Selection retains the complete
candidate inventory and before/selected/after counts, rejects exhausted windows
and recomputes its range during verification. See
[the source-window design](docs/SOURCE_WINDOW_DESIGN.md).

Version **0.14.0** exports complete fitted joint models and evaluates them on new
frozen cohorts without changing the center, scatter or calibrated radius.
Transfer freezes the full donor enrollment with the target before inference,
rejects source/group/lineage overlap and retains failures and exact replay data.
See [the transfer design](docs/MODEL_TRANSFER_DESIGN.md).

Version 0.13.0 reports joint bound widths, complete delivery denominators and
uncertain eye-span/mouth-width geometry in every joint qualification and replay.
The HELEN example samples photograph roots before faces to keep related source
files together. See [the frozen utility design](docs/BOUND_UTILITY_DESIGN.md).
These are research diagnostics; finite output does not establish useful accuracy.

Version 0.12.0 added prospectively selected, pinned original PTS cohorts to the
same native study pipeline. The supplied LFPW plan selects 60 original photographs
from 224 candidates, with 20 groups each for development, calibration and evaluation.
It retains full-precision labels, source bytes, member inventory and selection rules.
Published labels provide research agreement evidence, with unknown visibility,
training overlap and person independence. See [the cohort design](docs/SOURCED_COHORT_DESIGN.md).

The 0.11.2 runtime contract binds exact replay to recorded numerical kernel bytes, CPU
dispatch and thread settings, with a machine-readable compatibility command.
The 0.11.1 isolated native execution and single OpenCV 4 contrib environment
remain in place. The automated dataset pipeline can fetch assets,
preserve source coordinates, freeze splits, observe, fit, calibrate, evaluate and
replay. It also retains failed study artifacts and journals. No user-supplied
image database or new human annotation campaign is required to run it.
The prior 0.10.1 blocker repairs remain documented in
[the remediation record](docs/REVIEW_REMEDIATION_20261001.md).

## Run the sourced native pipeline

To reproduce the 0.15.0 HELEN holdout (the same enrolled sources, not a new sample):

```shell
uv run --locked --extra observer python -m facemath dataset cohort examples/helen-transfer-window.json --archive runs/lfpw-assets/source.tar.gz --model runs/lfpw-assets/face_landmarker.task --out runs/helen-holdout-intake
uv run --locked --extra observer python -m facemath dataset run runs/helen-holdout-intake/dataset --frozen-model evidence/source-window-20261003/helen-model.json --out runs/helen-holdout
uv run --locked --extra observer python -m facemath verify runs/helen-holdout
uv run --locked --extra observer python -m facemath replay runs/helen-holdout --out runs/helen-holdout-replay
```

The offset counts photograph roots after grouping, with the same archive,
candidate prefix, algorithm and seed as the donor. It does not count faces or
successful detections. Source separation is checked before native transfer;
all failures remain in the frozen enrollment. This published-label research
experiment does not authorize empirical queries or establish person independence.

To evaluate the saved HELEN model on the independent IBUG source selection:

```shell
uv run --locked --extra observer python -m facemath model-export evidence/bound-utility-20261002/preseal/native-pipeline/qualification --out runs/helen-model.json
uv run --locked --extra observer python -m facemath dataset cohort examples/ibug-transfer-cohort.json --archive runs/lfpw-assets/source.tar.gz --model runs/lfpw-assets/face_landmarker.task --out runs/ibug-intake
uv run --locked --extra observer python -m facemath dataset run runs/ibug-intake/dataset --frozen-model runs/helen-model.json --out runs/ibug-transfer
uv run --locked --extra observer python -m facemath verify runs/ibug-transfer
uv run --locked --extra observer python -m facemath replay runs/ibug-transfer --out runs/ibug-transfer-replay
```

Use the retained `evidence/model-transfer-20261002/helen-model.json` directly when
the donor bundle is not available. The capsule and prepared dataset are portable;
neither requires the original archive or donor images to execute. The target has
20 evaluation groups and no fitting splits. Its source-disjointness checks do not
prove person independence or exchangeability. `evaluation/transfer.json` records
coverage, bound widths, unchanged model digests and full enrollment. This research
transfer does not grant empirical query admission. See the
[operations guide](docs/NATIVE_PIPELINE_OPERATIONS.md) for resumable execution.

The original PTS cohort can be reproduced without a user-supplied image database:

```shell
uv sync --locked --python 3.11 --extra dev --extra observer
uv run --locked --extra observer python -m facemath dataset cohort-fetch examples/lfpw-cohort.json --out runs/lfpw-assets
uv run --locked --extra observer python -m facemath dataset cohort examples/lfpw-cohort.json --archive runs/lfpw-assets/source.tar.gz --model runs/lfpw-assets/face_landmarker.task --out runs/lfpw-intake
uv run --locked --extra observer python -m facemath dataset run runs/lfpw-intake/dataset --out runs/lfpw-001
uv run --locked --extra observer python -m facemath verify runs/lfpw-001
uv run --locked --extra observer python -m facemath replay runs/lfpw-001 --out runs/lfpw-001-replay
```

The archive download is about 1.8 GB; selected original assets plus the model are
about 22 MB. Use existing pinned local archive/model paths to skip downloading.
The prepared dataset is independently transferable and needs no network or full
archive for verification or new execution. iBUG annotations are restricted to
research use; their terms remain in the plan and all prepared datasets.

The earlier machine-reference pilot remains available:

```shell
uv sync --locked --python 3.11 --extra dev --extra observer
uv run --locked --extra observer python -m facemath dataset fetch examples/native-dlib-pilot.json --out runs/pilot-assets
uv run --locked --extra observer python -m facemath dataset prepare runs/pilot-assets/recipe.json --out runs/pilot-dataset
uv run --locked --extra observer python -m facemath dataset run runs/pilot-dataset --out runs/pilot-001
uv run --locked --extra observer python -m facemath verify runs/pilot-001
uv run --locked --extra observer python -m facemath replay runs/pilot-001 --out runs/pilot-001-replay
```

The recipe pins nine public PASCAL example photographs, 43 published dlib model
annotations, and Google's Face Landmarker model. The real Windows/Python 3.11
pilot observed all 43 crops and exactly replayed its fitted joint error model.
These labels measure **machine-reference agreement**. The small calibration
inventory correctly yields an unbounded radius; evaluation outcomes do not
change the frozen split, fit, or confidence level.

Transfer the complete run directory. It contains original asset bytes and pins,
crop transforms, labels, frozen group splits, native process records, observations,
fitted center/scatter, calibration, evaluation and replay. `verify` checks those
bindings and recomputes dataset conversions; `replay` recomputes retained
numerical results. Another `dataset run` creates a distinct native execution.
See [pipeline operations](docs/NATIVE_PIPELINE_OPERATIONS.md) and
[the design](docs/NATIVE_PIPELINE_DESIGN.md).

Check a transferred run's numerical prerequisites before replay:

```shell
python -m facemath runtime-profile --compare runs/pilot-001/qualification/environment.json --out runs/runtime-comparison.json
```

Exit status 2 means the runtime is incompatible. Matching package versions alone
are insufficient; even a matching profile still requires exact output comparison.
Byte verification and inspection remain available across hosts. See the
[numerical replay contract](docs/NUMERICAL_REPLAY_DESIGN.md).

For reproducible development, use `uv sync --locked --extra dev`, then
`uv run --locked python -m pytest -q`. The lock includes one OpenCV 4 contrib
distribution shared by the Haar locator and MediaPipe. The native recipe supplies explicit runtime/model pins;
human-reference empirical admission remains a separate optional evidence gate.

Run `uv run --locked python scripts/validate_current.py --out runs/validation-001`
for source, installed-wheel and extracted-archive checks, including milestone
controls and verify/replay. The destination must be new. This validates the
executing host; the hosted CI matrix remains configuration until actually run.
The root `RELEASE_MANIFEST.json` describes the historical Wave10 archive. New
manifests are generated only inside new release stages.

A local Python instrument for **projected facial geometry, observation error, and auditable source evidence**. It does not identify people or turn a visually chosen percentage into a biometric measurement.

Wave 10 consolidates source intake → independent annotation → frozen acquisition → error-model qualification → measured query sources → geometric comparison into one operational workflow. The existing `Observation`, `CanonicalState`, error models, relation graph and evidence bundles remain authoritative.

**New at this milestone:** an append-only reference workspace; a prespecified held-out acceptance policy; full-study numerical replay before the optional empirically gated query path; evaluation/context leakage checks; an actual native-execution smoke command; a compact result readout. Ordinary research analysis remains available and explicitly labeled.

## First executable result

```shell
python -m pip install -e ".[dev]"
python -m facemath demo --size 160 --draws 8 --no-flow --out runs/research-demo
python -m facemath overview runs/research-demo
python -m facemath verify runs/research-demo
```

Open `runs/research-demo/report.html`. This is a known-coordinate **synthetic** demonstration. The fraction of a declared relation inventory within a stated tolerance is a computable percentage; it is not a probability of resemblance. Unresolved/missing relations remain in the denominator.

## Complete milestone control

```shell
python -m facemath milestone-lab --out runs/wave10
python -m facemath overview runs/wave10/positive/qualification
python -m facemath overview runs/wave10/negative/qualification
python -m facemath overview runs/wave10/admitted_pair
python -m facemath replay runs/wave10/admitted_pair --out runs/wave10-pair-replay
python -m pytest -q
```

The positive and negative workflows are frozen independently before their scripted outcomes run. The positive branch is an **engineered containment/acceptance control**, not a representative or exchangeable human study. The negative branch keeps the same policy but shifts the observer's errors in one required category. All requested categories must pass; no successful subset is selected after seeing outcomes.

## Start an actual reference study

```shell
python -m facemath workspace prepare study/workspace.spec.json --out work/portraits
python -m facemath workspace status work/portraits
python -m facemath workspace import work/portraits labels/portrait-rater-a.json --case-id portrait_0001
python -m facemath workspace freeze work/portraits --out studies/portraits-v1
python -m facemath observer-smoke portrait.png --model models/face_landmarker.task --out runs/native-smoke
python -m facemath study-run studies/portraits-v1 --model models/face_landmarker.task --out runs/portraits-v1
python -m facemath study-finalize runs/portraits-v1 --out qualification/portraits-v1
```

The spec needs actual images, independently produced reference annotations, declared dependent groups/lineages, a pinned compatible runtime/model and a source-measurement policy designed on development data. The example template is not a sufficient dataset. See [the consolidated operator guide](docs/WAVE10_OPERATIONS.md).

`observer-smoke` never downloads a model or installs a runtime. Missing prerequisites produce **not_attempted**, not a fictional detector failure or accuracy result. A successfully running detector would still need independent reference evaluation.

## Research versus empirically gated queries

`--localization-calibration model.json` applies a research calibration as before. The report does not infer empirical approval merely because its radius is finite.

`--qualification qualification/portraits-v1` loads the **complete retained study**, replays its numerical qualification, requires all frozen acceptance criteria to pass, and rejects query source/context overlap with development, calibration, evaluation and excluded enrollment. Original source-group pixels remain necessary. The portable pair stores the qualification snapshot, not just an approval string.

```shell
python -m facemath analyze query/a.png query/b.png \
  --observation-a query/a.observation.json --observation-b query/b.observation.json \
  --qualification qualification/portraits-v1 \
  --acquisition-a query/a-source --acquisition-b query/b-source --out pairs/qualified
```

Use `--model` instead of imported observation files for a pinned native observer. Source and acceptance gates run before its deferred construction. Failed instrument invocations preserve an actual partial journal at `<out>.failure.events.jsonl`, without overwriting previous evidence.

## Read the evidence

- `summary.json`: compact view of existing measurements, bounds, relation fractions and acceptance status.
- `evidence.json`, `intrinsic.json`, `arrays.npz`: original measurements and complete numerical arrays.
- `events.jsonl`: actual instrument-stage events, not private neural telemetry.
- `qualification_snapshot.zip`: retained raw study, observations, labels, receipts and qualification behind an admitted pair.
- Workspace `labels/`: original annotations and explicit superseding revisions. Freeze records the complete chain and selected latest labels.

Hashes establish byte consistency; replay establishes consistency with the retained numerical inputs and implementation. Neither proves external chronology, true subject independence, human blinding, accurate source metadata, or resistance to an attacker recreating an entire consistent dataset.

## Scientific and release boundaries

The release's `evidence/WAVE10_VALIDATION.json` records what was executed. Synthetic controls and API fixtures are not live detector validation. The actual native smoke outcome is separate. No new similarity score for the original video's two people is manufactured.

Eight-point reference calibration does not validate all 478 vertices, 3D anatomy or identity. Source gradient rules are not face-quality classifiers. A passed held-out policy is conditional on its sampling assumptions and does not guarantee performance under hidden shift or control repeated adaptive studies. Exact replay requires the matching package code and numerical runtime.

## Development map

[Design and audit](docs/WAVE10_DESIGN.md) · [Mathematics](docs/WAVE10_MATHEMATICS.md) · [Operations](docs/WAVE10_OPERATIONS.md) · [Milestone architecture map](docs/MILESTONE_MAP.md) · [Next development gates](docs/WAVE10_ROADMAP.md) · [Contributor contract](AGENTS.md)

The inherited section-number registry is a research inventory, **not proof of implementation of every historical mathematical analogy**. The active implementation and its non-equivalent/missing research methods are separately documented by `facemath capabilities` and the retained Wave 3 audit.
