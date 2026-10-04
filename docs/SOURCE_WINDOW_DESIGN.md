# Frozen source windows and HELEN holdout

## Review

The 0.14.0 parent is the actual release archive
`packages/facemath-0.14.0-transfer-20261002.zip`, SHA-256
`054ad6bf5bccd9d2b8dd1b688f42a5a476e014e1ff192851e99d1cac3b38cb9a`.
A fresh extracted-package baseline passed 1,075 tests with one unavailable
Windows directory-symlink test skipped in 261.29 seconds. It is recorded before source changes under
`evidence/source-window-20261003/development/`. The historical mathematics audit
and repository review remain historical records.

The saved HELEN model has already been evaluated on an IBUG source selection.
The next Gate B experiment measures its delivery, coverage and bound utility on
another window of the original HELEN source universe. No outcome from either
evaluation selects a new center, scatter, radius, crop rule or observer setting.
The existing selector always starts at the first ranked source; changing the
seed to avoid overlap would permit informal resampling. An explicit rank window
makes the planned partition reproducible and keeps skipped sources visible.

## Design

Add versioned v15 PTS plans and recipes with a required integer
`selection_offset`. Only the existing HELEN/IBUG photograph-root selectors are
supported by this format. Validate the entire candidate inventory, group and
rank every source with the existing algorithm, then take the half-open window
`[selection_offset, selection_offset + sum(split_counts))`. Apply split blocks
inside this window. Never wrap, truncate, replace an invalid source, or skip an
observation failure. The offset counts photograph roots, not face annotations.

Retain zero-based source positions and a window ledger with the full candidate
annotation/root counts, before/selected/after counts and selected range. Recipe
verification recomputes the complete selection and ledger. v12 plans/recipes
remain byte-compatible for their existing inputs and reject the new field;
older readers refuse the v15 format rather than silently ignoring an offset.
Non-overlap of windows requires the same archive, candidate prefix, selection
algorithm and seed. The existing full-donor transfer guard remains authoritative
for encoded/decoded source aliases, crop hashes, groups and lineages.

## Prospective experiment

Retain the exact original HELEN seed, source archive/model pins, numeric-root
grouping, crop margin 0.25, PTS origin 1, reference8 mapping, observer/runtime and
alpha 0.1. The first 60 ranked roots belong to the prior 20/20/20 donor study.
The new plan selects roots 60 through 119 as 60 evaluation-only groups. Verify
the donor plan reproduces the first window; compare the new selected original
encoded/decoded hashes and group IDs against all retained LFPW, HELEN and IBUG
enrollment before inference. Freeze the complete capsule and selection first.

The model capsule and its exact serialized fit are unchanged. All source,
wheel and archive executions are explicitly distinct runs of the same frozen
plan. Preserve no-face/multi-face/runtime outcomes and every enrolled denominator.
Report finite delivery, coverage and uncertain eye-span utility separately.
No new empirical policy, success threshold, model selection or admission is added.
The experiment is chosen after prior studies; it is not a correction for adaptive
cross-study selection. Published-label agreement and filename-root separation do
not establish person independence, visibility, anatomical equivalence or absence
from native-model training.

## Plan and validation

1. Complete the inherited extracted-package baseline and retain raw output.
2. Implement and test window boundaries, grouping before slicing, split blocks,
   inventory-order invariance, malformed offsets, exhaustion, legacy stability,
   downgrade refusal and resealed selection/ledger tampering.
3. Freeze and execute the new real-portrait holdout with the existing native
   pipeline, preserving the unchanged model and original observations.
4. Run full source tests, milestone-lab, verify/replay, isolated installed-wheel
   tests and byte checks, and final extracted-archive tests and workflows. Compare
   all retained observation/sample arrays before and after finalization/replay.
5. Record commands, environments, failures, skips, artifact digests and exact
   remaining scientific/platform gates in a root-level continuation report.

## Recorded source experiment

The first completed Windows source run observed all 60 portraits. The unchanged
model covered 56 of 60 enrolled groups (93.33%). The four incomplete cases retain
their predicted chin coordinates below the stored-raster image boundary; their
reference labels are present. Those predictions are neither clipped nor replaced,
and all 60 groups and 480 reference-landmark slots remain in the denominators.
There are 476 supported landmark measurements. Exact per-case coordinates and
boundary distances are retained in `development/support-diagnostics.result.json`.

Finite bound delivery is not useful-accuracy acceptance: the median marginal
landmark diameter is approximately 0.683 observed-eye spans, and the median
mouth-width/reference-eye-span interval width is approximately 1.605. No utility
threshold or empirical policy was selected from these outcomes. Full package,
repeated native and platform results are recorded separately in
`evidence/source-window-20261003/VALIDATION.json` after validation completes.
