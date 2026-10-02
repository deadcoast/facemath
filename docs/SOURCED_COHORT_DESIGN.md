# Frozen published-reference cohort

This milestone starts from the actual 0.11.2 archive, SHA-256
`785d5e7cde06b32d52313ceefb0d8981b2453a6d3c99950ff523a1485273414d`.
Its extracted-package baseline completed before these changes: 1004 passed,
one unavailable Windows directory-symlink test skipped. The existing raw archive
test log and command receipt are reused explicitly, not represented as a new run.

## Prospective experiment

Use the public [dlib 300W repack](https://dlib.net/files/data/) with its exact
archive size and SHA-256 pinned in `examples/lfpw-cohort.json`. The candidate
universe is all original image/PTS pairs in `lfpw/testset`. Exclude the repack's
explicit `_mirror` derivatives from sampling. Do not use its randomized XML
train/test allocation. Rank original member paths by SHA-256 of a fixed seed,
NUL separator and UTF-8 path; take the first 60, assigning consecutive blocks of
20 to development, calibration and evaluation. Freeze this rule before native
inference. Fewer candidates, duplicate image bytes, ambiguous pairings or unsafe
archive members fail intake; they are not silently substituted.

Each source image has one PTS annotation and its own group/lineage. Encoded and
decoded duplicate checks prevent known exact aliases. Different photographs may
contain the same person or related crops: these groups do not establish verified
person independence, exchangeability, population coverage or absent model-training
overlap. No outcome-dependent replacement, retry, threshold or split tuning occurs.

## Coordinates and provenance

Preserve original encoded images and original PTS bytes. PTS coordinates use the
[published one-based convention](https://ibug.doc.ic.ac.uk/resources/300-W/).
Unlike the repack's rounded, zero-based XML, the PTS precision is retained. Derive
an inclusive integer box from all 68 original point extrema, expand by 0.25 of
its width/height, clip to stored raster and record its integer translation. Do
not rotate, mirror, resize, apply EXIF display transforms or fit points to the
observer. The shared dataset conversion applies the declared origin exactly once.

Map the existing reference8 indices. In-frame coordinates do not prove visibility;
occlusion and annotation uncertainty are unavailable. Missing normalization support
is excluded prospectively with a retained reason; other missing points survive in
the existing frozen denominators. Labels are `external_dataset` with prediction
exposure `unknown`, not newly performed human testing. The upstream labels are
[semi-automatic and research-only](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).
No commercial-use permission or independently validated anatomical equivalence is
inferred. This cohort does not request empirical admission.

## Pipeline and transfer

Add bounded, explicit archive download and local cohort intake. Hash the complete
archive before reading it; stream and hash every regular member. Reject unsafe
names, aliases, links/special files and excessive counts/expanded sizes. Extract
only selected originals into portable flat asset names; never call extractall.
Retain the complete member inventory, selection algorithm, selected-member pins
and model pin in the dataset recipe. Reverification recomputes selection and
coordinate conversion from the transferred retained inputs. The full archive is
not required for offline use; archive authenticity remains an upstream provenance
claim, while a fresh intake can reproduce membership from the pinned download.

Reuse the existing freezer, isolated observer, development joint fit, calibration,
evaluation and exact replay. Keep alpha 0.1, joint shrinkage 0.02 and floor 0.001
eye spans from the prior pilot. Report all enrolled/eligible groups and failures,
finite delivery and held-out agreement. A failed experiment remains evidence.

## Validation

Test deterministic selection independent of archive order, mirror exclusion,
origin/precision/crop translation, malformed PTS, missing pairs, duplicate aliases,
unsafe tar members, pin failures, post-transfer resealed selection tampering and
no empirical-admission escalation. Run the 60-case pinned native study through
source, installed wheel and extracted archive; compare observed arrays and exact
retained replay. Run inherited full tests, milestone controls and byte checks.
Keep hosted scheduler failures separate from actual local execution.
