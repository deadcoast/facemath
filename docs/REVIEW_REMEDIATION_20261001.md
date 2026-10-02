# Active workflow remediation

This development block implements the first blocker group from the retained
`REPOSITORY_REVIEW.json`: I01, I02, I03, I04, I07 and I08. Release safety (I06)
and current dependency metadata (I05/D03) are included because the required
installed-wheel and extracted-archive checks depend on them. The original review,
historical archives and sealed release evidence remain the baseline.

## Design

- Hold the acquisition writer lock through verification and the complete snapshot.
  Exclude only the top-level process lock from copying, then create a canonical
  unlocked sentinel in the snapshot. Nested evidence and all observation bytes
  remain included. Replaying must not change an inventory by creating a new file.
- Validate original decoded ZIP names before destination construction. Inspect
  every name before writing any member. Reject normalized aliases, file/directory
  prefix collisions, and the existing traversal/device/symlink hazards. Regression
  fixtures must verify the actual local and central header bytes.
- Retain Python 3.11 syntax support. Require OpenCV 4 for the existing Haar locator;
  explicitly reject missing APIs/assets with `ProviderUnavailable`. No alternate
  landmarks or automatic model acquisition are introduced.
- Keep held-out regional support as the observed count and its total as the
  declared anatomical inventory. Coordinates, fits and numerical values are
  unchanged.
- Package from explicit source/config/docs/example roots, a chosen wheel and
  explicitly selected new evidence. Exclude environments, caches, local runs,
  historical archives and generated metadata. Stage first; generate a manifest
  from the final staged bytes; write versioned outputs with create-only semantics.
  Preserve the root historical Wave10 manifest as historical evidence.

## Validation plan

Capture the actual Wave10 parent ZIP digest, environment, full inherited test
output and return code before source edits. Add focused regressions for locking,
malformed raw ZIP names, partial/absent support and packaging selection. Run the
complete source suite, milestone-lab, verify and replay against one final source
fingerprint. Check every retained observed-state array for qualification
invariance. Build a wheel, install it outside the checkout, verify its Python
bytes/import origin, and run the suite and replay there. Seal a fresh archive,
verify its inventory, extract it and repeat the suite and workflow checks.

Python 3.11 execution, Windows execution and any configured hosted OS matrix are
reported separately. All failures and skip reasons are retained. Synthetic
acceptance controls remain synthetic; independent real portraits, pinned native
observation and actual annotation-pointer validation remain separate open gates.

## Dependency and release operation

The patch version is 0.10.1. Regenerate metadata from `pyproject.toml`; legacy
egg-info is removed from Git tracking and retained locally as ignored generated
build output, never an independent version authority.
The original environment remains untouched during this run. New Python 3.11 and
3.14 validation environments use `UV_PROJECT_ENVIRONMENT` and the repository-local
`UV_CACHE_DIR`. The [uv project configuration](https://docs.astral.sh/uv/concepts/projects/config/)
documents this environment selection; [locked synchronization](https://docs.astral.sh/uv/concepts/projects/sync/)
checks that the project and lock agree. The existing locator uses the
[OpenCV cascade API and bundled XML assets](https://github.com/opencv/opencv-python/blob/4.x/README.md).

`scripts/validate_current.py --out runs/new-validation` executes the current
source suite, milestone, qualification/pair verify and replay, builds the wheel,
tests its outside-checkout installation, checks all wheel Python bytes, stages
and seals the archive, then extracts and repeats the suite and workflow. Actual
commands, return codes, raw logs and JUnit skip reasons live in the chosen run.
Failed destinations are retained; choose a new path for another attempt.
Use `--retain-evidence evidence/<new-run>` to copy completed source and installed
execution records plus the complete source laboratory into a new retained
evidence directory before sealing. The archive includes that directory; the
extracted-archive checks remain adjacent records generated afterward.

`scripts/package_release.py --wheel <wheel> --stage <new-stage> --output
<facemath-0.10.1-name.zip> --dry-run` prints the selected inventory without writes.
Remove `--dry-run` to seal it. Add `--evidence evidence/<chosen-run>` explicitly
when shipping retained evidence. Lock sentinels within that evidence are retained.
Old packages, environments, local runs and historical evidence are not implicitly
selected. Archive and sidecar creation refuse replacement. Post-extraction results
belong alongside the sealed archive, whose inventory is not rewritten afterward.

The real-reference gate, annotation-coordinate/browser gate, failed-finalization
journal recovery (I16), legacy defects (I09-I15), and the remaining operational
and scientific review items are outside this first blocker group. Passing these
software checks does not close them.
