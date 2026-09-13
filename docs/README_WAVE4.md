> Historical Wave 04 README. Its older relative artifact paths refer to the parent release, not new Wave 05 validation.

# FACEMATH · Wave 04

**Observation-aware facial geometry, not identity recognition.**

Wave 04 connects independent reference acquisition and localization-error calibration to the existing shared anatomical instrument. The core question is no longer only *“how closely do these coordinates align?”* It can also ask *“is that difference resolved by the observation accuracy we have actually established?”*

This is working research software, **not a validated human facial-accuracy product**. The included laboratory is explicitly synthetic. Live MediaPipe inference and independent human localization accuracy remain unvalidated in this release: runtime/model acquisition did not succeed in the development environment. Nothing silently substitutes a synthetic observer for the real provider.

## Install and run

Python 3.11+ is required; release execution used Python 3.13. Install from this source tree:

```powershell
python -m pip install -e ".[dev]"
python -m facemath doctor
python -m facemath validation-lab --out runs/wave4
python -m facemath verify runs/wave4/validation
python -m facemath replay runs/wave4/validation --out runs/validation-replay
python -m facemath verify runs/wave4/qualified_pair
python -m facemath replay runs/wave4/qualified_pair --out runs/pair-replay
python -m pytest -q
```

Every output directory is create-only. Choose a new destination for subsequent runs. Open `runs/wave4/reference_workbench.html`, `runs/wave4/validation/report.html`, and `runs/wave4/qualified_pair/report.html` in a browser. The reports and workbench need no remote assets. Third-party model-runtime telemetry is **not** attested by that statement.

For live observation, install the optional provider and supply a local model explicitly:

```powershell
python -m pip install -e ".[observer]"
python -m facemath doctor --model models/face_landmarker.task
python -m facemath analyze face_a.png face_b.png --model models/face_landmarker.task --out runs/pair
```

A successful `doctor` check proves availability only, never that inference or accuracy evaluation occurred. See [provider setup](docs/MODEL_SETUP.md) and the upstream sources in [Wave 4 design](docs/WAVE4_DESIGN.md).

## One instrument, two acquisition paths

```text
SOURCE RASTER ──┬─ observer → immutable Observation ────────────────────┐
                │                                                     │
                └─ blank reference workbench → independent annotation │
                                    │                                 │
                         source-coordinate validation                 │
                         group-held-out calibration                   │
                                    │                                 │
                         LocalizationCalibration ──────────────┐      │
                                                             ▼      ▼
                                            existing CanonicalState
                                             /       |        \
                            geometric field     image edges    error ellipses
                                  |                  |               |
                            shape measurements   diagnostics   gap outer bounds
                                             \       |        /
                                            same evidence bundle
```

Calibration never moves the fitted landmarks or replaces the ordinary measured profile. It adds an explicit accuracy-qualified envelope for the eight independently referenced points. The complete 21/478-point geometry remains available, but eight reference labels cannot validate all mesh vertices.

## Acquire independent references

```powershell
python -m facemath annotate portraits/a.png --source "Original portrait; frontal acquisition protocol" --out labels/a.workbench.html
```

The workbench contains the source pixels and eight semantic instructions, **not detector predictions**. It supports zoom/pan, pixel magnification, subpixel nudges, explicit unobservable labels, undo/redo, per-rater local drafts, and image-bound JSON export. Changing raters clears the previous labels. Drafts are convenience storage; retain exported JSON files. Blinding and rater identity are declarations, not externally verified facts.

The reference grammar uses viewer-side eye corners, nose tip, mouth corners and central chin. Never guess an occluded chin beneath a beard. An explicit PTS importer accepts a reviewed eight-point subset of iBUG68 with mandatory coordinate-origin selection:

```powershell
python -m facemath import-ibug-reference portrait.png landmarks.pts --origin 1 --source "Dataset name / authorized local sample" --out reference.json
```

No external face dataset or model weights are redistributed. Imported PTS files lack independent visibility labels and their semantic mapping requires review.

## Evaluate an observer

Write a study manifest using [the schema example](examples/validation.study.template.json). Paths are relative to the manifest. Curators must declare groups/lineages and split them **before** inspection of results. Use `development` for tuning, `calibration` for the frozen error envelope, and `evaluation` for the held-out check. A single case may contain several genuinely independent raters.

```powershell
python -m facemath validate-observer study.json --model models/face_landmarker.task --alpha 0.1 --out runs/observer-validation
```

Each case either invokes the supplied provider, imports an exact image-bound observation, or imports an explicitly recorded failure. The journal distinguishes all three. Errors are measured in **source pixels without registration to the reference**. Failed attempts remain in PCK, AUC and calibration denominators. Duplicate images, cross-split groups and declared-lineage leakage are rejected.

Calibration uses the maximum normalized reference error over the eight landmarks, then over each declared acquisition group. Too few groups or too much failed/missing evidence can correctly yield an **unbounded** envelope. It does not clip the quantile or turn an unavailable number into zero error.

## Use a matching localization calibration

```powershell
python -m facemath analyze query_a.png query_b.png `
  --model models/face_landmarker.task `
  --localization-calibration runs/observer-validation/localization_calibration.json `
  --query-groups query-group-A query-group-B `
  --lineage-a source-session-A --lineage-b source-session-B `
  --domain "EXACT acquisition domain from study.json" `
  --out runs/qualified-pair
```

Observer recipe, domain, source-image and declared-group/lineage checks are mandatory. A synthetic calibration cannot be attached to an ordinary model/human observation. The resulting intervals apply to the **observed coordinate gauge and normalizer held fixed**, not to an unknown re-fitted true facial shape. They are not a probability that two people resemble one another.

## Raw numerical outputs

The validation bundle contains references, predictions, images, covariance matrices, errors, masks, every attempted case, scores, calibration group maxima, held-out results, source digests, environment, actual execution journal and byte manifest. The qualified pair additionally contains the shared canonical state, sparse/dense fields, per-region measurements, edge profiles, full covariance pushforward, sampled ellipse support functions, and their bounds.

`verify` checks byte and array consistency. `replay` recomputes from frozen source rasters and observations using matching code and numerical runtime. Neither is external attestation of labels, training data or execution authorship. Replay does **not** pretend to rerun an ML model when it is replaying saved predictions.

See [raw array axes and units](docs/RAW_ARRAYS.md) before interpreting the exports.

## Development and validation

- [Wave 4 design, mathematics and boundaries](docs/WAVE4_DESIGN.md)
- [Acquisition and annotation protocol](docs/REFERENCE_PROTOCOL.md)
- [Wave 4 validation record](evidence/WAVE4_VALIDATION.json)
- [Critical next priorities](docs/WAVE4_ROADMAP.md)
- [Wave 3 audit of inherited mathematics](docs/WAVE3_AUDIT.md)

The earlier research modules remain available. Their section-name registry is not a proof that every original analogy is executable mathematics. Wave 04 extends the tested observation instrument without making that historical completeness claim.
