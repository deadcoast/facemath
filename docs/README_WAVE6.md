# FACEMATH · Wave 06

An integrated Python instrument for **facial geometry and observation error**, not identity recognition.

Wave 06 propagates the existing eight-point localization error sets into shape relationships whose **normalization and optimal alignment are themselves uncertain**. The original observations, canonical registration, measured geometry, masks and dense field remain unchanged. One cached source-relation graph supplies 28 distances, 168 angles, 56 oriented triangle areas and four bilateral contrasts, followed by an outer range for a newly fitted proper-rotation shape distance.

## Run the integrated development laboratory

```shell
python -m pip install -e ".[dev]"
python -m facemath intrinsic-lab --out runs/wave6
python -m facemath verify runs/wave6/pair_02_mouth_8px
python -m facemath replay runs/wave6/pair_02_mouth_8px --out runs/pair-replay
python -m pytest -q
```

Open `runs/wave6/pair_02_mouth_8px/report.html`. It includes a source-coordinate relation workbench, explicit uncertainty-set view, display-only zoom, anatomical filtering and exact selected-row CSV export. `intrinsic.json`, `intrinsic.csv`, `arrays.npz` and the existing execution journal expose the data behind it.

**The laboratory is synthetic.** It runs the inherited frozen study, scripted observation acquisition, development-only bias fitting and calibration, then four prespecified mouth-deformation queries. A separate set laboratory retains 1,024 sampled shape pairs and all 256 relation values for falsification checks. Sampling does not generate the reported bounds or prove universal containment.

## Analyze real inputs

```shell
python -m facemath analyze a.png b.png --observation-a a.json --observation-b b.json --out pairs/observed
```

Without a localization calibration, source-bound observations produce nominal relations but no invented uncertainty bounds. To attach a compatible independently fitted calibration:

```shell
python -m facemath analyze a.png b.png --observation-a a.json --observation-b b.json --localization-calibration qualification/localization_calibration.json --query-groups query_a query_b --lineage-a session_a --lineage-b session_b --domain "exact-declared-domain" --out pairs/qualified
```

Or supply a local compatible model through `--model` instead of observation files. The native study workflow is documented in [Wave 05 operations](docs/WAVE5_OPERATIONS.md). No model is downloaded implicitly.

**Execution boundary:** native model acquisition failed in this build environment. Real MediaPipe inference remains a separate skipped test, not a successful human-validation result. Neither synthetic controls nor a conditional geometric enclosure certify detector accuracy or population resemblance. The released execution records state the actual environment and commands.

## One state and two distinct geometric questions

```text
Image-bound observations + qualified source error model
                           │
              Existing CanonicalState
               ┌───────────┴─────────────┐
               │                         │
     Registered residuals         Intrinsic source relations
     fixed display gauge          uncertain reference eye spans
               │                         │
     Existing field/masks         Refitted proper-shape range
               └───────────┬─────────────┘
                    Same evidence bundle
```

The new relations use the same eight source coordinates, full covariance matrices, visibility and model-bound error centers. They do not infer another face, move landmarks, or add 256 independent votes. The shape distance is eye-normalized RMS after centering and **proper rotation only**; reflections are not permitted, and there is no second scale fit.

## Technical references

- [Wave 06 architecture and audit](docs/WAVE6_DESIGN.md)
- [Derivations and numerical limits](docs/WAVE6_MATHEMATICS.md)
- [Commands and result interpretation](docs/WAVE6_OPERATIONS.md)
- [Raw arrays](docs/RAW_ARRAYS.md)
- [Inherited mathematical-completeness audit](docs/WAVE3_AUDIT.md)
- [Prior release README](docs/README_WAVE5.md)

Hashes establish bundle consistency, not external attestation. Numerical replay recomputes from captured observations; it does not rerun a detector. Independent annotations, a suitable acquisition domain and a real live-observer evaluation remain necessary for human accuracy claims.
