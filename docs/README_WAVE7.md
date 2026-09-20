# FACEMATH · Wave 07

An integrated Python instrument for **facial geometry and observation error**, not identity recognition.

Wave 07 keeps a whole face's localization errors connected. A development-fitted 16-dimensional error ellipsoid over eight reference landmarks feeds the existing frozen study, calibration, canonical state, geometric relationship graph, proper-rotation shape bounds, report and numerical replay. Shared translation can cancel before it is mistaken for deformation. Observed coordinates and measured geometry do not change.

## Execute the integrated development laboratory

```shell
python -m pip install -e ".[dev]"
python -m facemath joint-lab --out runs/wave7
python -m facemath verify runs/wave7/pair_02_mouth_8px
python -m facemath replay runs/wave7/pair_02_mouth_8px --out runs/pair-replay
python -m pytest -q
```

Open `runs/wave7/pair_02_mouth_8px/report.html`. The appended joint-error inspector shows coherent displacement modes, the full coordinate-correlation matrix, and joint versus same-set marginal-relaxation shape ranges. Sliders are display-only; exact matrices and mode vectors are exported.

The laboratory is **synthetic**, with 97 enrolled renderer sources, a scripted observer, frozen split/regularization, retained failures and an intentionally shifted held-out category. Its 1,024-pair set control saves every sampled shape and 256 exact relationships. Samples test conservative bounds, not establish universal containment or human performance.

## One instrument, two error-set representations

```text
Image-bound predictions + independent source references
                         │
               Frozen prospective study
                         │
            Development-only mean and scatter
                         │
          Group-calibrated full16 error radius
                         │
                 Existing CanonicalState
           ┌─────────────┼─────────────────┐
           │             │                 │
      Raw geometry   Fixed-gauge      Intrinsic relations
      unchanged      point bounds    + refitted shape bounds
           └─────────────┼─────────────────┘
               Same evidence bundle / replay
```

A full16 ellipsoid is not a product of eight independent balls. The diagnostic product relaxation uses the same center, q and marginal blocks; comparing their widths isolates the cost of discarding the shared latent constraint (cross-point correlation and the single ellipsoid norm budget), not a population-calibration improvement.

## Real inputs

The existing `analyze`, `annotate`, `observer-pin`, `study-freeze`, `study-run` and `study-finalize` commands remain the workflow. Select `bias_mode: "development_joint_ocular"` and frozen `joint_error` settings in a complete study manifest. Attach its model through the existing `--localization-calibration` option. See [operations](docs/WAVE7_OPERATIONS.md).

**Execution boundary:** the local MediaPipe package and model could not be acquired in this environment. Live detector inference remains explicitly skipped. No human accuracy, original-video comparison, population resemblance score, or private model telemetry is manufactured by these controls.

## Inspect the release

- [Design, integration and audit](docs/WAVE7_DESIGN.md)
- [Mathematical definitions and bounds](docs/WAVE7_MATHEMATICS.md)
- [Commands and interpretation](docs/WAVE7_OPERATIONS.md)
- [Raw array definitions](docs/RAW_ARRAYS.md)
- [Prior release README](docs/README_WAVE6.md)
- [Historical mathematical-completeness audit](docs/WAVE3_AUDIT.md)

`evidence/WAVE7_VALIDATION.json` records the actual tested artifacts and execution results. Hashes establish self-consistency, not independent attestation. Numerical replay recomputes from captured observations; it does not rerun the detector. Independent real annotations and held-out acquisition-domain evaluation remain essential to human performance claims.
