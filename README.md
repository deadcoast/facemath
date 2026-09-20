# FACEMATH · Wave 08

One integrated Python instrument for **facial geometry and observation error**, not identity recognition.

Wave 08 binds localization uncertainty to prespecified acquisition conditions. One development-fitted full16 mean/covariance feeds category-local group calibration, two immutable query routes, and the same canonical geometry, shape-relation graph, evidence bundle and replay. A clear source and a degraded source can carry different error radii without altering their observed coordinates or rescaling the stored covariance. Unsupported categories never inherit a pooled fallback.

## Execute the integrated laboratory

```shell
python -m pip install -e ".[dev]"
python -m facemath stratified-lab --out runs/wave8
python -m facemath verify runs/wave8/qualification
python -m facemath replay runs/wave8/pair_02_mouth_8px --out runs/pair-replay
python -m pytest -q
```

Open `runs/wave8/qualification/report.html` to inspect category calibration, retained group scores and held-out failures. Open `runs/wave8/pair_02_mouth_8px/report.html` to inspect source-bound qA/qB, the unchanged geometry, coherent error modes and the full relationship inventory. Reports use embedded data and make no remote requests.

The default laboratory enrolls 155 schematic sources, with 154 eligible scripted attempts, frozen exclusions/failures, sparse and absent categories, and a deliberately hidden within-category error shift. Four fresh geometric queries and a separate 1,024-pair admitted-set control retain raw numerical inputs and outputs. These are synthetic controls, not human detection results.

## Shared computational path

```text
Frozen source inventory + independent references + acquisition categories
                                ↓
                 Existing resumable observation runner
                                ↓
              One development-only mean / full16 scatter
                                ↓
               Prespecified category group order statistics
                                ↓
            Exact source-bound routes and separate q_A / q_B
                                ↓
                      Existing CanonicalState
                   ┌────────────┼────────────┐
                   ↓            ↓            ↓
             Raw geometry  Fixed-gauge    Intrinsic relations
              unchanged    point bounds  + refitted shape range
                   └────────────┼────────────┘
                       Same bundle / replay
```

A declared category is not an automatic domain-shift detector. The hidden-shift control remains a failure, not an occasion to tune the policy using held-out labels. Coverage among finite delivered outputs is not silently substituted for performance across all enrollment.

## Real inputs

The existing `annotate`, `observer-pin`, `study-freeze`, `study-run`, `study-finalize` and `analyze` commands remain the workflow. The new mode is `development_joint_ocular_stratified`; calibrated query pairs supply `--query-strata` for A and B. Full instructions: [operations](docs/WAVE8_OPERATIONS.md).

**Execution boundary:** MediaPipe was not installed and actual host lookup/model-download attempts failed in this environment. The live integration test remains explicitly skipped. Independent real reference annotations and native observer evaluation are still required; no original-face score or human accuracy claim is supplied by the synthetic laboratory.

## Inspect

[Design and audit](docs/WAVE8_DESIGN.md) · [Mathematics](docs/WAVE8_MATHEMATICS.md) · [Operator guide](docs/WAVE8_OPERATIONS.md) · [Raw array definitions](docs/RAW_ARRAYS.md) · [Previous README](docs/README_WAVE7.md)

`evidence/WAVE8_VALIDATION.json` records executed commands, test counts, distribution equivalence and replay results. Hashes establish internal consistency, not independently attested authorship. Replay recomputes from captured observations; it does not rerun a detector or expose proprietary neural telemetry. The inherited mathematical section registry remains an inventory, not proof that every historical analogy has been implemented.
