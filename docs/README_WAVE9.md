# FACEMATH · Wave 09

An integrated Python instrument for **facial geometry, source acquisition and observation error**, not identity recognition.

Wave 09 makes acquisition routing reproducible from original image pixels. A frozen policy measures each explicitly declared source aperture, reduces the complete group's measurements, and evaluates every routing rule before landmark inference. Those source groups authorize the existing category-specific joint-error model and enter the same CanonicalState, shape-expression graph, evidence bundle and numerical replay. A category string alone is no longer sufficient for a measured calibration.

## Run the complete controlled experiment

```shell
python -m pip install -e ".[dev]"
python -m facemath measured-lab --out runs/wave9
python -m facemath verify runs/wave9/qualification
python -m facemath verify runs/wave9/query_group_a
python -m facemath replay runs/wave9/pair_02_mouth_8px --out runs/pair-replay
python -m pytest -q
```

Open `runs/wave9/qualification/report.html` for acquisition, calibration and held-out failures. Open `runs/wave9/pair_02_mouth_8px/report.html` for actual source apertures, exact measurement-grid displays, every routing predicate, source-specific radii, joint error modes and all shape relations. Open `runs/wave9/pixel_controls/noise/report.html` for an important negative control: noise can pass a gradient-detail rule. **These predicates are not a face-quality detector or a promise of localization accuracy.**

The default lab enrolls 119 schematic sources, includes a repeated mixed-quality group, retains 118 scripted attempts and two nondetections, and excludes one enrollment prospectively from observation. Four fresh pairs have prescribed mouth changes. Hidden observer-error shifts remain visible within the same measured source category. This is synthetic evidence for software/numerical contracts, not human validation.

## One shared instrument

```text
Original source pixels + declared apertures + complete group inventory
                             ↓
             Frozen source-only min/max predicates
                             ↓
          Existing prospective study and native observer
                             ↓
        Existing development mean / joint16 covariance
                             ↓
        Category-local calibration, retained failures
                             ↓
      Recomputed query groups → bound source qA and qB
                             ↓
                    Existing CanonicalState
                ┌────────────┼─────────────┐
                ↓            ↓             ↓
          Observed geometry  Error sets   Intrinsic shape
                └────────────┼─────────────┘
                             ↓
            Raw arrays + source PNGs + execution journal
                             ↓
                   Verify / recompute / report
```

No analyzer selects another alignment or changes source coordinates to improve agreement. All 135 observed-state arrays checked in the controlled pair remain unchanged when qualification is attached. Pair outputs retain 191 arrays for the demonstrated one-member A/two-member B comparison. Exact analysis grids, all min/max values and every rule outcome are retained beside the original pixels.

Unmatched, overlapping or explicitly forbidden rules cannot borrow a pooled radius, even with many calibration observations. Missing or inconsistent source inventory refuses the calibrated query before native model initialization. Group-wide measurement does not attest that a curator disclosed every real-world repeat or chose an honest aperture.

## Using original portraits

Read [the operator guide](docs/WAVE9_OPERATIONS.md). Prepare independent reference annotations and a pinned compatible local model. Select source predicates and grouping on development data, freeze them before calibration/evaluation, and assess both delivered availability and held-out reference coverage. The optional native MediaPipe path has not been executed in this release environment because its package/model were unavailable.

```shell
python -m facemath acquisition-group query_a.group.json --policy qualification/localization_calibration.json --out query/a-source
python -m facemath acquisition-group query_b.group.json --policy qualification/localization_calibration.json --out query/b-source
python -m facemath analyze portrait_a.png portrait_b.png --observation-a a.observation.json --observation-b b.observation.json --localization-calibration qualification/localization_calibration.json --domain "exact frozen domain" --acquisition-a query/a-source --acquisition-b query/b-source --out pairs/run
```

Existing nominal analysis and older calibration types retain their prior behavior. Outputs are create-only. Replay uses captured observations with the corresponding code/runtime, not a new detector invocation. Do not claim arbitrary-platform bitwise reproducibility from tests on one numerical dependency stack.

## Documentation and release evidence

- [Design and development audit](docs/WAVE9_DESIGN.md)
- [Exact mathematical and pixel-measurement definitions](docs/WAVE9_MATHEMATICS.md)
- [Operator workflow and complete query-group contract](docs/WAVE9_OPERATIONS.md)
- [Raw array inventory](docs/RAW_ARRAYS.md)
- `evidence/WAVE9_VALIDATION.json`: captured executions, source/wheel results and limits
- `evidence/wave9/laboratory`: original synthetic inputs, references, acquisition receipts, calibrated model, qualified pairs and refused-source pixels

The historical mathematics section registry remains an inventory, not proof that every mathematical analogy from the original conversation is implemented. No human-face similarity percentage is fabricated by these releases.
