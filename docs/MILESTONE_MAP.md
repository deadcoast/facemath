# Consolidated instrument map · Wave 10

| Stage | Owning implementation | Shared evidence / invariant |
|---|---|---|
| Source decoding | `study.load_rgb` | Stored raster; Unicode-safe byte decode; EXIF display rotation excluded |
| Source acquisition | `acquisition.ApertureSource`, `MeasuredGroup` | All declared members, fixed apertures, source-only rules; no label-only authorization |
| Annotation preparation | `workspace`, existing `annotation_ui` | Blank workbenches; no automatic human labels; immutable sources |
| Annotation reference | `reference`, `workspace` | Source-bound reference8, rater/visibility/provenance, retained revision history |
| Study freeze | `study.freeze_study` | Entire grouping, native recipe, selected estimator, acquisition policy and acceptance criteria frozen |
| Observation execution | `study_runner`, `worker`, `provider` | One committed attempt; actual process outputs, failures and no retry-until-success |
| Native smoke | `native_check` | Uses that same executor; preflight is not an observation or accuracy result |
| Reference evaluation | `validation_io`, `validation` | Source coordinates, not aligned-to-label error; failures retained |
| Localization mean/scatter | `bias`, `joint` | Development only; equal-group weighting; full16 correlation retained |
| Category radii | `stratified` | Calibration only; no pooled fallback for unsupported categories |
| Empirical acceptance | `admission` | Complete held-out groups; exact lower bounds; all requested categories pass |
| Deployable evidence context | `deployment.ValidatedQualification` | Full raw qualification replay; evaluation/context leakage forbidden |
| Geometric state | `instrument`, `canonical`, `field` | One unchanged CanonicalState, proper registration, masks and deformations |
| Fixed-gauge qualification | `error_model` | Original observed transform retained; distinct source radii |
| Intrinsic shape qualification | `intrinsic_sets`, `intrinsic` | Shared reference8 variables; uncertain normalization and refitting |
| Evidence export | `artifacts`, `study_io` | Original pixels, immutable arrays, actual journals, manifests and replay |
| Presentation | `readout`, existing `*_report` | Displays existing data; no new score computation; null is unavailable |

The 28 lengths, 168 angles, 56 signed areas and 4 bilateral contrasts describe the same eight points. They are never independent votes. Proper-rotation shape RMS is separately defined. No single universal resemblance percentage is computed.

## Mathematical ownership versus historical inventory

The old research modules remain available for signal, topology, temporal, surface and representation experiments. Some are exact primitives; some are proxies; some require external data/model telemetry. `facemath capabilities` and `WAVE3_AUDIT.md` record the non-equivalences. Wave10 does not transform those research interfaces into validated native measurement merely by preserving their names.

## Release evidence layers

1. Analytic unit controls test specified numerical identities.
2. Synthetic full-pipeline controls test data flow, failure accounting, raw-state invariance and gates.
3. Native smoke records an actual executable outcome or honest preflight failure.
4. A group-disjoint real-reference study remains necessary for empirical human accuracy.
5. Distribution-shift and deployment monitoring remain separate from all preceding layers.

A successful layer is not a substitute for the next. The milestone combines the existing components; it does not hide missing evidence with another formula inventory.
