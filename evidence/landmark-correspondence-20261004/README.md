# Landmark correspondence repair evidence

`source-audit-final.json` reports the final native study and exact raw-array checks.
All 60 sources produced observations; one development case had incomplete
in-frame reference support and remains in the enrollment. The model fit used
19 of 20 development groups. Calibration and evaluation each retain 20 groups.

Held-out nose error: median 0.120262 eye spans for the historical MP1 assignment,
0.052549 for the prespecified MP4 assignment, on the same native outputs.
The fixed 0.05-threshold counts are 2/20 and 10/20 respectively. Chin remains
7/20 within that threshold, with median error 0.070121 eye spans.

The distinct joint model covers 19/20 evaluation groups. Its median marginal
bound diameter is 0.753717 eye spans, maximum 2.526403; these broad bounds are
not evidence of useful precision or empirical query admission. Model parameters,
thresholds and enrollment were not changed in response to evaluation outcomes.

- `plan.json`, `registry.json`, `prospective-study.json`: frozen plan, prior
  enrollment and declaration recorded before the native execution.
- `intake/dataset/`: pinned original photographs, PTS, conversion records,
  source selection and native model bytes.
- `model-final.json`: the complete newly fitted model and donor enrollment.
- `native-pair-final/calibrated/report.html`: actual full 478-point pair result with
  flow, edges, intrinsic relations and 96 sensitivity draws enabled.
- `native-pair-final/invariance.json`: all 142 raw geometry arrays unchanged when
  adding the research model; `native-pair-final/replay/` retains its exact replay.
- `historical-preservation.json`, `legacy-model-refusal.json`: old observation
  byte/digest preservation and refusal to reuse the old calibration.
- `next-registry.json`: all 315 enrolled cases, including this study. Repeated
  wheel/archive executions do not add independent portraits.
- `execution/`: raw baseline, reproduction, command, failure and native audit
  records. The failed stale-environment baseline is distinct from the successful
  supported-environment baseline.
- `preseal-final/`: source tests, milestone controls, native pipeline and installed
  wheel checks, copied only after they complete. Archive validation is recorded
  beside the sealed archive and in the final root `LANDMARK_REPAIR.json`.

The first `source-audit.json`, `model.json` and `native-pair/` are superseded by
the explicitly named final artifacts. That run exposed a false human-validation
flag for external datasets. `execution/qualification-before-provenance-fix.json`
retains the original report. The final run corrects provenance reporting with
identical plan/settings and already known outcomes; it is not a new holdout.

The original root diagnosis is unchanged. Published PTS agreement does not
establish anatomical equivalence, visibility, person independence or native
training separation. No human testing or full-mesh anatomical validation is
claimed. The historical mathematics inventory remains a separate open scope.
