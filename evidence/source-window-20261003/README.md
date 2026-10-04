# Source-window validation evidence

This is a distinct 0.15.0 experiment. Historical donor, IBUG and failed runs are
unchanged. The frozen plan selects HELEN photograph-root ranks 60 through 119
using the original donor seed; it does not refit the saved model.

- `prospective-freeze.json`: pre-inference plan/model pins, full selection and
  original encoded/decoded source and group overlap checks against prior cohorts.
- `dataset/`: portable original images, PTS annotations, model, conversions,
  complete candidate inventory, selection window and all 60 enrolled cases.
- `helen-model.json`: byte-identical retained 0.14 model capsule.
- `development/`: parent baseline, focused tests, commands, raw logs and the
  exact helper scripts used from `runs/transfer-holdout-20261003`. These helpers
  retain their original run-relative paths and are forensic records, not CLI
  entry points to invoke from this evidence directory.
- `preseal/`: source and installed-wheel checks available when the archive was
  sealed, including source native observations, finalization and exact replay.
- `postseal/`: final archive and installed-wheel evidence retained after sealing.
- `VALIDATION.json`: completed host/package/scientific status and next gates.

The sealed archive contains the evidence available at sealing. Later archive
execution and its validation report are adjacent records; the archive is never
rewritten to insert its own result or digest. The root `CONTINUATION_REVIEW.json`
is the current development handoff, separate from the historical review.

Published PTS agreement does not establish anatomical accuracy, visibility,
person independence or absence from native-model training. Repeated native
executions test observed repeatability. They are not additional independent
portrait enrollment or a population-coverage guarantee.
