# Wave 07 · Keep shared errors shared

## Priority

The starting point is the actual Wave 06 release archive, not a recreated code tree. The parent was installed and its full test suite executed before modification. The live observation gate remains open: the local runtime has no MediaPipe package and requests to the package/model hosts failed DNS resolution. `evidence/wave7/native_environment.json` retains those actual results.

This wave closes a distinct numerical gap. Wave 06 bounded refitted geometry using products of marginal localization ellipses. That is safe as an outer approximation but loses cross-landmark covariance. A coherent translation can become eight separate possible deformations. Wave 07 models an entire reference8 error field, while keeping source-coordinate accuracy validation separate from relative shape.

## Architectural decision

There is no second face analyzer. `JointCalibration` extends the existing `BiasCalibration` and `ErrorCalibration` contracts. The existing `validation_io` remains the only reference evaluator. The frozen study selects the joint estimator and its regularization before any observation. The existing acquisition runner still handles receipts, failure outcomes, interruption and resume.

The source-coordinate rows feed the development-only mean and scatter. Calibration rows set the full-shape radius. `FacialInstrument` uses the unchanged `CanonicalState`; the ordinary point bounds use marginal projections of the new set, and the existing intrinsic expression graph uses the full16 covariance. All results return to the same bundle, report, journal, verification and replay paths.

## Contracts that matter

* The development second moment averages case outer products within each group, then averages groups equally. Within-group variation cannot disappear when errors average to zero.
* Fitting scatter rechecks the bound development bias against the actual rows, not just matching their IDs. Evaluation cannot change beta, scatter, regularization, or calibration q.
* Missing and failed complete-shape observations remain unavailable/infinite calibration scores. The scalar joint score is not replaced with easier marginal scores.
* The full16 covariance is transported with the observed ocular frame, including its off-diagonal blocks. It is not confused with the original perturbation covariance.
* A joint calibration has a distinct v7 wrapper, score identity, scatter digest and bias digest. A detached residual fragment is rejected by older loaders instead of being accidentally applied as a zero-centered marginal model. A malicious relabeling of all metadata is not prevented by self-signed hashes; replay of retained raw inputs remains the stronger consistency check.
* The same-set marginal relaxation is diagnostic. It uses the same center and q and is not a separately fitted Wave 06 baseline. Its interval width cannot be read as a population-calibration comparison.
* Proper-rotation shape bounds retain uncertain recentering and reference eye spans. No reflected alignment or new scale fit is added. All original geometric observations remain unmodified.

## Workbench

A joint-error section is appended to the existing intrinsic report. It reads exported eigenvalues, eigenvectors, centers and covariance. A mode slider displays one admitted coherent displacement; it does not recompute any result. Correlation normalization is a display of the stored covariance, not a new similarity score. Variance fractions are explicitly labeled as modeled error variance.

The original ellipse view now renders the selected error model's marginal covariances. It must not multiply q by the much smaller raw jitter covariance merely because that array already existed in the report. The old raw covariance remains exported unchanged.

The mode sign convention is deterministic; degeneracy still permits nonunique eigenbases. Modes are coordinates of a fitted set, not established physical causes of detector failure.

## Executed falsification design

The default experiment enrolls 97 schematic sources: 32 development, 40 calibration, 24 evaluation, and one excluded source. Two nondetections are prescribed. The first bounded batch commits nine attempts; completion commits 96; resuming completion adds none. The final third of evaluation receives a deliberately unseen local error shift. Its failure is retained rather than tuning the model on it.

A separate source supplies each of four prespecified mouth-deformation queries: 0, 4, 8 and 12 pixels per side. Their render coordinates remain ground truth only for this synthetic control. A separate set laboratory retains 1,024 full16 unit-ball draws per source, all 256 exact scalar relationships, and proper-rotation values.

No learned model, human annotation or population face-similarity result is substituted for the missing live gate.

## Faults caught during development

1. The inherited CLI reported a hard-coded `0.6.0` while the package was `0.7.0`. The first full source run exposed it. The CLI now reads the package's single version definition.
2. The new replay tests initially asserted an invented return key `replayed` instead of the actual API's `replay_verified`. Execution had succeeded; the test harness was corrected to the real contract. Logs are retained.
3. The display initially still selected raw jitter covariance. A targeted integration assertion and the report change now bind displayed ellipses to the selected error model's marginal blocks.
4. Regularization now rejects nonnumeric, boolean, nonfinite, and overflowing values with explicit observation-contract errors. No NaN covariance is exported.

No inherited numerical assertion was relaxed to obtain a passing result. The historical section-number registry remains an inventory rather than mathematical-completeness evidence.

## Next priorities

Live observation and independent real references remain first. Once actual localization rows exist, compare frozen zero-centered, centered marginal and joint models on appropriate held-out studies; do not select a model by final evaluation performance. Evaluate whether the joint set's narrower geometric ranges retain reference coverage and whether covariances change under acquisition shift. Only then expand annotation coverage, contours, visibility, or 3D.

The current bounds have floating-point allowances, not formal interval arithmetic. They are outer enclosures, not exact ellipsoid-constrained extrema. Tightening them further without real reference data is lower priority than validating the source observation.
