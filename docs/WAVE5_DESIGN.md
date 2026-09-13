# Wave 05 — frozen acquisition and development-qualified error geometry

## Priority and executed boundary

This wave extends the actual Wave 04 package. The next scientific gate remains actual native observations against independently annotated, group-disjoint real portraits. This environment has no installed MediaPipe and cannot resolve the required package/model hosts. `evidence/wave5/native_environment.json` records the executed check. No live landmark output, human accuracy measurement, or human-face similarity percentage is manufactured in its place.

The development contribution is operational and numerical: a prospective, resumable acquisition instrument; a fixed development-only localization-bias model; independent residual calibration; prespecified held-out strata; and error-center propagation into the existing canonical geometry. This is not an identity recognition system.

## One state, not a second measurement pipeline

The new runner supplies the same immutable `Observation` objects used by Wave 04. `validation_io` remains the only source-coordinate reference evaluator. Its `LocalizationRow` records feed the new bias/calibration layer. `ErrorCalibration` is the shared structural interface for zero-centered and learned-center error models. Both ultimately qualify the same `CanonicalState`, with the same rasters, original coordinates, registration, normalizer, field and raw measured profile.

The selected model changes an **error-set center**, not the displayed face or the geometric measurements. It is impossible for a smaller calibrated residual score to retroactively become a better raw detector coordinate in the active path. The lab compares all unqualified pair arrays byte-for-byte against the corresponding qualified arrays.

## 1. Freeze before acquisition

`freeze_study` creates a self-contained study directory. It freezes:

- decoded source PNGs, original encoded-file digests, image-bound independent references;
- development/calibration/evaluation membership, curator groups and declared acquisition lineages;
- enrolled and prospectively excluded cases, reasons, maximum enrolled group size;
- acquisition strata, alpha, minimum-rater policy, minimum development groups;
- estimator selection (`none` or `development_ocular`), observer/model/runtime hashes;
- augmentation count, noise floor, per-source timeout, fresh-process isolation mode;
- source-code digest and the reference-coordinate protocol.

References, groups, lineages and split consistency are checked before constructing a detector. Predictions and previously observed failures cannot be imported into a prospective lock. The legacy retrospective/import workflow remains explicitly named separately.

SHA-256 binds a record; it is not an externally trusted timestamp and cannot prove who annotated it or when they saw predictions. Curator group IDs are supplied metadata, never inferred identities. New IDs cannot prove independent people. Group exchangeability is not established by schema validation.

## 2. Stored-raster coordinates everywhere

Studies, CLI annotations, legacy validation and pair queries share one image decoder. It uses byte-based `imdecode`, Unicode-safe paths and explicit `IMREAD_IGNORE_ORIENTATION`. The numerical coordinate frame is the stored decoded raster, not an EXIF-adjusted display. Source PNG snapshots remove ambiguity during replay. An EXIF rotation regression test checks dimensions, pixel values and equality across the three entry points.

Previously exported observations bound to different decoded pixels must be reacquired/re-exported, not silently rotated into compatibility.

## 3. One committed outcome per source

Each case has a create-only start record and an atomically published receipt containing byte inventories. Terminal outcomes are `observed`, `no_face`, `multiple_faces`, `timeout`, `interrupted`, `runtime_error`, and `contract_error`.

Completed cases are never reinvoked on resume. A prior started-but-uncommitted case becomes `interrupted`; partial outputs are retained but never promoted to a successful result. A bounded batch is not a changed study. Repeating a completed invocation is an idempotent verification operation. There is no "retry until detected" policy.

An operating-system writer lock prevents simultaneous collectors. The file may remain after a crash; the OS lock, not file existence, determines ownership. Contract errors stop remaining collection and prohibit qualification. Infrastructure failures stay in availability/error denominators but are not relabeled no-face detections. Missing runtime/model preflight failures create no fictional attempted cases.

The native executor pins the installed MediaPipe version and local model digest, starts a fresh Python worker for each source, bounds execution time, and preserves argv, request, response, process exit, stdout and stderr. The worker's request contains the raster path and observation recipe, not labels, split, or stratum. This input separation is **not a filesystem or network sandbox**. Native numerical-library versions and the frozen code are checked. The timeout cannot be changed when a session resumes.

## 4. Development-only ocular-frame bias

Let eight observed source coordinates be p_i and independent reference coordinates be r_i. For each observed face define the eye-midpoint vector

    v = mean(p_2,p_3) - mean(p_0,p_1)
    E(p) = [ v , rot90(v) ]

E is a 2×2 basis with pixel units; its determinant is ||v||². The frame must have at least an eight-pixel eye span. It depends only on observable predictions, not reference locations at deployment.

For development case c, local error at landmark i is

    z_ci = inverse(E(p_c)) (p_ci - r_ci)

Cases within a declared group are averaged first. The final coefficient is an equal-weight average over fitted development groups:

    beta_i = (1/G) sum_g (1/n_g) sum_{c in g} z_ci

This is a small prespecified vector model, not a flexible regressor fit to held-out errors. It is covariant under an orientation-preserving global scale/rotation when the observations transform accordingly. Every contributing group vector and rejected development case is retained. Incomplete/failed development cases cannot be fitted; their exclusion from fitting is explicit accounting, not deletion from the study.

At a new image:

    b_i(p) = E(p) beta_i
    centered_error_i = (p_i - r_i) - b_i(p)

The raw `p`, raw error vector, raw NME, reference coordinates and detector covariance are preserved. A centered residual is not presented as a newly measured detector output. Development image hashes/groups/lineages cannot reappear in calibration, evaluation or a qualified query.

## 5. Residual calibration after a frozen fit

For covariance C_i from the unchanged observer recipe, the selected scalar score is

    s_case = max_i sqrt(centered_error_i^T inverse(C_i) centered_error_i)
    s_group = max_{cases in group} s_case
    k = ceil((n_calibration_groups + 1) (1 - alpha))

The kth order statistic supplies q. Rank beyond n or an infinite selected score remains unbounded. Failed/incomplete calibration cases contribute infinity, not an easy zero. Evaluation is not consulted when fitting beta, choosing the estimator, or computing q.

Conditional on the frozen development fit, the ordinary split-conformal rank argument applies to exchangeable group scores [1]. It does not require a correct bias model or Gaussian errors. It also does **not** guarantee coverage for a new acquisition distribution, each stratum, or each individual image. Repeated observations do not become independent by renaming them.

The exported residual fragment has a distinct `centered_residual_calibration.v5` format and binds the fitted bias digest. The v4 zero-centered loader rejects that fragment by itself. This prevents the smaller residual quantile from being detached and applied around raw predictions. The raw zero-centered calibration is reported as a diagnostic comparison. The program never selects whichever estimator scored best on evaluation. The freeze already selected the method.

## 6. Fixed-gauge pair propagation

In the observed canonical transforms T_A(x)=L_A x+t_A and T_B, reference-set centers become

    c_Ai = T_A(p_Ai) - L_A b_Ai
    c_Bi = T_B(p_Bi) - L_B b_Bi
    delta_i = (c_Ai-c_Bi)/d_observed

Covariances remain

    S_Ai = L_A C_Ai L_A^T / d_observed²
    S_Bi = L_B C_Bi L_B^T / d_observed²

The existing directional support algorithm receives delta and these full covariance matrices:

    h_i(u) = q sqrt(u^T S_Ai u) + q sqrt(u^T S_Bi u)

It produces conservative lower/upper gap bounds using 128 normals, the finite-angle cosine upper correction and enclosing-circle intersection. Raw observed gaps are reported separately from model-centered gaps. A raw gap need not lie in an interval about a reference-position gap; these are different quantities.

Transforms and normalization are fixed at the actual observed values. These are not confidence intervals for refitted true anatomy, population resemblance, all 478 vertices, or identity. The nominal two-source union bound remains conditional on the per-source calibration assumptions. An unsuccessful held-out study must not be hidden behind that nominal formula.

## 7. Slices expose generalization failure

Every frozen stratum receives calibration representation counts, eligible held-out counts, complete-group coverage and exact Clopper–Pearson binomial intervals. The binomial interpretation separately requires independent, comparable Bernoulli groups; marginal conformal exchangeability alone does not establish it. Overlapping within-group slices are not pooled as independent observations.

An evaluation stratum absent from calibration is flagged. Undercoverage is reported without changing alpha, deleting cases, enlarging the interval using evaluation labels, or declaring the smaller q a success. The included `unseen_shift` fixture is explicitly nonexchangeable stress data, not evidence against a valid exchangeable-data rank theorem.

## 8. Evidence and replay

The qualification directory contains the complete acquisition session, nested unchanged raw validation, selected error model, raw centered/uncentered arrays, all per-case records, strata and failure outcomes, offline HTML, and an execution journal. `verify` checks inventories, array contracts, source bindings, receipt consistency and nested bundles. `replay` recomputes the numerical reports from captured observations; it does not reinvoke the detector or claim to reproduce native nondeterminism.

The new report has local filters, case-to-source and outcome-to-receipt links, a selected-row CSV export with spreadsheet-formula escaping, and no remote assets. Native runtime privacy is separate: MediaPipe's official notice says input processing is on-device but its Tasks APIs send performance/utilization metrics [3]. This process wrapper does not certify or disable upstream network activity.

## 9. Scope and next gate

The 91-enrolled/90-attempt laboratory is a deliberately biased schematic observer with known render coordinates. It demonstrates resume idempotency, frozen exclusions, fitting separation, raw-state invariance and visible domain-shift failure. It is not live inference or a human accuracy benchmark. API fixtures and a real subprocess failure-path execution do not close the live-detector gate.

Next: run the frozen native pipeline on independently annotated real portraits, review reference8 point equivalence, preserve unobservable points and detector misses, and evaluate supported acquisition strata. Only then expand reference coverage to contours/visibility and the full mesh. None of the legacy 0–180 formula inventory is newly claimed complete by this wave.

## Primary references

[1] Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511. https://arxiv.org/html/2107.07511v6

[2] Google AI Edge, *Face landmark detection guide for Python*, updated August 17, 2026. IMAGE-mode `detect`, local compatible model, 478 landmarks, mode-specific tracking behavior. https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

[3] MediaPipe maintainers, official PyPI project and Privacy Notice (June 5, 2026). The page lists version 1.0.1 released August 14, 2026; release listing is not a compatibility test of this package. https://pypi.org/project/mediapipe/

Accessed September 11, 2026. No upstream model or dataset is redistributed.
