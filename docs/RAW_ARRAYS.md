# Reading the Wave 04 numerical records

Use `numpy.load(path, allow_pickle=False)`. Array names are exact; shapes and SHA-256 values are recorded in each bundle's `manifest.json`. A number outside its associated support mask is not an observed zero error.

## Validation bundle

Axis 0 follows `validation.json → cases` and `study.lock.json → cases`. The eight-point axis follows `facemath.observation.reference.REFERENCE_NAMES` in its fixed order.

| Array | Axes | Meaning |
|---|---|---|
| `reference_xy` | case, point, xy | Independent reference consensus in source pixels |
| `predicted_xy` | case, point, xy | Observer location, never fitted to reference |
| `covariance` | case, point, xy, xy | Observer repeatability covariance plus declared noise floor |
| `reference_visible` | case, point | Eligible reference labels under minimum-rater policy |
| `observed` | case, point | Eligible reference AND supported prediction |
| `error_vectors_px` | case, point, xy | Prediction minus reference, on observed support |
| `error_norm_px` | case, point | Source-pixel Euclidean error, on observed support |
| `mahalanobis` | case, point | Covariance-normalized reference error, not a calibrated probability |
| `rater_count` | case, point | Number of visible rater labels |
| `rater_disagreement_px` | case, point | RMS disagreement; not an agreement claim for a single rater |
| `reference_normalization_span_px` | case | Independently labeled eye-corner midpoint distance |

A failed observation has explicit false support and zero sentinels in numerical arrays; its JSON error/score is `null` with a failure reason. Group calibration maxima and finite-sample quantile are in `localization_calibration.json`. Infinity is explicit `null` there, not an absent sample.

## Qualified-pair envelope

These arrays append to the existing shared-state bundle; they do not replace its geometry.

| Array | Axes | Meaning |
|---|---|---|
| `localization_source_radii_px` | A/B, reference point | Conservative enclosing-circle radii in original pixels |
| `localization_canonical_radii_px` | A/B, reference point | Those radii mapped into the observed canonical frame |
| `localization_canonical_points` | A/B, reference point, xy | Source predictions transformed by the same state used by geometry |
| `localization_radial_budget_ipd` | reference point | Sum of canonical radii divided by observed normalizer |
| `localization_observed_gap_ipd` | reference point | Observed positional disagreement |
| `localization_lower_ipd`, `localization_upper_ipd` | reference point | Conservative bounds for the fixed-gauge reference gap |
| `localization_supported` | reference point | Both reference-mapped predictions observable |
| `localization_ellipse_normalized_covariances` | A/B, reference point, xy, xy | Full covariance pushforward, in squared normalized units |
| `localization_ellipse_normals` | angular direction, xy | 128 evenly spaced unit directions; this is not a landmark axis |
| `localization_ellipse_combined_support` | reference point, angular direction | Sum of the two calibrated ellipses' support functions |
| `localization_ellipse_lower_ipd`, `localization_ellipse_upper_ipd` | reference point | Directional/circle-intersected bounds before the final support mask |

The final support mask governs admissibility, including for diagnostic pre-mask quantities. Bounds refer to the eight reference labels, not to unvalidated 478-vertex coordinates. Full covariance transformation preserves orientation, while the angular upper bound includes the `cos(pi/128)` correction described in `WAVE4_DESIGN.md`.

## Image-edge profiles

The `edge_a_*` and `edge_b_*` namespaces identify the source member and anatomical contour (`left_eye`, `right_eye`, `mouth`, `jaw`). Points/normals index sampled rays; response/orientation/source-support arrays index ray and search offset. Accepted flags, strengths, ambiguity and selected offsets remain separate. Negative/positive offset directions follow the exported unit normal.

These are landmark-guided gradients, not independent contour ground truth. An excluded ray or zero gradient cannot become a confident match. The underlying observations are never moved to the selected edge.

## Provenance

The validation and pair journals record executed named stages. Replaying saved observations does not rerun ML inference; this distinction is in the journal and command descriptions. Hashes can detect changed bytes relative to the recorded manifest, not establish an external author, truly independent raters or unreported duplicate subjects.


## Wave 05 qualification arrays

Case order is `qualification.json.array_order.cases`; point order is `reference_points` (the fixed reference8 grammar).

| Name | Shape | Unit / meaning |
|---|---|---|
| raw_error_vectors_px | cases × 8 × 2 | prediction minus independent reference, stored source px |
| source_bias_vectors_px | cases × 8 × 2 | development-fit model error, in observed ocular frame |
| centered_error_vectors_px | cases × 8 × 2 | raw minus modeled bias, not changed detector outputs |
| centered_error_valid | cases × 8 | explicit support mask; a zero sentinel is not zero error |
| source_predicted_xy | cases × 8 × 2 | untouched source prediction coordinates |
| source_reference_xy | cases × 8 × 2 | independently supplied reference coordinates |
| covariance_px2 | cases × 8 × 2 × 2 | unchanged observer covariance in source pixel squared units |
| development_group_bias_vectors | fitted groups × 8 × 2 | equal-group inputs to the frozen beta; dimensionless ocular units |
| development_bias_beta | 8 × 2 | fitted group mean in observed ocular vector coordinates |

The last two arrays exist only for `development_ocular`. Nested `raw_validation/arrays.npz` retains all original Wave 04 per-case arrays and mask semantics. Native request/response/stdout/stderr and receipts remain in `acquisition/cases/case_XXXXX/`.

### Additional qualified-pair arrays

`localization_source_bias_px` and `localization_canonical_bias_px` are 2 × 8 × 2 vectors before/after the observed transform. `localization_reference_center_points` stores the two shifted reference-set centers in canonical pixels. `localization_model_centered_gap_ipd` stores their normalized separation (8 values). Original `localization_observed_gap_ipd` remains the raw observed gap. All earlier support masks, full covariance matrices and directional bounds remain retained. No array supplies dense478 reference accuracy.

## Wave 06 — shared intrinsic reference8 state

The `intrinsic_` arrays are appended to the same pair bundle. Source index is A=0/B=1. Reference index is the exact `reference_names` ordering in `intrinsic.json`; relation index is its fixed 256-row inventory. The active implementation adds **23 arrays**:

| Array suffix (prefix `intrinsic_`) | Shape | Meaning |
|---|---|---|
| `raw_points`, `source_centers` | (2,8,2) | Observed coordinates vs error-set centers in each observed source-eye chart; do not interchange |
| `source_covariance` | (2,8,2,2) | Full observation covariance transported into that chart |
| `source_chart` | (2,3,3) | Stored raster pixels → observed source-eye chart |
| `source_supported` | (2,8) bool | Observable source points |
| `nominal`, `centered` | (2,256) | Raw or model-center relation values, in each row's units |
| `nominal_valid`, `centered_valid` | (2,256) bool | Validity of those point estimates |
| `source_lower`, `source_upper` | (2,256) | Conservative source relation bounds |
| `source_available` | (2,256) bool | Source bounds exist; zero sentinel alone is not an observation |
| `eye_span_bounds` | (2,2) | Each reference eye-span outer range, in the observed source chart |
| `eye_span_available` | (2,) bool | Eye-span range exists; positive lower bound is separately required for division |
| `gap_lower`, `gap_upper` | (256,) | Absolute inter-source relation-gap bounds |
| `gap_available` | (256,) bool | A finite relation-gap bound exists |
| `statuses` | (256,) int | 0=no calibration, 1=within, 2=outside, 3=unresolved, 4=unbounded/unobserved |
| `relation_a`, `relation_b` | (256,8) | Shared linear-expression coefficient vectors; dependent landmark appearances are not separate errors |
| `shape_deviation_radii` | (2,8) | Bounds on recentered, true-eye-normalized point displacement from the set-center shape |
| `shape_candidate_rotation` | (2,2) | Proper row-vector rotation for the nominal set-center shapes |
| `support_normals` | (M,2), M=128 default | Actual sampled support directions, with finite-angle correction used by the bounds |

Inspect the JSON's reason, units and calibration-bound-state fields before interpreting numerical sentinels. The original raw arrays and the observed-only intrinsic arrays must remain unchanged when a calibration is attached. All profile arrays own immutable storage in memory; NPZ is a persisted copy.

The separate set laboratory's `controls.npz` records the actual sampled points, scalar relation values, independent refit values, source error sets and bounds. Those samples are not independent human observations or the source of the analytic enclosure.

## Wave 07 · joint error geometry

All coordinates below use the same source charts as the existing intrinsic profile. The original `intrinsic_source_covariance[2,8,2,2]` is still **raw perturbation covariance**, not the joint model's error matrix.

| NPZ key | Shape | Interpretation |
|---|---|---|
| `intrinsic_joint_covariance` | [2,16,16] | Full transported score-shape matrix; interleaved reference8 x,y |
| `intrinsic_joint_marginal_covariance` | [2,8,2,2] | Diagonal point blocks of that same full matrix |
| `intrinsic_joint_marginal_lower` / `upper` | [2,256] | Source relation bounds for same-q marginal-product relaxation |
| `intrinsic_joint_marginal_available` | [2,256] bool | False entries are unavailable sentinels, not zero error |
| `intrinsic_joint_mode_eigenvalues` | [2,16] | Descending covariance eigenvalues in squared source-chart units |
| `intrinsic_joint_mode_vectors` | [2,16,16] | Column eigenvectors, sign fixed by largest-magnitude entry |

The displayed coherent mode is `center + amplitude * q * sqrt(eigenvalue) * eigenvector`, with amplitude between -1 and 1. Zero eigenvalues and tied subspaces require their ordinary linear-algebra interpretation, not anatomical labels.

The qualification bundle adds `development_joint_group_second_moments[G,16,16]`, `development_joint_covariance[16,16]`, and `development_joint_group_case_counts[G]`, alongside retained group bias vectors and beta. Their ocular-frame units and regularization are specified in `WAVE7_MATHEMATICS.md`. Raw per-case prediction errors, reference coordinates and covariance remain separate.

The set laboratory's `controls.npz` stores both unit-ball latent draws and their resulting [sample,8,2] source coordinates, factors, full matrices, exact scalar values, reported outer bounds and independently evaluated proper-rotation discrepancies. Latent draws are mathematical control inputs, not captured neural-network activations.

## Wave 08 additions: ordered source radii and category provenance

Pair bundles add `localization_source_quantiles` and `intrinsic_source_quantiles`, each float64 `[2]` in source A/B order, plus `intrinsic_source_quantile_available` boolean `[2]`. Missing intrinsic radii use zero sentinels with false masks. The authoritative available-state field is explicit; a null legacy common scalar merely means A and B have different radii when both source masks are true.

Study qualification arrays add:

| Name | Shape / interpretation |
|---|---|
| `routing_category_per_case` | int64 `[N_cases]`; category indices in frozen policy order; case IDs in `array_order.cases` |
| `routing_category_quantiles` | float64 `[N_categories]`; zero sentinel when unavailable |
| `routing_category_available` | bool `[N_categories]`; selects actual finite radii |
| `routing_calibration_group_category` | int64 `[N_calibration_groups]`; category of each group score |
| `routing_calibration_group_score` | float64 `[N_calibration_groups]`; zero sentinel for infinite/unavailable score |
| `routing_calibration_group_score_available` | bool `[N_calibration_groups]`; false means retained failure, not zero error |

`acquisition_routing.group_array_order` binds group IDs to these columns. `localization_calibration.json` preserves the complete category inventories, selected ranks, policy and original shared joint model. In pair bundles it additionally binds both source observations, images, groups and lineages. No radius is encoded by silently modifying the covariance matrix.

`set_controls/controls.npz` contains the synthetic unequal-radius source centers, full16 covariance/factors, `source_radii`, unit-ball draws, shape draws, every exact scalar relation value/gap, bounding arrays and proper-rotation values. Shapes and factors use the same reference8 interleaved coordinate order. Its separate circle counterexample saves delta, source covariances, radii and an explicit admissible witness. This is a control dataset, not a native-observer evidence bundle.


## Wave 09: measured acquisition arrays and original source rasters

Each pair source has `acquisition_a_` or `acquisition_b_` arrays: `features` [members,9], `aggregate` [2,9] (min then max), `predicate_truth` [predicates], `predicate_offsets` [rules+1], `rule_matches` [rules], `apertures` [members,4], and `analyzed_NNNN` [analysis_height,analysis_width]. RGB source PNGs, complete policy JSON and measurements JSON are flat prefixed sidecars under the same manifest. Sidecar NPZ uses the common `acquisition_` prefix; the main NPZ distinguishes A and B. Members are ordered by their decoded pixel digest, not by caller order. Units and feature names are in each source measurement JSON.

Study qualification adds `acquisition_member_features`, `acquisition_group_aggregates`, `acquisition_member_group_index`, and `acquisition_group_rule_matches`. Group order and member order are explicitly represented in `measured_acquisition.groups` in qualification JSON. This inventory includes prospectively observation-excluded enrollments; these are not silently added to the detector-attempt denominator.

Values/masks are not invented zeros for missing source data: invalid apertures, absent members and unsupported routes fail their contracts. Code-value fractions and gradient detail are signal diagnostics, never similarity percentages. Visual previews quantize the exact exported float64 grids for display only.
