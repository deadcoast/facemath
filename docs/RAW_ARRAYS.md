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
