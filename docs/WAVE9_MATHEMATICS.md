# Wave 09 mathematics and numerical contract

## 1. Source-only measurement

For a declared uint8 RGB aperture, define a code-value scalar field

    Y(x,y) = (299 R(x,y) + 587 G(x,y) + 114 B(x,y)) / 255000.

The integer-weighted numerator is exactly representable in float64 for uint8 inputs. This is NOT linear-light physical luminance; there is no transfer-function or color-profile correction. Code values 1 and 254 on a neutral raster are included at their configured tail thresholds.

Native-aperture measurements are minimum edge length, pixel area, mean Y, the 5th and 95th percentiles using the linear quantile convention, their difference, and fractions Y <= low_code and Y >= high_code. All eight are computed before analysis-grid reduction.

Only spatial detail uses a reduced grid J. It is reduced with OpenCV AREA interpolation to analysis_max_side while preserving rounded aspect ratio and never enlarged. Extremely elongated apertures requiring fewer than three analyzed pixels on the short side are rejected rather than geometrically distorted. If no reduction is needed, J is a copy of Y.

    Kx = [[-1,0,1],[-2,0,2],[-1,0,1]] / 8
    Ky = transpose(Kx)
    dx = Kx * J
    dy = Ky * J
    detail = mean(dx^2 + dy^2) / (var(J_interior) + variance_floor)

Only the interior with a full 3x3 neighborhood is included; the outermost row/column is excluded. Derivatives use float64, not uint8 (which cannot retain signed values) [1]. The full exact J array is exported, not just a thumbnail. The HTML thumbnail is a display quantization of that stored array.

“detail” has analysis-pixel gradient-energy/variance units. It is not calibrated blur, pose, visibility, likelihood of a face, localization error, or a learned similarity metric. A flat raster gives zero. Additive image noise can increase it. The negative-control noise raster is retained in NPZ.

## 2. Whole-group reduction and simultaneous predicates

Let f(I_j,A_j) in R^9 be the measurements for group member j under aperture A_j. Compute

    m_k = min_j f_k(I_j,A_j)
    M_k = max_j f_k(I_j,A_j)
    a = [m_1,...,m_9,M_1,...,M_9].

Every member in the declared enrolled group contributes. The max member count is a frozen protocol constraint. Sorting by pixel digest makes the proof's member order deterministic without changing the extrema.

A rule r is a conjunction

    R_r(a) = AND_l [a_index(r,l) op(r,l) threshold(r,l)]

with op in {<, <=, >, >=}. All rules are evaluated; there is no priority or nearest-rule distance. Let B = {r : R_r(a)}. Route to the single category if |B|=1, otherwise withhold under the explicit unmatched or ambiguous category. Exact predicates, thresholds, truth values, offsets and matches are exported.

Outside-aperture changes cannot affect f when the aperture pixels are unchanged. This is tested with byte-different full rasters and equal feature arrays. Replacing a clear-only group by a group containing both clear and smoothed sources can change min.detail and thus the route. A selected source's individual feature values do not replace the group extrema.

## 3. Existing calibration, now with a reproducible category map

The development-only ocular-frame bias and full16 scatter are unchanged. The frozen calibration score remains the group maximum of each case's complete reference8 Mahalanobis residual. For a measured category c:

    k_c = ceil((n_c + 1) * (1-alpha))
    q_c = kth sorted category group score,
          unavailable if rank exceeds n_c, selected score is infinite,
          minimum group policy fails, or measured rule forbids deployment.

The study retains every category, including rejected categories with enough collected calibration groups. Their scores are not discarded and their q is not silently borrowed from a permitted category.

Conditional on a frozen development model and feature partition, the usual category/group rank argument is applicable to exchangeable group scores [2]. Conditioning does not create exchangeability. Comparable enrollment/group composition remains necessary. This release does not infer a posterior confidence per image, detect arbitrary distribution shift, or fit predicates to held-out residuals.

## 4. Pair propagation is unchanged, not rescaled to favor agreement

Recompute c_A, c_B from each complete source group. The existing source ellipsoids use q_A=q(c_A), q_B=q(c_B) and their original covariance matrices. For directional support,

    h(u) = q_A sqrt(u^T S_A u) + q_B sqrt(u^T S_B u).

These remain the radii used by fixed-gauge reference error bounds, joint intrinsic expression graphs, uncertain reference eye normalization and proper-rotation-refitted shape ranges. Wave 09 does not introduce new observed coordinates, covariance scaling, reference labels at query time, or another alignment. It records which source calculation selected each already-defined radius.

## 5. Serialization and recomputation

Digests bind pixels, aperture, policy and group inventory. They establish content consistency, not external truth. Standalone group and pair verification remeasure pixels and require exact equality with JSON and NPZ source metrics. Altering stored feature values and recomputing their hashes cannot make them agree with unchanged pixels.

Exact numerical replay uses the matching code and numerical runtime. Other OpenCV implementations can differ in low floating-point bits; a digest is not a claim of bitwise portability to untested environments. No measurements are rounded before routing or CSV export. The user-facing table rounds display strings only.

## Primary sources

[1] OpenCV 4.13 image-gradient tutorial and Sobel documentation. The fixed coefficient, aperture and group-policy recipe above is FACEMATH's declared design, not a validated OpenCV face-quality method. https://docs.opencv.org/4.13.0/d5/d0f/tutorial_py_gradients.html

[2] Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification, arXiv:2107.07511. See group-balanced/conditional coverage and distribution-shift limitations. https://arxiv.org/html/2107.07511v6

[3] Google AI Edge FaceLandmarker Python guide: the optional inference path requires the installed package and a compatible local model. No live execution is claimed when they are absent. https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

Sources accessed 2026-09-20. No model weights, third-party portrait datasets or human-accuracy results are supplied.
