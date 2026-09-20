# Wave 07 — joint reference-error sets

## 1. The object being modeled

The active reference is eight two-dimensional vertices. Stack them in **interleaved x,y order** into a 16-vector. Nothing in this model identifies a person, estimates biological causation, or reads a language model's internal telemetry.

For an observation p and independent source reference r, use the existing observable eye frame E(p). Wave 05's equal-group development mean beta remains the error center:

    z_c = vec(E(p_c)^-1 (p_c - r_c) - beta)       [16]

Only eligible, complete development rows contribute to fitting. Their absence remains in enrollment and fit-rejection accounting. Neither calibration nor evaluation errors enter beta or the scatter estimate.

## 2. Equal-group scatter, not a claim of Gaussian errors

For each development group:

    M_g = (1/n_g) sum_{c in g} z_c z_c^T
    M   = (1/G) sum_g M_g
    S   = (1-lambda) M + (lambda trace(M)/16 + tau^2) I_16

Lambda and tau are fixed in the prospective study. Defaults are lambda=0.02 and tau=0.001 observed eye spans. They are engineering choices for the controlled development experiment, not human-calibrated optima.

We retain each **case residual outer product in its group average**, rather than taking only an outer product of each group mean. Two oppositely biased acquisitions in one group have zero mean but nonzero scatter. Groups receive equal weight despite different numbers of acquisitions.

S is a positive-definite **score-shape matrix**. There is no Bessel correction or unbiased population-covariance claim, no Gaussian posterior, and no assertion that the raw perturbation covariance estimates localization accuracy. The isotropic floor is in the ocular frame. Matrix reconstruction from retained group moments is checked on import, and the scatter is bound to the exact development-bias fit.

## 3. One score per complete shape, then one maximum per group

For a new case with a valid observed eye frame:

    score_case = || chol(S)^-1 z_case ||_2
    score_group = max_{cases in group} score_case
    rank = ceil((G_calibration+1)(1-alpha))

The selected order statistic is the radius q. Incomplete/failed calibration cases contribute infinity; an unavailable finite rank or infinite selected score stays unbounded. We do not replace it with the largest finite value. Development, calibration, evaluation and qualified-query source/group/lineage overlaps are rejected by the inherited study contracts.

Conditional on a frozen development fit and exchangeable calibration/new group scores, the ordinary split-conformal rank construction provides marginal new-group coverage. It does not require Gaussian errors or a correct fitted scatter. It also does not guarantee an individual case's probability, every acquisition slice, unreported deployment domains, or post-selection coverage [1]. The supplied shifted-condition experiment violates the stationary-distribution premise deliberately.

The full16 score is **not** the maximum of eight marginal Mahalanobis scores. Even a block-diagonal full16 ellipsoid is a single sum-of-squares budget, not a product of eight independently bounded balls. Consequently its q is not numerically comparable to a marginal-product calibration's q.

## 4. Source transport and the unchanged observation

With K=I_8 tensor E(p):

    center_reference = vec(p) - K vec(beta)
    C_source = K S K^T
    admitted_reference = center_reference + q L u
    L L^T = C_source, ||u||_2 <= 1

The sign of u is immaterial for a centered ellipsoid. This set describes plausible reference positions, not edited detector coordinates. Raw observations, samples, repeatability covariance, canonical registration, images, field and previous measurements remain unchanged.

The existing canonical transforms push forward all **cross-point covariance blocks**, not only the eight diagonals. The existing fixed-gauge point ranges use marginal projections of this same full set. Intrinsic relations use the whole set. Two sources have separate 16D sets; no independence between sources is asserted. The pair's union-bound coverage statement remains conditional on the two per-source statements.

## 5. Linear geometry retains coherent errors

For a zero-sum anatomical coefficient vector a over eight points:

    v(a) = sum_i a_i center_i
    A(a) = sum_i a_i L_i                  [2 x 16]
    v_reference(a) = v(a) + q A(a) u
    support(a,n) = n^T v(a) + q ||A(a)^T n||_2

The largest error radius is q times the spectral norm of A(a). A shared translation has identical L_i; it cancels whenever sum_i a_i=0. The eye vector, triangle arms and recentered vertices are all expressions in the **same cached SourceRelations graph**, so this cancellation occurs before normalization or shape bounds.

Finite directional support evaluation keeps Wave 06's enclosing-circle intersection and 1/cos(pi/M) upper correction. All bounds remain conservative floating-point calculations with explicit padding, not formally verified interval arithmetic.

## 6. Bilinear geometry uses the same latent variable twice

For two arms v+q Au and w+q Bu, write the dot or oriented cross product with matrix H (identity or the 2D skew matrix):

    f(u) = v^T H w + q ell^T u + q^2 u^T Q u
    ell = A^T H w + B^T H^T v
    Q = sym(A^T H B)

An outer interval over the unit ball is:

    lower = center - q ||ell|| + q^2 min(0, lambda_min(Q))
    upper = center + q ||ell|| + q^2 max(0, lambda_max(Q))

The linear and quadratic extrema need not occur at the same u, so this is an outer bound, not an exact quadratic trust-region solve. We intersect it with the valid marginal-product bound. Repeated-vector cross products cancel algebraically. Angles, signed areas and contrasts reuse the inherited validated normalizers and denominator-withholding rules.

## 7. Refitted shape discrepancy

Wave 06's metric is preserved:

    F(x) = (x - mean(x)) / eye_span(x)
    D(x,y) = min_{R in SO(2)} ||F(x) R - F(y)||_F / sqrt(8)

The sources' reference eye spans remain uncertain. The new shared-factor radii feed the existing recentering, reciprocal-normalizer and proper-rotation range calculation. Neither reflections nor an additional scale fit are introduced. These are outer ranges for reference-shape discrepancy, not exact outer extrema.

## 8. Same-set relaxation audit

A full ellipsoid projects to eight marginal ellipses. Their Cartesian product contains the original set. We compute both with the **same centers, q and marginal blocks** and retain both results. This isolates the cost of discarding the shared latent constraint (cross-point correlation and the single ellipsoid norm budget); it is not a comparison against a separately fitted Wave 06 calibrator and does not constitute another independent evidence source.

A source may have large absolute localization error and little relative-shape uncertainty. Conversely, identical marginal covariance blocks can hide opposing errors that change eye spacing substantially. Both cases have executable controls.

## 9. Inspectable modes

Eigenpairs of the retained 16x16 chart covariance are exported in descending order. Display signs are canonicalized by the largest-magnitude coordinate. Tied eigenspaces are not unique physical mechanisms. The inspector displays center + t q sqrt(lambda_k) v_k for |t|<=1, an admitted one-mode displacement. Sliders never refit, rescore, or mutate the stored result.

The fraction of modeled variance in the common-translation subspace is trace(T^T C T)/trace(C), T=tile(I_2,8)/sqrt(8). It is **not facial similarity or confidence**.

## Evidence boundary

Synthetic sampled sets falsify violated bounds; passing samples do not prove universal containment. The study uses renderer-known reference coordinates and a scripted observer, not MediaPipe. No real portrait performance or original-video similarity is claimed. The procedure for actual native inference still requires a compatible installed package and local trained model [2].

[1] Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511v6 (2022). https://arxiv.org/abs/2107.07511

[2] Google AI Edge, *Face landmark detection guide for Python*. https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

The joint-shape estimator and set propagation above are this project's explicit construction; the references do not establish its empirical human performance.
