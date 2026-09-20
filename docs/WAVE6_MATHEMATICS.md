# Wave 06 mathematics and Python correspondence

## 1. Source sets and coordinate convention

For each of eight reference points, let the reference position be

    x_i = c_i + e_i,    e_i in q C_i^(1/2) B_2.

`c_i` is the observed coordinate minus any frozen development-fit localization bias. `C_i` is the same observation covariance transported through its observed source-eye chart. The product of these ellipses is an **outer set**. It does not assert independence or fit cross-point covariance. Statistical validity comes only from the upstream simultaneous reference calibration assumptions.

The source chart is a fixed proper similarity transform. An intrinsic relation is unchanged by that choice; it does not fix the true reference eye span. With

    v = (-1/2,-1/2,1/2,1/2,0,0,0,0),
    d(x) = ||sum_i v_i x_i||,

`d` remains uncertain. Source chart units are observed eye spans; final normalized relations use each admissible reference shape's own `d`.

Python: `EllipticSource`, `ocular_basis`, `measure_intrinsic`.

## 2. Shared linear combinations

For coefficients `a` summing to zero:

    z(a) = sum_i a_i x_i = z0(a) + sum_i a_i e_i.

The error set support function in unit direction `u` is

    h_a(u) = q sum_i |a_i| sqrt(u^T C_i u).

Using `M` equally spaced normals in 2D,

    lower = max(0, max_j [u_j dot z0 - h_a(u_j)]),
    upper = max_j [u_j dot z0 + h_a(u_j)] / cos(pi/M).

For any vector, some sampled normal is within pi/M of it; its projection is at least its norm times cos(pi/M). Therefore the corrected upper is conservative. The lower follows from norm duality. Intersect with the enclosing-circle bounds `max(0,||z0||-rho)` and `||z0||+rho`, where `rho = q sum_i |a_i| sqrt(lambda_max(C_i))`.

`SourceRelations.norm()` caches this expression. `radius()` bounds the corresponding zero-centered error combination. Identical expressions are simplified before propagating uncertainty. For example, `||v x||/||v x|| = 1` when the denominator is bounded away from zero.

## 3. Distances and uncertain normalization

A source distance is

    L_ij(x) = ||x_i-x_j|| / d(x).

If numerator is in `[nlo,nhi]` and denominator in `[dlo,dhi]` with `dlo>0`, endpoint division encloses the ratio. The numerator and denominator can share variables; interval division is still an outer bound, although generally not tight. No independence is assumed. A nonpositive denominator lower bound causes a withheld result, not a frozen-denominator approximation.

Python: `length_ratio`, `divide`.

## 4. Bilinear terms and shared-point cancellation

Let `u=a x`, `w=b x`. For dot or cross products, expand the exact degree-two polynomial:

    B(u,w) = B(u0,w0)
             + sum_i [a_i B(e_i,w0)+b_i B(u0,e_i)]
             + sum_ij a_i b_j B(e_i,e_j).

The first-order term is one linear expression per physical point, bounded by its ellipse support. Its two appearances are not treated as independent errors.

For cross products, `cross(e_i,e_i)=0`; for i<j the coefficient is

    a_i b_j - a_j b_i.

For dot products the off-diagonal coefficient is `a_i b_j+a_j b_i`. A diagonal `a_i b_i ||e_i||^2` has its sign preserved, with squared radius in `[0,r_i^2]`. Other second-order terms are bounded in magnitude by coefficient magnitude times `r_i r_j`. These calculations preserve shared-apex cancellation but do not claim the exact joint extrema.

Python: `bilinear(a,b,'dot')`, `bilinear(a,b,'cross')`.

## 5. Angles, oriented areas and bilateral contrast

The included angle is

    theta = atan2(|cross(u,w)|, dot(u,w)) in [0,pi].

The nominal value uses atan2. Its interval uses the dot-product bound, tightened by the independent outer constraint `(||u||^2+||w||^2-||u-w||^2)/2`, divided by bounded positive arm lengths, then mapped through decreasing acos. Vanishing arms withhold the angle. Equal/opposite arms are simplified algebraically.

Signed **double** triangle area is

    A_ijk = cross(x_j-x_i,x_k-x_i) / d(x)^2.

It preserves orientation under fixed point correspondence: a reflection flips its sign. It is not an unsigned area or a correspondence detector.

Bilateral contrast for two lengths is

    s=(L-R)/(L+R).

On nonnegative lengths it increases in L and decreases in R, giving interval endpoints from `(Llo,Rhi)` and `(Lhi,Rlo)`. Equal expressions cancel to zero. A denominator that can vanish withholds the result.

Python: `angle`, `signed_area_ratio`, `asymmetry`.

## 6. Scalar difference decisions

Given source intervals `[a,b]` and `[c,d]`, an absolute relation-gap outer interval is

    [max(0,a-d,c-b), max(|a-d|,|b-c|)].

It is within tolerance if its upper bound is <= the frozen tolerance; outside if its lower bound is > tolerance; otherwise unresolved. Withheld dependencies are not zero gaps. Fractions use the prespecified relation-family size as denominator, with unresolved/withheld cases contributing only to the possible upper agreement count.

There are 28 pair distances, 8*C(7,2)=168 angles, C(8,3)=56 signed double areas and four contrasts. All come from the same eight-point variables and same source coverage event. No 256-way independence claim or extra confidence is introduced.

## 7. Refitted proper-rotation shape metric

For each shape x, define

    F(x) = (x - mean(x)) / d(x).
    D(x,y) = min_{R in SO(2)} ||F(x) R - F(y)||_F / sqrt(n),  n=8.

Centering removes translation and eye normalization removes scale. The fit is proper rotation only. It does not use reflection, affine alignment, or another least-squares scale fit. The ordinary SVD minimizer uses `R=U diag(1,det(U Vt)) Vt` from `F(x)^T F(y)`.

An independent two-dimensional check uses

    best = hypot(M00+M11, M10-M01),
    D^2 = (||F(x)||_F^2+||F(y)||_F^2-2*best)/n.

Python: `proper_shape_distance`; independent test/lab function `_so2_minimum`.

## 8. Bound normalization and recentering before fitting

For source-set center c, let `d0=d(c)`, and bound the admissible eye span by `[dlo,dhi]` with `dlo>0`. For centered point i, coefficients are `a_i = unit_i - ones/n`. Let `rho_i` bound its recentered error using the shared linear-combination support from section 2. Then

    ||F(x)_i - F(c)_i||
      <= rho_i/dlo
         + ||c_i-mean(c)|| * max(|1/dlo-1/d0|, |1/dhi-1/d0|)
      = kappa_i.

The first term covers point/recentering error. The second explicitly covers uncertain scale. Both vary under the same reference shape; summing valid upper bounds remains conservative.

Let `rA=RMS(kappa_A)`, `rB=RMS(kappa_B)`, and `D0=D(cA,cB)`. The quotient distance under orthogonal rotations is 1-Lipschitz in each Frobenius input, hence

    max(0,D0-rA-rB) <= D(x,y) <= D0+rA+rB.

This is a bound on a fresh optimum for every admissible shape pair, not on the residual of a frozen observed transform.

## 9. Independent relational lower bound

For any rotation, let residuals `e_i=F(x)_i R-F(y)_i`. Both shapes are centered. Reverse triangle inequality gives

    |D_ij(F(x))-D_ij(F(y))| <= ||e_i-e_j||.

Also,

    sum_{i<j} ||e_i-e_j||^2 = n sum_i ||e_i||^2.

If each pairwise normalized distance gap is bounded below by `l_ij`, then

    D(x,y) >= sqrt(sum_{i<j} l_ij^2)/n.

This lower bound does not detect a pure reflection by itself; that is why it is combined with the proper-rotation quotient bound rather than replacing it.

## 10. Candidate-rotation upper bound

Take the admissible proper rotation minimizing the set-center objective and residuals `e0_i`. For every admissible shape pair its objective at that rotation is at most

    RMS(||e0_i|| + kappa_Ai + kappa_Bi).

The optimal objective cannot exceed an admissible candidate's objective. Intersect this upper bound with the quotient Lipschitz upper, and intersect the two lower bounds. Export every component and the candidate rotation. These are conservative component bounds, not a claim to have solved the exact outer extremization problem.

As q tends to zero, kappa tends to zero and the range contracts to the proper-rotation value. This also holds for a mirrored configuration with matching distance matrices; the SO(2) restriction remains active.

Python: `bound_refitted_shape`.

## 11. Numerical and statistical limits

Source charts provide dimensionless, eye-scale-conditioned coordinates. The implementation uses float64, explicit finite-angle corrections, 128-epsilon scale-aware allowances and outward nextafter in scalar operations. It is **not** a formally certified interval arithmetic library. Extreme conditioning can cause withholding or explicit failure. Random boundary/interior controls and analytic invariants test the implementation but do not prove all floating-point executions.

The source error calibration supplies one simultaneous event over the eight reference points, subject to its frozen protocol, group exchangeability and domain assumptions. On the intersection of the two source events, all deterministic geometric consequences hold together. The inherited union-bound lower coverage is 1-2*alpha; no new statistical guarantee arises from counting relations. The synthetic study's deliberately shifted evaluation stratum remains evidence that transfer assumptions matter.

Reference-position error, conditional perturbation repeatability, annotator disagreement, population resemblance and identity verification remain distinct quantities.

## Primary references and boundaries

1. Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, https://arxiv.org/html/2107.07511v6 — upstream rank construction and coverage assumptions. The source-set propagation and refit bounds above are the explicitly derived application here, not claims made by that paper.
2. SciPy, `scipy.linalg.orthogonal_procrustes`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html — orthogonal fitting conventions; the generic routine permits reflection. FACEMATH implements the proper-rotation restriction explicitly.
3. Google AI Edge, *Face landmark detection guide for Python*, https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python — optional observer's API boundary. No assertion that its predicted coordinates are ground truth.

Accessed during Wave 06 development. No external neural telemetry or live detector accuracy is inferred from these references.
