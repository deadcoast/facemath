> Retained Wave 3 record. Current extensions and priorities are in [Wave 4 design](WAVE4_DESIGN.md) and [Wave 4 roadmap](WAVE4_ROADMAP.md).

# Executed Wave 3 mathematics

These equations describe this implementation, not recovered neural activations. `facemath capabilities` separately audits the inherited section-to-function registry.

## 1. Symmetric registration and its units

Let corresponding anchor arrays be A,B ∈ R^(n×2) and normalized weights w_i ≥ 0. Define weighted centroids ā,b̄, centered arrays A_c,B_c, and weighted centered variances

    v_A = Σ_i w_i ||a_i − ā||²
    v_B = Σ_i w_i ||b_i − b̄||².

For H=A_cᵀ diag(w) B_c=UΣVᵀ, the proper rotation (column-vector convention) is

    R = V diag(1, sign(det(VUᵀ))) Uᵀ.

The active instrument uses **balanced** scale

    s = sqrt(v_B / v_A),    t = b̄ − sRā.

This intentionally differs from one-way Umeyama least-squares scale. The latter primitive, repaired in the inherited API, is

    s_LS = trace(ΣD) / v_A, D=diag(1,sign(det(VUᵀ))).

Dropping D from that trace is wrong in a reflection-constrained problem. An analytic reflected-rectangle control verifies the repaired optimum, not merely that det(R)>0.

For balanced fitting, swapping A and B reverses the scale and rotation. Let d_A,d_B be the original eye spans and d_A'=s d_A. Define

    d_* = sqrt(d_A' d_B),    r_i = (s R a_i + t − b_i) / d_*.

In the reverse direction both the coordinate residual and normalizer scale by 1/s, so ||r_i|| is unchanged. The implementation tests this property across input-order swaps. The B-anchored display chart need not itself be symmetric.

Robust weights are iteratively updated as

    w_i ← min(1, δ / max(||e_i||, ε)),    δ = huber_delta_ipd × d_*.

They control the coordinate gauge only. Downweighted anchors still appear in the evaluated residuals. This reweighting is a deterministic robustness procedure, not a calibrated detector uncertainty model or guaranteed global optimum.

## 2. Regional positional and relational metrics

For the observed supported landmark set R in a region,

    RMS_R = sqrt(Σ_{i∈R} ||r_i||² / |R|).
    within_R(τ) = #{i∈R : ||r_i|| ≤ τ} / |R|.

The second is an empirical fraction with an explicit threshold. It is not identity probability, rarity, or a universal resemblance scale.

Pairwise distances D^A_ij,D^B_ij use the same d_* normalizer. Relational RMS evaluates (D^A−D^B) on the supported strict upper triangle. Pairs sharing landmarks are dependent. Their number is not an independent sample count.

For model-contour regions, centered residual RMS removes only each region's mean residual translation. It does not secretly fit a different local scale/rotation to make contour similarity improve. It is a sampled model-contour comparison, not arc-length-invariant continuous curvature matching.

## 3. Held-out registration

For each target region Q, refit the transform without anchors belonging to Q. Evaluate Q with that transform and its corresponding symmetric eye-span normalizer. Report both the held-out RMS and its signed difference from the primary RMS.

This is a nuisance-gauge sensitivity experiment. It does not create new subjects or independent anatomical observations. A high held-out residual can expose an apparently close fit that depended on using the tested region as an alignment anchor.

## 4. Piecewise-affine anatomy

For a source triangle (a_0,a_1,a_2) and its corresponding target triangle,

    P_A = [a_1−a_0, a_2−a_0],
    P_B = [b_1−b_0, b_2−b_0],
    F = P_B P_A^(-1).

Barycentric interpolation supplies the mapped position φ(x) and dense displacement u(x)=φ(x)−x inside the source triangle.

The implementation retains det(F), source-basis condition number, singular values σ_1,σ_2, Green–Lagrange strain and principal log stretch:

    E = 1/2 (FᵀF − I),    ℓ_i = log σ_i.

For any finite rigid rotation R, RᵀR=I and σ_i=1, so E=0 and ℓ_i=0. The old infinitesimal strain sym(∇u) does not share this property for finite rotations; it is unsuitable as an unqualified rotation-invariant metric.

A regional field statistic is sqrt(mean_x(ℓ_1(x)²+ℓ_2(x)²)) over supported A-domain pixels. Source triangulation, interpolation, and pixel-area sampling define its meaning. It is directional, not a symmetric surface distance, and not a tissue-mechanics observation.

Folded or collapsed maps, badly conditioned source triangles, pixels outside either valid image domain, and incomplete interpolation support are excluded. Invalid rows and support masks remain exported. Zero UV outside support is a sentinel, not a perfect match.

## 5. Independent optical-flow diagnostics

Compute A→B flow f and B→A flow g on registered gray 8-bit rasters. At source x, cycle error is

    c(x) = ||f(x) + g(x+f(x))||.

Texture observability uses the smaller eigenvalue λ_min of a locally smoothed image structure tensor, transformed as λ_min/(λ_min+floor). Source/target values are combined conservatively. Flat images have zero observability even when an algorithm returns zero flow and zero cycle error.

The instrument separately exports cycle error, texture value, photometric residual, image support, flow-vs-anatomy disagreement ||f−u||, and threshold-derived masks. None is interpreted as a calibrated anatomical correctness probability. Image flow between different people need not track corresponding anatomy.

## 6. Joint observation sensitivity

Let X^(k) be a full shape predicted under input perturbation k, mapped back to source coordinates. Center the samples around their sample mean. A conditional resampled observation is

    X* = X_primary + (X^(k) − mean_k X^(k)) + ε,
    ε_i ~ Normal(0, sigma_floor² I).

The first term preserves the cross-landmark pattern of the sampled perturbation. The independent noise term is an explicitly declared residual assumption, not an inferred confidence. A and B are drawn, the registration is refit, and all regional positional RMS values are reevaluated.

The exported 2.5/97.5 percentiles describe this conditional experiment. They are not guaranteed frequentist coverage bounds for unknown true anatomy. Unknown shared detector bias and pose/expression confounding remain outside the model.

## 7. The crossfade null: exact algebra, not an empirical resemblance claim

Vectorize two arbitrary images as a,b ∈ R^p. For frame t,

    x_t = α_t a + (1−α_t)b = b + α_t(a−b).

Stacking frames gives X = 1 bᵀ + α(a−b)ᵀ, hence rank(X)≤2. Centering time gives

    X_c = (α−mean(α))(a−b)ᵀ,

so rank(X_c)≤1. This holds for unrelated random textures, different objects, or strongly mismatched faces. A low-rank crossfade does not establish anatomical compatibility.

A frame-mean or RPCA component contains a mixture, not an identifiable common facial anatomy. Smooth latent paths, temporal prediction, and low-rank decomposition require appropriate null controls before being treated as evidence. Wave 3 explicitly marks these video diagnostics inadmissible as stand-alone facial-similarity evidence.

Time derivatives use supplied timestamps, not frame indices. Video inspection preserves both decoded timestamps and the constant-frame-rate index-derived timestamps used for this bounded sampling diagnostic. Variable-frame-rate scientific analysis requires a stronger demux/timestamp contract.

## 8. Conditional reference-cohort rank

For a lower-is-closer scalar q and cohort values v_i,

    percentile(q) = 100 × mean_i[1(v_i>q) + 0.5·1(v_i=q)].

The comparison direction is reversed for higher-is-closer metrics. Ties use midrank. This is a finite reference-cohort statistic, not an identity probability or calibrated universal percentage.

Rows sharing a source image or supplied subject ID belong to one connected component. Bootstrap sampling chooses components with replacement and includes their rows. The stated statistic remains a row-weighted percentile; the resampling acknowledges within-component dependence. Fewer than three independent components yield no interval. It does not solve nonrepresentative sampling, unknown subject duplication, or dataset-selection bias.

## 9. What is not silently claimed

This release does not implement an exact Kolmogorov-complexity oracle, recover historical transformer activations, prove a facial identity/geometry latent factorization, fit a 3D morphable model, optimize full Gromov–Wasserstein transport, equate 0D persistence with all persistent homology, or identify causal resemblance circuits.

Some inherited numerical primitives remain useful research code. Their mappings to the prior discussion are not equivalence proofs. `capabilities` records known gaps instead of treating a registry entry as a completed equation.
