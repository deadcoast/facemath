# Wave 08 · Prespecified acquisition categories and source-specific radii

## Objects and units

The same development-only ocular-frame error mean `beta` and regularized full16 scatter `C` from Wave 07 are retained. Eight source-reference locations are flattened in interleaved `(x,y)` order. This is an observation-error model, not an identity embedding or a facial-resemblance model. Its scatter is not silently rescaled when a category changes.

For observed reference8 coordinates `p`, independent labels `r`, and the observable eye basis `E(p)`, the existing score is

    e = vec(E(p)^(-1) (p-r)) - vec(beta)
    s_case = sqrt(e^T C^(-1) e)
    s_group = max_{cases in the declared group} s_case.

Code: `joint.fit_joint_calibration` remains the only full16 fit/score implementation; `stratify_joint` partitions its retained scores. Failed/incomplete cases retain infinite scores. No new fit is made inside a category. Development groups are disjoint from calibration, evaluation and qualified queries.

## Prespecified partition

A frozen, finite acquisition-category map `h` assigns each whole declared dependent group to exactly one category. All images in that group must share it. The map is acquisition metadata and must be determined before inspecting observer errors, held-out outcomes or pair similarity. It is NOT inferred identity, a demographic classifier, or a learned post-error cluster.

For category `c` with `n_c` calibration groups:

    k_c = ceil((n_c + 1) (1-alpha))
    q_c = kth order statistic of its retained group maxima.

If `k_c > n_c`, an infinite score is selected, no groups exist, or the separately prespecified minimum-group policy is not satisfied, the instrument withholds a deployable radius. It does not clip `k`, drop failures, use a neighboring category, or substitute the pooled quantile. The containing model preserves the original pooled calibration only as a named diagnostic.

The conventional split-conformal rank argument applies conditional on the fixed development fit and the prespecified category, with exchangeable calibration/new whole-group scores within that category [1,2]. It is not a guarantee conditional on a particular observed calibration dataset, on an individual image, or on selecting only finite delivered outputs. It is not simultaneous correctness of every future deployment. Group composition and dependence must match the declared acquisition protocol. A metadata label cannot establish these assumptions.

Abstention is an operational decision. An unbounded theoretical set is not delivered as finite geometry. The report separates ordinary category empirical coverage, the fraction of ALL evaluation groups that received finite bounds AND were covered, and the supported-only diagnostic. It never renames the last of these as a new conformal guarantee. For a heterogeneous category mixture with abstentions, the report does not assert an aggregate iid-binomial interval. Per-category binomial intervals separately require independent comparable Bernoulli groups.

## Different source sets in one pair

For source `s` in declared category `c_s`, its admitted full shape is

    r_s = center_s + q_{c_s} L_s z_s,   ||z_s||_2 <= 1.

`L_s L_s^T` is the SAME transported full16 covariance from the shared development model. The source-bound wrapper records the exact observation digest, raster digest, group, lineage and category. There is no `max(q_A,q_B)` replacement and no `min(q_A,q_B)` shortcut.

For the inherited fixed-observed-gauge gap, using transformed marginal blocks `S_Ai,S_Bi`, directional support is

    h_i(u) = q_A sqrt(u^T S_Ai u) + q_B sqrt(u^T S_Bi u).

The existing angular support bounds, finite-angle cosine correction and enclosing-circle intersection remain in place. `ellipse_gap_bounds` accepts either a legacy scalar radius or an explicitly ordered two-source vector. Equal vector radii give the exact scalar computation. Boolean/string/nonfinite/mis-shaped radii and nonrepresentable bounds are rejected.

For intrinsic expressions, each `SourceRelations(EllipticSource(...))` receives its own `q_s`. Cross-landmark covariance and the single source norm budget remain shared. Uncertain reference eye spans, bilinear shared-landmark cancellation, orientation-preserving refitted shape bounds, and the same-set marginal relaxation therefore all use the correct source radius. Neither raw landmarks nor the canonical pair alignment is changed.

The nominal two-source failure union bound requires valid per-source assumptions; no cross-source independence is asserted. Increasing the number of derived relations does not create additional independent evidence or a similarity probability.

## Analytic counterexample

Take a center gap `(0.5,0)`, isotropic source standard deviations `0.1` and `0.2`, and source radii `q_A=1`, `q_B=3`. A displacement to `(1.2,0)` is admissible. The correct maximum gap is `0.5 + 0.1 + 3*0.2 = 1.2`.

Using the smaller radius for both gives an upper bound `0.8`, excluding that admissible point. Using the larger radius for both gives `1.4`, introducing unnecessary over-approximation. The executed control saves the witness, covariances, radii and calculated bounds.

The mixed-source full16 control separately retains all unit-ball samples, shape draws, exact relation values and refitted shape values. It tests the production bounds; samples are not the method by which those bounds are defined. Existing bounds remain conservative float64 calculations with padding, not formally verified interval arithmetic or exact ellipsoid-constrained extrema.

## Primary sources

[1] Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, section 4.1 (group-balanced conformal prediction). https://arxiv.org/html/2107.07511v6

[2] Vovk, *Conditional validity of inductive conformal predictors*. https://arxiv.org/abs/1209.2673

The application to whole acquisition-group joint localization scores and source-bound propagation is this instrument's explicitly specified implementation. These papers are not evidence of FACEMATH's empirical detector accuracy.
