# Wave 10 · Prespecified held-out acceptance, without changing shape

## 1. Freeze the complete question

An `AdmissionPolicy` fixes required acquisition categories C, minimum evaluation-group count m, delivery target d0, coverage target c0, family error δ, declared evidence provenance and sampling assumptions before observation. Category definitions and the selected estimator are already frozen by the existing study. Every required category must pass. The procedure does not rank models or select the categories that happened to succeed.

Within category c, group g is one evaluation unit. Its eligible cases share the existing frozen grouping, not a new grouping manufactured by the acceptance layer. Let q_c be the existing category-calibrated radius and s_gj the existing full-shape error score for case j. A missing/incomplete observer output or reference8 set is not a finite delivered result.

    D_g = 1{q_c finite and every eligible case has a complete finite score}
    H_g = D_g * 1{max_j s_gj <= q_c}
    n_c = number of enrolled eligible evaluation groups in c
    k_Dc = sum_g D_g
    k_Hc = sum_g H_g

Failed observations and unavailable radii stay in n_c. The maximum is not taken only over successful cases. Groups outside the requested category scope remain in separate full-enrollment accounting; they are not used to improve the requested subset's rate.

Implementation: `assess_admission()` consumes the selected model's existing scored `LocalizationRow`s. It does not fit beta, scatter, q, landmarks, registration or geometry. The complete evaluation inventory must match the frozen cases exactly.

## 2. Exact one-sided binomial lower bound

For successes k out of n and one-sided tail η:

    L(0,n;η) = 0, for n>0
    L(k,n;η) = BetaInverseCDF(η; k, n-k+1), for k>0
    L(0,0;η) = unavailable

This is the Clopper–Pearson exact lower bound. For k=n, the closed form is η^(1/n). It inverts P_p[X >= k] = η. The implementation cross-checks the inverse-beta result against a separately implemented binomial-test confidence interval, the binomial tail equation, and the all-success closed form.

There are 2|C| prespecified endpoint/category comparisons. We use η=δ/(2|C|). By the union bound, simultaneous lower-bound coverage is at least 1-δ **when the individual binomial assumptions hold**. The two endpoints are dependent (H<=D); their independence is not required for Bonferroni. The allocation is deliberately conservative and is not optimized post hoc.

Acceptance requires, for every requested category:

    n_c >= m
    L(k_Dc,n_c;η) >= d0
    L(k_Hc,n_c;η) >= c0

The policy also requires an explicit declaration of independent comparable groups. The application cannot establish that assumption from labels or hashes. The statistical interpretation is conditional on the frozen development/calibration model and a suitable evaluation sampling process. Marginal conformal exchangeability does not by itself imply independent Bernoulli groups.

Acceptance is a decision under declared empirical criteria, not a per-image posterior. Selection after passing does not create a stronger conditional conformal theorem. This implementation does not control multiple attempts across adaptively redesigned studies; new IDs do not make such attempts statistically independent.

## 3. Pre-observation algebraic feasibility

Even a perfect evaluation cannot exceed the all-success bound η^(1/n). Therefore:

    n_perfect >= ceil(log(η) / log(max(d0,c0)))
    n_min = max(m,n_perfect)

`plan_admission()` reports this before any observer is called. It is an algebraic minimum, **not** an expected pass rate, power analysis, practical sample-size recommendation or proof of representativeness. Underpowered studies may be collected for research, but cannot meet this frozen acceptance rule.

## 4. The geometric computation is unchanged

The accepted model still supplies the same developmental ocular-frame bias, full16 covariance and category radii. Query A and B retain their separate q_A, q_B. The same CanonicalState, full joint uncertainty sets, SourceRelations coefficients, eye-span normalization and proper-rotation shape bounds compute the result.

    reference source set: r = c + q L z, ||z|| <= 1
    pair support: h(u) = q_A sqrt(uᵀ S_A u) + q_B sqrt(uᵀ S_B u)

Neither the CP calculation nor an acceptance status changes raw coordinates, model centers, q, covariance or a geometric threshold. `require_query()` only authorizes the bound source/calibration combination and forbids enrolled source reuse. All earlier raw geometry arrays remain byte-identical in the control.

## 5. Three distinct quantities

1. A **relation fraction** within a fixed geometric tolerance is a deterministic count over a prespecified, dependent inventory. Its existing lower/upper count bounds may be multiplied by 100 for display. No new score is fitted.
2. A **held-out coverage lower bound** concerns complete reference-error sets under the declared sampling model. It is not a face-resemblance percentage.
3. An **intrinsic shape interval** encloses a particular normalized, refitted geometric discrepancy under the admitted source error sets. Finite localization sets can still yield unresolved or unavailable intrinsic ratios when a denominator degenerates; empirical delivery is not a promise of narrow or useful geometric bounds.

The new readout separates these quantities rather than collapsing them into one numeric verdict.

## 6. Evidence admission is a recomputation, not a hash lookup

`ValidatedQualification` loads the raw source/reference/observation/receipt inventory, reruns the existing qualification and compares the complete report, model and arrays. Its portable snapshot is bounded and rejects traversal, symlinks, duplicate/case-colliding paths and Windows device names. An ordinary hash-verification pass can accept a consistently rehashed false derived report; a regression test demonstrates that full numerical replay rejects it.

Query overlap is checked against **all** enrolled development, calibration, evaluation and excluded source images, declared groups and lineages, including unselected context images. This protects a prespecified evidence workflow; it does not discover undeclared re-encodings or dishonest source identities.

## 7. Laboratory interpretation

The positive demonstration intentionally places evaluation errors at the development mean while calibration probes surround it. This guarantees a useful software branch control without selecting a successful random seed afterward. The negative demonstration freezes the same acceptance policy and deliberately shifts a required category's evaluation errors. Both are generated under explicit synthetic provenance before execution.

This construction is NOT an exchangeable empirical human study. Its CP outputs exercise the implementation and branch behavior; they do not validate the sampling assumptions or a real detector. Real calibration/evaluation cohorts are still required.

## Primary mathematical references

- SciPy, `binomtest` and `BinomTestResult.proportion_ci` (exact method): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
- Clopper, C. J. and Pearson, E. S. (1934), *The use of confidence or fiducial limits illustrated in the case of the binomial*. Referenced by the SciPy exact-method documentation.
- The union-bound allocation above is this instrument's prespecified simultaneous decision rule. It does not rely on independence between the two endpoints.
