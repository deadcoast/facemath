# Wave 04 · From repeatability to reference-qualified geometry

## Priority decision

Wave 3 established controlled geometric computation after observations existed. It did not establish how accurately a real observer located anatomy. Adding more unrelated similarity formulas would not repair that gap. Wave 4 therefore builds an observation-validation loop around the existing `Observation` and `CanonicalState`, plus a separately retained image-edge diagnostic.

The live runtime/model could not be acquired here. API-fixture tests remain separate from live inference. A synthetic biased observer is a falsification control for this software, not a stand-in result for MediaPipe. No human similarity score or human error calibration is released.

## Independent reference contract

`ReferenceAnnotation` contains eight source-pixel coordinates, explicit visibility, source pixel SHA-256/shape, rater, acquisition source, provenance kind and prediction-exposure declaration. Arrays own immutable bytes. Coordinate ordering is bound to a protocol digest. Unobservable coordinates have zero sentinels plus a false visibility mask; zero alone never means a valid missing landmark.

The workbench starts without predictions. Raters locate viewer-left/viewer-right eye corners, nose tip, two lip junctions and the lowest observable central chin contour. Changing rater clears the current annotations. Local draft corruption is rejected before coordinates can be restored. Blinding cannot be externally established by an HTML file; the export records that limit explicitly.

Consensus retains each original annotation, per-point count and RMS disagreement. A single-rater zero RMS does **not** show inter-rater agreement. The reference mean is an operational label, not known anatomical truth. Detection error, perturbation repeatability and annotation disagreement remain three separate quantities.

MediaPipe478 and semantic21 are projected to the same eight named vertices through the inherited anatomy grammar. The PTS importer maps only the nominally corresponding eight iBUG68 indices. It requires an explicit 0/1 origin and flags that visibility and anatomical equivalence need review. It does not fabricate a 68-to-478 mapping.

## Source-coordinate evaluation

For observed point p_i, independent reference r_i and reference eye-corner midpoint span d_ref:

    e_i = p_i - r_i
    NME = mean_i ||e_i|| / d_ref
    PCK(t) = count(eligible AND observed AND ||e_i|| / d_ref <= t) / count(eligible)

No rotation, translation or scale is fitted to the labels. A stable global localization bias must remain an error, even though downstream face registration could absorb it.

Failed or partially unobservable predictions do not receive zero NME. Successful-case mean NME is explicitly conditional. PCK retains eligible labels from every failed attempt. Unconditional AUC up to t_max is computed exactly as the mean of max(0,1-NME/t_max), with failed cases contributing zero.

Cases have immutable source bindings and curator-declared group/lineage IDs. Repeated images cannot masquerade as separate cases. The same acquisition lineage cannot cross a split or be assigned to multiple independent groups. These are checks on supplied metadata, not identity inference. A provider returning coordinates bound to another image causes a contract error, not a silently pooled nondetection.

## Calibration score and corrected order statistic

The observer supplies a covariance C_i from perturbation samples plus its declared noise floor. This covariance by itself describes modeled repeatability, not localization accuracy. For each calibration case:

    s_case = max_i sqrt(e_i^T C_i^-1 e_i)
    s_group = max_{cases in group} s_case
    k = ceil((n_groups + 1) * (1 - alpha))
    q = kth sorted group score, or +infinity when k > n_groups

An unsuccessful case, incomplete set of the eight prespecified labels, or insufficient raters supplies +infinity. Infinity is serialized as `null` with an explicit unbounded flag. It is not discarded or clipped. Group maxima avoid counting correlated landmarks or repeated group images as independent calibration draws.

The standard split-conformal rank argument applies to the scalar group scores **under exchangeable groups and a frozen scoring/acquisition protocol** [2]. That includes comparable group composition and acquisition conditions. The artifact does not prove exchangeability or establish deployment coverage. It cannot support a per-image posterior probability. The evaluation split reports actual empirical group coverage separately, retains all failures and cannot reuse calibration images/groups/declared lineages.

The provider fingerprint, schema, attempted perturbation count, noise floor, reference protocol and score recipe are bound. Realized successful perturbations do not define a selectively easier protocol. The reference kind, minimum raters, domain and training/calibration source identities are retained. Runtime/code provenance is recorded in the containing bundle.

## Integrating uncertainty without changing the measured shape

Let T_A(x)=L_A x+t_A and T_B(x)=L_B x+t_B be the *observed* canonical transforms, and d the *observed* shared normalizer. We push the full calibrated covariance into that same frame:

    delta_i = (T_A(p_Ai) - T_B(p_Bi)) / d
    S_Ai = L_A C_Ai L_A^T / d^2
    S_Bi = L_B C_Bi L_B^T / d^2

For a unit direction u, the error ellipses have combined support:

    h_i(u) = q sqrt(u^T S_Ai u) + q sqrt(u^T S_Bi u)

For M equally spaced unit normals u_j:

    lower_i = max(0, max_j [u_j dot delta_i - h_i(u_j)])
    upper_i = max_j [u_j dot delta_i + h_i(u_j)] / cos(pi/M)

The lower bound follows from supporting halfspaces and norm duality. The upper correction is essential: every possible vector points within pi/M of a sampled normal, so its nearest projection is at least its length times cos(pi/M). Using only the sampled maximum could understate the maximum distance. The instrument intersects these bounds with the enclosing-circle triangle-inequality bounds and includes a numerical roundoff allowance.

M=128 is recorded, as are all normals, covariance matrices and support values. This preserves directional uncertainty that a scalar radius discards. An analytic perpendicular-uncertainty control improves the enclosing-circle interval [0.30,0.70] to approximately [0.49,0.54], without adding evidence or moving a point.

The resulting interval concerns reference positions under the **observed transforms and observed scale held fixed**. It is NOT an interval for a latent true-shape Procrustes fit, not for full 478-vertex geometry, and not for population resemblance. Whole-shape perturbation/refit sensitivity from Wave 3 remains separately available. Under the same calibration assumptions, the two-source union bound is at least 1-2*alpha; no independence between the two sources is asserted.

The profile classifies a point as within tolerance only when the upper bound is below the declared geometric tolerance, outside only when its lower bound exceeds it, otherwise unresolved. Calibration adds evidence; it never rewrites the existing measured geometry to make a desired classification.

## Image-edge diagnostic

On the shared canonical rasters, source-supported contour segments generate normal sampling rays. Sobel normal-gradient response, orientation consistency, all search offsets, source support, competing peaks and selected offsets are retained. The search window is normalized by the canonical interocular span.

Weak, multiply plausible, boundary-maximal or source-padding edges are rejected. A selected texture edge is not automatically an anatomical boundary. These are **landmark-guided image diagnostics**, not statistically independent ground-truth contours. The algorithm does not snap landmarks to local maxima or force A and B to agree. It does not turn edge confidence into facial similarity or calibrated localization accuracy.

## Bundle integration and replay

The pair output retains the same state, transform, masks, field, profile and journal as Wave 3. Edge diagnostics and optional calibrated ellipse bounds append typed results and raw arrays to that bundle. Validation bundles retain every source image/reference/observation, immutable study lock, per-case arrays, separate split summaries, the order-statistic inputs, coverage outcomes, source hash chains and actual execution events.

The journal distinguishes an actual provider invocation, an imported observation, and an imported failed attempt. Frozen-observation replay recomputes numerical reports exactly but makes no claim to rerun the live detector. Byte verification is self-consistency, not externally signed provenance. Current replay binds the complete package code fingerprint and numerical runtime; even a code-only presentation change can require a new run.

## Evidence delivered

The 60-case schematic laboratory has forty calibration groups and twenty held-out groups, a fixed prescribed nondetection in each split, systematic observer bias and much smaller perturbation noise. It is designed to falsify the equation “repeatable = accurate.” It is not a sample of human faces, a face-recognition benchmark, or a calibration of the upstream runtime.

The browser acceptance test executes the actual UI, pointer-coordinate mapping, zoom, missing labels, undo/redo, keyboard nudges, JSON download, strict Python import and rater switching at desktop/mobile widths. This environment blocks file/HTTP browser navigation; the same HTML was rendered in memory, with local PNG bytes inlined where necessary. Draft-storage fallback was exercised; persistence across browser restarts was not validated. Browser reports name these limits.

## Primary references and upstream boundaries

[1] Google AI Edge, Face landmark detection guide for Python. Official image API, required local model, 478-point output, configuration. Accessed 2026-09-11.
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

[2] Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification, arXiv:2107.07511 (2021; revised 2022). The group-max score and fixed-gauge geometric pushforward are this instrument's explicitly specified application of the standard rank construction.
https://arxiv.org/abs/2107.07511

[3] iBUG, Facial point annotations. The published 68-point annotations have their own provenance and terms; do not assume commercial redistribution rights or perfect visibility labels. No dataset assets are included here.
https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

Upstream MediaPipe is optional. FACEMATH does not download models implicitly. The workbench/report makes no remote requests; that does not certify all upstream runtime telemetry behavior. Inspect and pin your chosen runtime and model before deployment.
