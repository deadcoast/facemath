# Wave 06 · Uncertain normalization and refitted reference shape

## Why this work

Wave 05's error model qualified point gaps while holding the *observed* registration and normalizer fixed. It did not enclose an independently normalized, refitted shape discrepancy. That distinction is material: an error in the eye span changes every normalized measurement, even when the measured mouth coordinates do not move.

The native model download was attempted again and failed. This wave does not rename synthetic observations as real inference. Instead, it closes the separate geometric gap within the actual shared instrument. Live observation against independent, group-separated real portraits remains the outstanding empirical gate.

## Shared computation rather than disconnected metrics

`FacialInstrument` constructs its existing `CanonicalState` and qualified localization envelope. `measure_intrinsic()` consumes those exact image-bound observations and the same validated calibration. It maps only `reference8`, transports full covariances into source-local observed eye charts, and establishes one `EllipticSource` per source. The chart is a coordinate convention; the *reference* eye span remains an uncertain expression.

`SourceRelations` caches linear-combination bounds. Distances, triangle arms, the eye-midpoint vector and recentered vertices refer to the same eight variables. Bilinear expansions retain shared-apex cancellation. No analyzer reacquires landmarks, refits the localization model, or performs its own pair registration.

`IntrinsicProfile` is immutable. It records source charts, raw and model-centered positions, covariance, masks, directions, relation inventory, bounds, reasons, decisions and the refitted shape range. The existing `InstrumentResult` exports it in the same manifest/journal/bundle. `raw_geometry_arrays()` explicitly distinguishes unqualified measurement arrays from newly derived qualification arrays. In the laboratory, calibration attachment must leave the entire observed subset byte-identical.

## What is measured

The prespecified reference8 inventory contains 28 pairwise lengths, 168 included angles, 56 oriented double areas and four signed bilateral contrasts. Each retains its own units and tolerance. All 256 are algebraically dependent descriptions of eight points. They are never treated as independent samples or pooled into a resemblance probability.

The full-shape metric centers each reference shape, divides it by its own eye-midpoint span and minimizes RMS discrepancy over SO(2). The reported range encloses this freshly optimized value for every pair in the admitted error sets. It does not use the observed robust pair transform as if known exactly. It is not exact optimization of the interval extrema, not all-478-vertex coverage and not a 3D similarity claim.

This range and the inherited fixed-observed-gauge point ranges are different questions. Both remain available, separately labeled. No earlier result is silently replaced.

## Missingness and conditioning

Eye corners must be observed to define a usable source chart. Missing non-anchor reference points withhold dependent relations without shrinking the prespecified fraction denominator. If a conservative eye-span range reaches zero, normalized lengths, areas and the full-shape range are withheld; nominal observation values remain separate. Angles and contrasts are withheld when their own denominator/arm can vanish. A conservative range touching zero does not prove that the exact feasible eye vector actually reaches zero.

The existing production gate still rejects an unbounded localization calibration. The intrinsic low-level representation distinguishes absent calibration from unbounded calibration rather than substituting a zero error multiplier.

The fraction of prespecified relations within a declared tolerance has deterministic lower/upper bounds. Unknown relations remain possible members of either class. It is a count over this inventory, not a population percentile, independent-evidence count, or probability of facial resemblance.

## Numerical controls

Analytic and synthetic checks cover: SO(2) SVD versus an independent trace/hypot solution; mirrored shapes with identical unsigned distance matrices; contraction as error radii tend to zero; normalizer collapse; global rotation/scale and A/B exchange invariance; finite-angle support bounds; anisotropic ellipses; shared-point bilinear cancellation; false agreement caused by deleting missing points; and preservation of all original measured arrays.

The set laboratory retains the actual sampled points, exact scalar values and all reported bounds. Samples test the formulas, not define the enclosures. Production bounds are analytic conservative calculations with floating-point allowances, not a formally verified interval-arithmetic implementation.

The integrated laboratory uses the inherited frozen-acquisition study with 12 development, 24 calibration and 12 evaluation groups plus one excluded enrollment. It retains the prescribed failed cases and unseen-shift evaluation stratum. Four fresh synthetic queries apply prespecified mouth expansions of 0, 4, 8 and 12 pixels per side. Neither the calibrator nor the query selection is tuned using these query errors.

## Faults found during development

1. **Unsupported chin affected canonical orientation.** A missing-point regression showed that `eye_chart` used an off-image chin before later support masks could exclude it. It now chooses the first supported lower anchor (chin, mouth corners, nose) and fails if none is available. This change concerns orientation support; it is not a claim of a new comprehensive occlusion detector.
2. **Version checks were hard-coded to 0.5.0.** The two inherited assertions now expect the actual 0.6.0 release. No inherited numerical assertion was weakened.
3. **Source execution requires installation for subprocess tests.** The first uninstalled parent run failed subprocess imports. Installing the actual parent editable package produced 370 passes and one skip before source changes. Both runs are retained; the import failure was not relabeled an algorithm defect.
4. **Unbounded versus absent calibration.** A low-level representation initially conflated the two because both use a null finite multiplier. The representation now distinguishes them, while the pre-existing production refusal remains in place.
5. **Qualification versus raw-array invariance.** The inherited study laboratory previously compared every array from an uncalibrated result. New error-set-derived arrays legitimately differ on qualification. The explicit raw subset now compares all old measured arrays plus the new observed-only intrinsic arrays; bounds and model-centered quantities are excluded by name, not by a tolerance that could hide mutations.

## Browser and artifact behavior

The report reads one embedded immutable JSON profile. Diagrams use stored source-chart coordinates; ellipse display, filters, selection and zoom do not recompute measurements. The selected row names its exact NPZ columns. The CSV export preserves the underlying values rather than rounded table strings. Missing values export empty, not zero.

`intrinsic.json` must agree with the profile in `evidence.json`. Byte hashes catch accidental changes; consistently rehashed false narrative can pass consistency verification but is caught by numerical replay. This distinction has an explicit test. Hashes remain self-consistency tools, not externally trusted provenance.

## Next work, in order

Run the frozen native observer pipeline against real independent reference annotations. Audit equivalence and visibility of those eight definitions. Evaluate acquisition shift rather than assuming conformal coverage transfers. Then use real localization data to judge whether the conservative intrinsic bounds need tighter joint-error modeling. Expand independently observed contour/3D support only with evidence appropriate to those measurements. Do not reopen the earlier unsupported claim that a section-name registry implements all preceding mathematics.
