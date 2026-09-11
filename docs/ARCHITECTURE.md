> Retained Wave 3 record. Current extensions and priorities are in [Wave 4 design](WAVE4_DESIGN.md) and [Wave 4 roadmap](WAVE4_ROADMAP.md).

# Wave 3 architecture and operating contracts

## The system boundary

`FacialInstrument.run` is the single acquisition-to-evidence boundary. It accepts explicit RGB `uint8` images and either two image-bound `Observation` objects or a concrete `LandmarkProvider`. The provider is only an observation source. It does not choose downstream coordinate frames, invent region-specific scores, or own calibration.

`Observation → Anatomy → CanonicalState → AnatomicalField → MeasurementProfile → InstrumentResult` is a refinement of one shared state, not a chain of mutually incompatible representations. Each derived object carries, or is bound to, the common coordinate convention. Metric functions receive the state, not arbitrary image pairs that invite another hidden alignment.

The package keeps the inherited Wave 1/2 mathematical modules and APIs. Those are a research inventory, not an alternative path around the active observation contracts. The CLI names the distinction: `analyze` is Wave 3; `compare` remains legacy.

## Observation truth, repeatability, and missingness

An observation contains source raster SHA-256, source height/width, schema identity, ordered landmark names, original coordinates, and complete successful perturbation samples. Its metadata records the provider/version/model digest, applied transformations, failures, and the explicit noise assumption. Immutable arrays are backed by copied immutable bytes, so making a NumPy view writable cannot silently change a hashed observation.

A MediaPipe result and an operator annotation are different provenance types. Synthetic laboratory coordinates are a third type. They are not relabeled to make a pipeline look automatic. The schema binds anatomical naming, not certainty. Repeated stable model predictions can all share the same bias.

No probability is synthesized for landmarks when the provider does not expose one. Test-time perturbation spread is a repeatability estimate. The residual pixel-noise floor is an operator/modeling assumption. These jointly support conditional sensitivity analysis, not a validated localization error distribution.

Primary detection must contain exactly one face. Zero detections and multiple detections are separate failures. Failed augmented detections remain listed; they are not counted as successful observations. No-face failure does not invoke a geometric template as a substitute.

## Anatomical grammar

The grammar is shared by registration anchors, the face support envelope, eye/nose/mouth/jaw regions, contour membership, region holdouts, and measurement denominators. Stable MediaPipe indices are never sorted by apparent X position. Semantic annotations may be reordered explicitly through their names; both members of a pair must present identical ordering.

Model contours are predicted sample paths. Region masks are deterministic polygons/hulls of that grammar. They are not semantic skin segmentation, and their area is not a ground-truth anatomical area. A source-domain field may be supported in one part of a polygon and withheld in another. Coverage counts remain separate from residual magnitude.

## Registration is part of the measurement

The display uses B's eye-centered chart. Both images are resampled once into that chart with constant borders. A warped support image determines which pixels came entirely from real source pixels. Reflected or padded pixels never become evidence.

Off-image vertices are excluded from registration and triangulation, not merely hidden from final measurement rows. The eye anchors defining the chart must be observed in both sources. Missing regions have explicit withheld measurements and raw support masks.

Primary alignment uses robust, orientation-preserving, symmetric-scale Procrustes fitting over a declared anchor set. The scale is the square root of the ratio of centered weighted variances. This choice differs intentionally from one-way predictive least squares: exchanging A/B produces the inverse transformation. Landmark residuals use a matching symmetric normalizer.

Huber reweighting makes the chosen gauge less vulnerable to isolated bad anchors. It does not declare those points incorrect or remove their residuals from the profile. A dissimilar but visible face pair must not be rejected simply because its geometry differs.

Each eyes/nose/mouth/jaw holdout removes the evaluated region's anchors and refits the gauge. The resulting residual and its signed change expose registration dependence. These are repeated measurements of the same observed data, not independent confirmations.

## One anatomical field; independent image correspondence

The anatomical field is a piecewise-affine interpolation over A's Delaunay triangulation. It supplies dense displacement, per-triangle Jacobian, determinant, Green–Lagrange strain, principal log stretch, conditioning, triangle validity, and pixel validity. Negative-orientation/folded, collapsed, badly conditioned, out-of-domain, and unsupported regions are excluded without disappearing from diagnostic counts.

This is a 2D geometric map inferred from landmarks. It is not recovered physical motion, a 3D surface, tissue strain, or an independently observed dense anatomical correspondence.

Farnebäck flow is computed separately from the two registered gray `uint8` rasters, in both directions. Admissibility requires actual source/target support, a texture-observability threshold, and acceptable forward/backward cycle error. Its disagreement with the anatomical interpolation is retained as an explicit diagnostic. No confidence-weighted average can make these fields validate each other.

The current field is directional A→B. Exchanging the input images changes triangulation and area weighting; dense deformation summaries are not claimed to be exchange-symmetric. Scalar landmark residuals and their relational distances use symmetric normalization and are tested for exchange symmetry.

## Conditional uncertainty

An uncertainty draw selects a complete centered perturbation shape from each observation. This retains cross-landmark covariance within that sample. It adds the explicitly assumed independent residual pixel noise, refits registration, recomputes the normalizer, and evaluates the regional residual.

The run exports every draw. The 2.5th/97.5th sample quantiles are labeled conditional observation-sensitivity bounds. They do not include unidentified shared detector bias, population variation, 3D projection uncertainty, or human resemblance variability. The number of draws, seed, sampling protocol, and noise assumptions are recorded.

## Recipe identity and calibration

A recipe includes engine source digest, numeric library versions, anatomical grammar, configuration, source-resolution classes, provider fingerprints, noise floors, successful sample counts, and attempted/accepted sampling protocols. This is conservative: changing a display/implementation source file can invalidate a previous whole-engine recipe even when one metric's formula is unchanged. Fine-grained metric recipe versioning is future work, not silently approximated here.

Cohort ingestion reads verified bundles, not copied-looking scores. It requires one compatible recipe and explicit requested metrics. Missing measurements do not cause silent row filtering. Each row retains the contributing source manifest. Synthetic runs remain synthetic.

Repeated images/declared subjects form connected components. Bootstrap resampling operates over those components, acknowledging dependence among pair rows. Query images and supplied query subject identifiers cannot appear in the reference. If subject identifiers are absent, the instrument cannot detect same-person reuse across different images; it does not identify people to fill that gap.

A per-metric percentile is a rank against a particular cohort. The system does not average marginal percentiles and label that a calibrated joint statistic. Metric redundancy is exposed through a Spearman matrix, with constant columns explicitly undefined.

## Evidence publication and failure behavior

Successful runs produce strict JSON, no-pickle NPZ arrays, source/canonical rasters, an offline report, and an execution event chain. Each exported array is described by dtype, shape, and digest. File hashes and a manifest digest bind the artifacts. `verify` checks integrity; `replay` recomputes the instrument from stored observations and rasters under matching code and numeric runtime.

Replay is not detector reinference. Nor is a manifest an independent attestation: someone able to replace all files and hashes can replace the bundle. The journal records actual function-stage activity during this run; it is not a model's hidden reasoning or a reconstructed historical explanation.

Expected acquisition/geometry failures raise `InstrumentFailure`, preserving the real partial journal in `exception.journal_jsonl`. It ends at the failed stage, not a fictional completion event. CLI input failures produce a nonzero exit. Unexpected programming errors remain ordinary tracebacks instead of being rewritten as reassuring quality warnings.

## Cost and boundaries

Dense raster work is bounded by the configured 64..1024 canvas. The observation/sensitivity budget is explicit. No native model or weights are downloaded during inference. Separate independent source views are the intended input; blended screen recordings first require the acquisition inspector and deliberate source selection.

Automatic pose neutralization, expression neutralization, independent edge refinement, semantic occlusion segmentation, real detector-accuracy calibration, learned human perceptual scoring, and live proprietary telemetry are not delivered by this state model. The architecture retains the inputs and diagnostics required to add those capabilities without pretending they already exist.
