# Current priority after Wave 07

Joint full-shape error sets now feed the active instrument. The next gate remains live native observation against independent real references; neither synthetic covariance controls nor interval tightening closes it. Freeze an observation study, check reference8 semantics and visibility, compare coverage across declared acquisition strata, then decide whether further contour/3D annotation and nonlinear error geometry are justified. Do not tune the selected model on final evaluation data or expand formula names as a substitute for observation validation.

The previous roadmap is retained below as historical planning, not a claim that all items are complete.

---

> Retained Wave 3 record. Current extensions and priorities are in [Wave 4 design](WAVE4_DESIGN.md) and [Wave 4 roadmap](WAVE4_ROADMAP.md).

# Next priorities after Wave 3

## 1. Close the live observation validation gap

Run the optional MediaPipe integration against locally installed, hash-recorded weights on selected single-face source images. Evaluate pixel-coordinate conventions, all semantic index mappings, augmentation remapping, face-absence/ambiguity behavior, clipping, glasses, facial hair, oblique pose, and expressions. Compare with independently annotated landmarks. Measure localization error and failure rates separately from augmentation repeatability.

Acceptance is not “the detector returned 478 numbers.” It is a repeatable observation package with audited correspondences and known conditions under which measurements are withheld or qualified. This is the largest unclosed release boundary.

## 2. Establish an independently labeled error and reference dataset

Version dataset inclusion criteria, source provenance, subject grouping, camera/pose/scale conditions, annotation protocol, inter-annotator disagreement, and held-out splits. Use independent labels to estimate error floors and test interval coverage. Build reference cohorts only after selecting a stable measurement recipe. Avoid fitting thresholds on the pair being evaluated.

A useful initial instrument may report residuals and within-tolerance fractions without a population percentile. Do not fill the absence of a cohort with invented sample statistics.

## 3. Add independent contour and occlusion observations

The present contour and dense anatomical fields derive from the same landmarks. Independent gradient/edge evidence, visibility/occlusion masks, and disagreements with model contours should enter as separate observation channels with their own provenance and failure modes. They must not silently move landmarks to make the pair look more similar.

## 4. Address nuisance factors beyond 2D similarity transforms

Quantify pose/expression effects before introducing 3D neutralization. A predicted Z coordinate is not metric depth. Add camera/pose or 3D reconstruction only with observable scale conventions and independent tests. Keep raw projected measurements alongside any neutralized result.

## 5. Measure sensitivity of the instrument itself

Extend holdout registration to prespecified alternative anchor sets; test confidence/texture thresholds, contour sampling, triangulation dependence, field resolution, and point weighting. The present mesh field is A→B. Develop a justified symmetric surface statistic rather than averaging two arbitrary outputs and calling it canonical.

## 6. Repair inherited mathematics by scientific value

Use `capabilities` to select concrete missing methods only when they improve the observation problem. Full transport optimization, correct SVCCA, SPD metrics, higher-dimensional topology, mesh curvature and valid information-theoretic estimators need their own analytic controls. Do not prioritize the number of formula names over the ability to observe, diagnose, and reproduce a meaningful facial comparison.

## Release discipline

Future waves retain parent hashes, source-difference inventories, raw test logs, explicit skipped integrations, model/data fingerprints, and deterministic replay tolerances. Passing tests or a polished report never substitute for live observation validation or a correctly defined scientific claim.

## Wave 06 continuation

Implemented: uncertain-reference normalization, correlated-expression propagation, proper-rotation refit enclosures and interactive reference8 inspection integrated into the existing canonical evidence bundle. Parent geometry and prospective acquisition remain in place.

The empirical priority remains unchanged: a real local observer run against independently annotated, group-disjoint portraits under a frozen domain. Only after that, evaluate whether conservative set widths should be tightened using justified joint error models, rather than increasing apparent confidence through additional dependent relations. Independently validated contours/visibility and 3D geometry follow their own evidence requirements.
