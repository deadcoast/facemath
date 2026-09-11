# Next development priorities after Wave 04

## P0 · Live observation and independent labeling

Acquire the official model through an authorized network, pin its digest/runtime, execute the existing live-observer test, and run the reference workflow on independent source portraits with documented labels. Do not remove the skipped status until actual inference succeeds. Runtime availability is not localization accuracy.

Build a sufficiently broad, group-disjoint development/calibration/evaluation dataset under explicit usage rights. Report acquisition exclusions, no-face/multiple-face outcomes and rater disagreement. Use the source-coordinate benchmark to identify systematic nose/chin/eye definition differences before calibrating. This is the current release gate for a human accuracy-qualified result.

## P1 · Error-model qualification, not numerical decoration

Validate held-out coverage under the actual acquisition domain, including pose, expression, resolution, blur and occlusion slices defined before evaluation. Check group-size matching and source/subject/session dependence. Compare bias-aware alternatives using development only. Keep the current fixed-gauge reference envelope distinct from future joint true-shape-refit intervals.

## P2 · Independent contour and visibility acquisition

The new edge scanner observes gradients but is landmark-guided and can select hair, shadows or texture. Add independently labeled contour/visibility references and test the scanner's acceptance and localization errors. Do not promote edge support into anatomical confidence without that evidence. Only then consider constrained contour refinement and dense uncertainty propagation.

## P3 · Reproducible deployment and compatibility

Test Windows/macOS with pinned optional observer environments, audit third-party runtime network behavior, test persistent draft restore and signed dataset provenance where available, and define a carefully reviewed compatibility protocol for presentation-only code changes. Current bundle replay intentionally requires exact code/runtime compatibility.

## Deferred deliberately

Human population resemblance calibration, 3D shape recovery, dense 478-vertex accuracy claims and model-internal telemetry remain distinct projects requiring their own data. The inherited mathematical inventory remains audited; function-name coverage is not implementation completeness. Do not expand formulas while the observation release gate remains open.
