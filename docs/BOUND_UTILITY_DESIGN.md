# Frozen bound utility study

The parent is the actual 0.12.0 release archive, SHA-256
`9cd1c27496e4c0d95906f57d8cda2d9b769b7a702fc98ff3a32bd1642240d49d`.
Its extracted-package baseline completed before these changes: 1033 passed,
one unavailable Windows directory-symlink test skipped. That execution is reused
explicitly; its raw log, command receipt and a fresh 8781-file extracted byte
check are retained. This is not represented as a new test execution.

## Review and design

Coverage alone does not describe useful geometry. Qualification already retains
the complete development bias, full16 scatter and independently calibrated radius,
but lacks a systematic width/delivery summary. Add a diagnostic report to that
same qualification, recomputed during exact replay. Keep its calibration binding,
all source arrays and the existing held-out coverage calculation unchanged.

Use the existing `EllipticSource` and `SourceRelations` graph. Its source chart is
conditioned by the observed ocular basis, with the error-model bias subtracted
only from set centers. It never edits observed coordinates or fits another
alignment. Report each supported landmark's marginal ellipse diameter, the
uncertain eye-midpoint span, mouth width and mouth-width/eye-span interval.
All geometry is derived from the same full16 ellipsoid, including cross-landmark
dependence. The ratio divides by the uncertain reference-set eye span; a span
that may vanish makes that ratio unavailable. Widths in observed-eye-span units
are descriptive scale units, not a fixed reference denominator or an accuracy
score. Existing floating-point enclosure limits still apply.

Retain every enrolled case and eight-point denominator by split and category.
Excluded cases, failed observations, unavailable radii and missing dependencies
have reasons, never zero widths. Publish per-case intervals and complete-group
delivery, plus min/median/max widths with explicit finite counts. Development
and calibration diagnostics are labelled separately from held-out evaluation.
Do not introduce a utility acceptance threshold or select a fit from evaluation.
Reference availability in validation rows is not a visibility judgment.

## Prospective HELEN study

Freeze `examples/helen-utility-cohort.json` before inference. Use the pinned public
300W repack already retained from the LFPW study, now selecting from the 330
original `helen/testset` image/PTS pairs. The initial pre-inference filename audit
found two selected faces with the same photograph root; its plan/intake remains
retained as superseded, never observed. The replacement selection ranks numeric
photograph roots by SHA-256(seed, NUL, root path), then takes the lowest-ranked
original member within each selected root using SHA-256(seed, NUL, image path).
Require the declared numeric `source_face` filename convention, normalize leading
zeros in source roots and select at most one face per root. Rank roots independently
of how many faces they contain. Assign consecutive blocks of 20 development,
20 calibration and 20 evaluation sources. Exclude mirror derivatives.
Original full-precision PTS conversion, 0.25 crop margin,
reference8 mapping, native model/runtime, alpha 0.1, shrinkage 0.02 and floor
0.001 remain fixed before outcomes. Fit a separate development model and radius
for this domain; neither the old LFPW fit nor its evaluation is changed.

Each selected photograph root is a group, not an assertion of distinct identity.
The complete retained member inventory reproduces both root and member selection.
Exact aliases are rejected by intake. Unknown relationships,
visibility and native-model training overlap remain unknown. Published PTS labels
are research agreement references, not a new human testing campaign or verified
anatomical equivalence. No empirical admission is requested.

## Implementation and validation

Add the report to joint qualification only; other fit families explicitly report
unsupported utility. Reuse frozen category radii without pooled fallback. Test
dependency cancellation, scale/rotation behavior, missing/excluded denominators,
unavailable radius and eye-span refusal, unchanged raw arrays and fit bindings.
Run the HELEN pipeline through source, an isolated installed wheel and the final
extracted archive. Verify/replay each bundle and compare all observed and fitted
arrays. Run full tests and milestone controls under the final code fingerprint.
Record failures and actual environment; hosted scheduling is a separate gate.
