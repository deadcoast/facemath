# Core regional comparison functions

Parent: `facemath-0.18.0-structure-20261004.zip`, SHA-256
`4c0e2933e6f61b45f17fd8e4942e5c6d862848ccda7c8649ce081c90a1a8c81b`.
The checkout was clean and synchronized at `26563714` before this pass.

The original analysis, `04_facial-recognition.md` sections 30 and 31, separates
movement from deformation and specifies polar decomposition of local affine
maps. The active instrument already constructs those maps but publishes mainly
regional positional RMS and combined log-stretch RMS. This pass implements the
component functions in the production pair engine.

## Displacement and support

For each fixed anatomical region, use corresponding supported landmarks in the
existing canonical state. The displacement is B minus A, divided by the existing
geometric-mean registered eye span. Its mean is the regional translation vector;
the RMS of mean-centered displacement is contour change after translation only.
Total squared RMS equals squared translation magnitude plus squared centered
RMS. No new rotation, scale, landmark assignment or alignment is fitted.

Keep every region and its declared landmark denominator. Export the count within
the existing tolerance and the frozen-inventory fraction range
`[within/total, (within+missing)/total]`. This is a missingness range, not a
statistical confidence interval. Unsupported points never contribute zero error.

## Finite deformation

Consume the existing A-to-B triangle Jacobians J. For positive, nonsingular,
supported maps, compute J = R U and H = log(U) from the SVD. Export the proper
rotation angle, symmetric logarithmic stretch, log area change trace(H), and
the deviatoric component H - trace(H) I / 2. Rigid rotations have zero stretch;
uniform expansion has zero deviatoric stretch. The energy identity is
`||H||_F^2 = trace(H)^2/2 + ||dev(H)||_F^2`.

The SVD construction follows [Higham's polar-decomposition formulas](https://eprints.maths.manchester.ac.uk/694/1/high86p.pdf),
and the trace/deviatoric split follows the definitions in
[Neff, Eidel and Martin](https://arxiv.org/abs/1505.02203), specialized here to 2D.
Their mechanical interpretation is not transferred to cross-person images.

Aggregate over the existing supported A-domain pixels per region, keeping the
original mask denominator and minimum support rule. Report expansion/compression,
rotation, anisotropic stretch and signed chart-axis shear separately. Preserve
RMS alongside signed means so cancellation cannot look like zero deformation.
The inherited A-domain triangulation and pixel weighting are explicit; no claim
of exchange-symmetric surface statistics is introduced. Arrays retain masks for
folded, collapsed and unsupported triangles.

These are projected model-contour measurements, not physical tissue strain,
independent contour observations, localization confidence or identity scores.
Attaching calibration cannot alter them. The full component arrays and recipes
travel through the existing evidence export and replay.

## Partial diagnostics

A valid primary registration must not be discarded solely because a regional
holdout leaves too few usable anchors. Retain that diagnostic's failure reason,
array mask and fixed denominator while continuing supported primary results.
Failed sensitivity refits likewise retain their requested draw slots; an interval
is unavailable if the complete requested refit experiment cannot be evaluated.

## Execution

Run the actual parent package baseline first. Check displacement and polar
decomposition against analytic transformations, missing regions, folds and
partial holdout failures. Exercise fresh source-disjoint pinned native portraits
with the existing frozen model, without fitting or changing its radii. Complete
source, outside-tree installed-wheel and final-archive tests plus milestone,
verify, replay, byte parity and observed-array invariance. Preserve failures and
publish the implementation as a separate synchronized release.
