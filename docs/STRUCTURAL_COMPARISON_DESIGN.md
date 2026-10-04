# Structural comparison in the existing instrument

The original analysis describes eye--nose, nose--mouth, mouth--chin, eye spacing
and bilateral configuration (`03_facial-mathematics.md`, section 5). The active
instrument already owns observations, registration and joint reference error
sets, but the CLI returns bundle metadata and its main readout omits these named
source measurements. This pass connects that use case to the existing graph.

Parent: `facemath-0.17.0-correspondence-20261004.zip`, SHA-256
`ea5fec7d4f162696d2c61390801147f7d38f86adbb37a5ead7f9ccd676b1d2a2`.
The source and evidence were committed and synchronized as `26c1fdc2` before
this pass. The inherited tests run from the actual extracted archive.

## Measurement contract

Use the same eight reference points and `SourceRelations` instances that feed
the intrinsic profile. Define fourteen frozen measurements: two eye widths,
inner-eye gap, mouth width, five vertical heights (eye-to-nose, eye-to-mouth,
eye-to-chin, nose-to-mouth, mouth-to-chin), three lateral offsets (nose, mouth,
chin), mouth-corner height difference, and eye-width asymmetry.

Let E be viewer-left eye midpoint to viewer-right eye midpoint and V a declared
point or midpoint displacement. Axial offset is dot(E,V)/dot(E,E); vertical
height is cross(E,V)/dot(E,E), positive toward the lower image under the normal
viewer-side eye ordering. Width is norm(V)/norm(E). These are projected ratios,
not physical depth or pose-neutral anatomy. Nominal values use observed points.

Bounds use the already fitted center/scatter/radius and the same shared latent
error graph. They retain repeated-point cancellation and the uncertain eye
span. If that span may vanish, the bound is unavailable. Nothing shifts the
observed coordinates or introduces a second alignment. Signed differences are
B minus A; absolute-gap classification uses the existing length/asymmetry
tolerances. Unknowns remain in the frozen fourteen-item denominator. Nominal
counts and reference-envelope classifications are separate results.

The pair API, CLI readout and existing report expose A, B, signed difference,
units, support and the available reference interval. No overall resemblance or
identity probability is derived. All component arrays and recipe coefficients
are exported and replayed through the existing bundle format.

## Publication defect

Windows `tempfile.mkdtemp` gives staging directories a private ACL. Renaming
those directories into a user's chosen output location preserves that ACL.
Six v0.17 pair outputs consequently denied reads to the normal user during
staging even though the execution account could validate them. Their parent
inheritance was restored without changing a file's bytes.

Use one random, create-only staging-directory helper for published outputs,
with the destination parent's ordinary access inheritance. Keep genuinely
private temporary extraction directories private. Preserve atomic rename,
collision refusal and caller-owned cleanup behavior.

## Execution

First reproduce the parent baseline. Implement analytic measurement/correlation
and missing-support checks, integrate the real pair path, and exercise ordinary
user access on Windows. Use actual native portraits with the frozen existing
model without fitting on new outcomes. Complete the inherited source, wheel,
archive, milestone and replay checks. Preserve failures and synchronize this
development pass as a separate commit.

## Observed integration defects

The first native transfer stopped before inference: its model capsule included
an enrollment registry, but `freeze_study` omitted decoded `image_shape` when
calling the inherited registry check. The source now passes those dimensions,
and the regression exercises public freezing with a registry-bearing capsule.
The failed `native-01` directory remains beside the corrected `native-02` run.

The native pair returns all fourteen values and retains 146 raw arrays when the
model is attached. All fourteen reference differences remain unresolved. The
twenty-source label-agreement audit uses the fixed existing tolerances and keeps
all twenty entries per measurement, including any missing values; it cannot
authorize fitting, empirical admission or an anatomical-equivalence claim.
