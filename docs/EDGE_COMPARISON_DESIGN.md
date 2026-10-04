# Paired image-edge displacement

The original `03_facial-mathematics.md` section 3 asks how aligned image
structures differ: double edges and contour displacement through a blend.
The active Wave 4 channel retained each image's gradient rays but exposed only
separate absolute-offset summaries. Those values do not measure A/B displacement.
This pass extends that existing channel in the single CanonicalState.

Parent: 0.20.0 archive SHA-256
`97ba5d1da07a37922c8d86954bcf57448eac94e1261e1fd6d662f36632181900`.
Its actual extracted-package run completed immediately before this pass with
1223 passes and one Windows directory-symlink skip. The unchanged extracted
6081-file inventory was reverified; original execution times and logs are retained
in `evidence/edge-comparison-20261004/parent`. No duplicate baseline is claimed.

## Shared rays

For each existing A contour ray `(p,n)`, sample **both** canonical image rasters
at `p + t*n` using the unchanged normal-gradient sampler. Repeat on the B contour
rays as a separate inventory. Do not subtract offsets from independently located
A/B rays: those coordinates differ. Do not fit another alignment, pair contours
by nearest neighbor, snap landmarks, or pool the two dependent inventories.

On rays accepted in both images, `delta = t_B - t_A` and the normal component
of image displacement is `delta*n`. Report signed mean, RMS, median and 90th
absolute displacement in canonical pixels and the existing registered eye span.
The direction is always B minus A, including B-anchored rays. Positive normal
direction is `(-tangent_y, tangent_x)`; it is not universally outward.

Reuse existing grayscale smoothing, Sobel, source erosion, bilinear remapping,
gradient floor, ambiguity, orientation and search-boundary checks. The numerical
operators follow the [OpenCV Sobel documentation](https://docs.opencv.org/4.13.0/d2/d2c/tutorial_sobel_derivatives.html)
and [remapping contract](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html).
This is a finite-window image diagnostic, not a claim of anatomical matching.

## Missingness and export

Every requested ray remains in its region/anchor denominator. Each image's
first failing rule is retained as a code: unsupported source, weak gradient,
ambiguous peak, oblique gradient or search-boundary maximum. Codes partition
the inventory and reproduce the existing sampler's accepted mask. A missing
anchor retains its complete requested denominator with an explicit reason.

For the existing geometric tolerance, export `[within/total,
(within+unresolved)/total]`. This is a missingness range, not a confidence
interval. With no paired supported rays, measurements are null and the range
is `[0,1]`. Raw rejected deltas use zero sentinels plus the false accepted mask;
zero alone never means measured agreement. Full per-image responses, offsets,
normals and rejection masks travel through arrays.npz and exact replay.

The normal measurement omits tangential displacement; a strong gradient may be
texture or lighting rather than the intended anatomical contour. Independent
contour/visibility accuracy remains open. The reference8 error model does not
calibrate this image channel. Adding calibration must preserve every observed
array, including these new diagnostics.

## Operational checks

Check analytic raster translations, sign reversal, identical images, cancelled
signed means, flat/ambiguous/truncated/padded rays, missing anchors and immutable
exports. Exercise retained native observations from the first fixed prior pair,
without new cohort selection or fitting. Compare inherited array bytes with the
parent and inspect API, terminal, actual report interactions, verify and replay.
Complete source, installed-wheel and extracted-archive qualification once the
source fingerprint is final.

The existing report owns table styling, scrolling and accessibility. Add a finite
eight-row table to its measurement readout; preserve its tokens and opacity
interaction. No new input form or visual redesign is introduced.
Two image overlays reuse the existing figure owner and marker legend to locate
accepted gradient candidates; rejected rays are omitted from the drawing only,
never from the numerical denominator. Coordinates are rounded only for display.
