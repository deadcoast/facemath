# Reference-coordinate input repair

Parent 0.19.0 archive SHA-256:
`387acf24ae6714b610a83075be4746444cb8b4bb4ba4264c3d5a55e0398416c4`.
The immediately preceding extracted-archive run is the unchanged parent baseline:
1,219 passed, one directory-symlink skip. Its command, timestamps, raw log and XML
are copied with an explicit reuse record; it is not described as a newly run test.

`REPOSITORY_REVIEW.json` D05 identifies the raster-center defect. The original
`docs/01_image-analyze.md` and `docs/02_image-analyze.md` motivate retaining actual
input data rather than treating plausible summaries as measurements. The active
`REFERENCE_PROTOCOL` fixes zero-based source pixel centers. A known indexed-raster
browser control reproduced a maximum 0.535-pixel export error in the parent.
The old browser QA constructed clicks with the same incorrect inverse transform,
so it could pass despite that error.

## One coordinate convention

The decoded PNG is drawn using image edges. Source point (x,y) is drawn at
`(tx + (x + 0.5) scale, ty + (y + 0.5) scale)` in canvas content-box CSS units.
Pointer conversion removes border/padding and CSS scaling before applying the
inverse. Canvas backing dimensions use device-pixel ratio; the actual rounded
backing/content ratio sets the drawing transform. Zoom preserves the source point
under its event position. Markers and the magnifier use the same center offset.
This follows the [HTML canvas bitmap and image drawing model](https://html.spec.whatwg.org/multipage/canvas.html#drawing-images).

Capture uses the existing 0.001-pixel export resolution before the domain check.
This prevents float-rounded browser events from excluding exact edge centers
(the retained narrow-view failure was about 1.4e-6 pixels outside). Inputs beyond
that quantization remain rejected; no general off-image clamping is introduced.
The browser oracle uses the actual Canvas CTM and DOM content box, not the
application's sourceAt/canvasAt helpers. It includes all four edge centers,
fractional DPR, CSS scale, padding/borders, zoom, mouse/keyboard pan, subpixel
nudging, marker and magnifier location, actual download, and narrow layout.

## Draft and intake compatibility

Reference JSON stays `facemath.reference.v4`: the scientific coordinate contract
has not changed. Metadata identifies `facemath.reference-workbench.v20` and its
capture convention. New drafts use a separate namespace bound to image, reference
protocol, reference kind and rater. Existing ref4 drafts are retained and are not
silently imported or numerically corrected. Restore/Keep actions are explicit;
invalid bindings are not imported and unavailable storage does not block export.
No historical reference file or model is rewritten.

The same workflow review reproduced a second parent defect: synthetic prepared
workspaces emitted `human_blind`, while intake required `synthetic_fixture`.
Workspace generation now selects the reference kind from the frozen execution
kind. The browser export remains byte-retained and its coordinates, metadata and
reference digest survive the existing import/freeze path without modification.

## Scope and execution

The ordinary Windows regression also exposed a redirected-output failure:
`overview` printed Unicode decoration to CP1252 and raised UnicodeEncodeError.
The readout now accepts the actual stream encoding, retains its UTF-8 appearance
where representable, and falls back to ASCII decoration plus reversible escapes
for other labels. Numeric formatting is unchanged. A strict CP1252 subprocess
and the retained native pair exercise this path without a UTF-8 override.

This change repairs data entry and provenance routing, not observer accuracy.
Automated raster clicks and renderer controls are explicitly synthetic. Native
workspaces still need actual independent reference annotations. Existing native
pair observations are reused only for regression; there is no new cohort, fitting,
calibration, threshold tuning or anatomical validation claim.

Preserve the parent failures and intermediate browser-control failures. Execute
source, outside-tree wheel and extracted archive suites, milestone verify/replay,
raw-state invariance and byte checks for the final code. Browser interactions and
intake/freeze exercise the changed behavior; the broader checks protect inherited
comparison and qualification behavior.
