# Bilateral configuration in the existing instrument

The original mathematics document, section 5, calls for bilateral configuration.
Eye-width contrast alone does not describe horizontal centering or unequal
heights of paired corners. Add five fixed rows: outer eye corners, inner eye
corners, mouth corners, nose midline and chin midline.

For eye midpoint C and the existing eye vector E, paired horizontal residual is
dot(E,L+R-2C)/dot(E,E); vertical residual is cross(E,R-L)/dot(E,E).
These are the components of R minus the reflection of L across the eye-defined
midline. For a central point P, report its signed midline offset
dot(E,P-C)/dot(E,E), with vertical component identically zero. Its display
segment joins the midline projection to P. These definitions do not fit an axis,
register a reflected face, infer depth or change observed coordinates.

Keep A and B components, residual magnitudes, signed component changes and the
norm of the component change distinct. Equal residual magnitude can conceal a
reversal of asymmetry. A zero change means the same bilateral configuration,
not necessarily that either image is symmetric. Units are each source's own
eye span; pose and expression affect these projected quantities.

Use the very same SourceRelations instances, fitted centers, scatter and
per-source calibration radii as the intrinsic profile. Bounds on projected
components use the uncertain eye vector/denominator and shared dependencies.
Component intervals imply analytic rectangular outer bounds on magnitudes and
changes; these are not new independent probabilities or fitted extrema.
Unknown rows keep their five-row denominator. A missing radius or possibly-zero
reference eye span withholds reference bounds, while observed values remain.

The existing pair API, CLI, application and portable report expose the results.
Expandable rows locate observed versus mirrored/projected points on the existing
canonical images. Coordinates are transformed for display through the existing
source chart and canonical transform. They are never substituted into geometry.

Parent archive: facemath-0.22.0-local-application-20261004.zip, SHA-256
7c965167f1c9658d86c86818adc73a8e4837faa0a0d381c47d386387a5fea6fe.
Its bytes were checked again before edits. The inherited actual archive test
execution is retained at runs/workspace-20261004/release-01/archive-tests.log
and archive.xml: 1257 passed, one Windows directory-symlink skip. This reuses
the completed parent execution; it is not a claim of another baseline run.

Verify analytic controls, missingness, correlated translation cancellation,
swap/sign behavior and qualification's preservation of all observed arrays.
Exercise native image input and expandable measurement locations in a browser.
The scientific independent-reference gate remains open.
