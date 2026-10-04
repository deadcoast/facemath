# Regional spatial inspection

Continue the movement/deformation methods in REGIONAL_COMPARISON_DESIGN.md and
the original facial-recognition design sections 30–31. The production engine
already computes the quantities; this pass makes their locations and support
inspectable through normal image input, shared browser readout and saved report.

Use the existing CanonicalState and A-domain triangle map. Seven regions retain
their original landmark and pixel denominators. Display measured A/B landmarks
and signed displacement, regional translation and mean-centered RMS. Individual
landmarks keep source names, canonical pixel centers, normalized displacements
and residuals after subtracting the existing regional mean. Missing points retain
their slots and explicit null values. No fit or alternate correspondence occurs.

Raster maps use exactly the pixels admitted by regional aggregation, including
the three-landmark and eight-pixel rules. Positive rotation follows image-down
coordinates. Area colors encode log2(B/A area), with fixed endpoints 0.5 and 2;
rotation uses ±30 degrees; deviatoric logarithmic stretch uses 0 through ln(2).
These are display limits, not acceptance thresholds. Every view reports saturated
pixel counts, and numerical tables keep full values. Missing pixels are hatched;
pixels outside the declared region are dimmed. No smoothing fills missing data.

Triangle rows expose the original triangle ID, vertex IDs, regional pixel weight,
rotation, area ratio, anisotropic stretch and chart shear. Images, JSON and tables
derive from the same read-only selector. No new observed arrays or qualification
rules are introduced. Derived inspection data belongs at the instrument/report
boundary, never in the intrinsic source-frame relation arrays.

Parent archive and inherited execution are recorded in
evidence/regional-inspection-20261004/parent. Analytic controls check support,
pixel lookup, scale saturation, unavailable rows and nonmutation. Native browser
interaction checks disclosures, numeric values, image loading, download, reload
and narrow-screen access. Final source/wheel/archive execution remains required.
This work does not close the independent human-reference validation gate.
