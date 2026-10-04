# Source image selection

The application's existing observer requires one clearly visible face. Complete
the image-input workflow from WORKSPACE_DESIGN.md by allowing an explicit crop
before observation, without requiring an external image editor.

Decode previews and analysis using stored-raster orientation. Preview resizing
is display-only; the source slice is never resampled. A crop is the integer
half-open slice [left, top, right, bottom], with at least eight pixels on each
axis. The first cropped pixel center corresponds to original center (left, top).
The crop-to-original transform is translation by that offset. Pointer selection
and accessible numeric controls share this slice; full-image reset restores the
original input path and original uploaded bytes are always retained.

Input preparation owns the crop. The existing instrument receives a new,
image-bound cropped raster and runs every existing channel without a second
alignment or any change to measured coordinates. The full-image path continues
to analyze the original upload. No automatic face choice, synthetic annotation,
reference qualification or recalibration is inferred from selecting a crop.

Saved input records include original encoded and decoded hashes, slice bounds,
analysis dimensions and decoded hash, lossless analysis-file hash, and the
crop-to-original transform. The download retains original files, decoded source
previews and lossless crops alongside the existing result bundle. A portable
input-selection page identifies the crop frame and links to the analysis.
Published result bundles are never edited to insert this metadata afterward.

Use indexed-raster and EXIF controls for exact pixels and coordinates. Exercise
actual pointer selection, keyboard/numeric edits, reset, replacement, invalid
input, native comparison, saved history and portable download. Keep the inherited
baseline, full source/wheel/archive checks and all real execution records.
