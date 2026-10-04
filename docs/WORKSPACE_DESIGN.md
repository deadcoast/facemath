# Local image-pair application

The approved image-input-to-results workflow needs an executable entry point.
`facemath app --model …` serves a local form, observes a selected pair,
and opens the existing measured structure, regional and edge readouts. The full
report, raw arrays, original uploaded bytes and execution logs remain available.

The parent is the actual 0.21.0 archive, SHA-256
`bccd8509edcd81c2fe5992477405ba852aa0b929d83d34fd13a7403a66a281ea`.
Its preceding extracted-package run reports 1240 passes and one Windows
directory-symlink skip in `runs/edge-comparison-20261004/release-02`.
This inherited run is reused, not represented as a newly executed baseline.

The worker executes the existing `analyze` command with the pinned model digest,
default 96 sensitivity draws, all observation, intrinsic, deformation, optical
flow and edge channels enabled. The application adds no alignment or score.
Browser tolerance controls only the existing landmark/edge threshold. Research
calibration and qualification remain available through the established CLI;
the initial workspace makes observed comparisons without claiming those bounds.

Each submission creates a unique directory. Uploaded PNG/JPEG bytes are retained;
the worker decodes stored-raster coordinates and browser result previews use
that same orientation. In 0.22, browser selection previews could display EXIF
orientation; 0.25 replaces them with stored-raster previews and explicit integer
crops, as defined in [SOURCE_SELECTION_DESIGN.md](SOURCE_SELECTION_DESIGN.md).
One native worker runs at a time. Status moves from running to complete, failed,
cancelled or interrupted. Results become available after the process exits
successfully and its bundle verifies. Failed work keeps inputs and logs; retry
creates a new run. Restart marks unfinished status interrupted, never complete.
Refresh and reconnect restore saved work. Cancellation terminates the worker.

The server binds only 127.0.0.1, uses a random capability path, checks the Host
and Origin, limits upload size, and serves only explicit run artifacts. No
arbitrary local paths or remote model downloads are accepted. Images and results
persist in the caller-selected output directory. This is a local application,
not a multiuser or internet-hosting service.

The established dark report tokens remain owned by artifacts.py and are shared
with the workspace. Existing readout renderers own numerical rows. The desktop
input is a two-image workbench; narrow screens stack it. Result tables scroll
horizontally without compressing rejection text into unusably tall columns.
The local app shows pending, failure and reconnect actions; no percentage
progress is fabricated for the worker's opaque computation.

Exercise browser file selection through actual native inference, result row
values, image decoding, blend controls, downloads, refresh and narrow scrolling.
Exercise invalid files, no-face inputs, concurrency, cancellation and restart
recovery. Then perform the required release checks on the final source.
