# Automated native dataset operations

Use the locked Python 3.11 observer environment for the supplied pilot. Its
tested MediaPipe version is `1.0.1`, with model SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The model comes from Google's versioned [Face Landmarker bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task).
The [upstream dlib XML](https://github.com/davisking/dlib/blob/46fa4a281a35b16aed6577250d7dabbda0e74a2b/examples/faces/training_with_face_landmarks.xml)
explicitly identifies its landmarks as shape-predictor outputs on PASCAL VOC
photographs. The recipe retains this attribution, revision and all asset hashes.
The retained Boost license covers dlib; it does not transfer photograph rights.

Version 0.11.1 uses `opencv-contrib-python>=4.9,<5` as the sole `cv2` owner.
OpenCV's [packaging instructions](https://pypi.org/project/opencv-contrib-python/)
require selecting one wheel variant. Older FACEMATH observer environments may
contain both `opencv-python` and `opencv-contrib-python`, with different binaries
winning on different hosts. Preserve old environments for their original replay;
create a fresh environment for new acquisition, for example by setting
`UV_PROJECT_ENVIRONMENT=runs/observer-0111` before `uv sync --locked --python 3.11
--extra dev --extra observer`. Do not reuse a mixed environment. Native preflight
refuses multiple variants, unsupported versions or a metadata/binary mismatch.

Ubuntu native execution needs `libegl1` and `libgles2`; the native CI job installs
them explicitly. `scripts/check_native_runtime.py --dataset DATASET --out NEW_RUN`
requires actual inference through the isolated worker and retains failures. The
full release validator invokes it before pytest, so a native abort cannot erase
the test controller's diagnostics. A successful process gate is not a coverage
or accuracy claim.

`dataset fetch` is the explicit network operation. It bounds downloads by pinned
sizes and rejects changed bytes. A failed fetch retains the received artifact and
a failure receipt. `dataset prepare` works offline, requires the entire declared
XML image inventory, converts every face box using the frozen margin and origin,
and retains original encoded photographs, labels, model and recipe. It performs
no model inference. Missing reference support is a prospective exclusion with a
reason; successful detections cannot retroactively change enrollment.

`dataset run DATASET --out NEW_RUN` copies the complete dataset and invokes the
existing study freezer, native worker, finalizer and numerical replay. A model
override is allowed only when it matches the frozen digest. The worker receives
the crop and pinned observer declaration, never its reference labels or split.
Native process stdout/stderr, raw normalized model outputs and actual attempt
receipts remain available in `session/cases/` and the qualification acquisition.

The pilot freezes two source photographs for development, four for calibration
and three for evaluation. Its 43 face crops are grouped by source photograph;
there are nine groups, not 43 independent samples. The fixed 90% calibration
target requires rank five from four calibration groups, so the radius remains
unbounded. The fitted center and scatter are retained and transferable, but this
pilot does not enable `--qualification` deployment. Machine references cannot
satisfy the independent-reference admission gate, even with a changed exposure
declaration. Research calibration reports its machine-reference scope.

To expand the data, create a new pinned recipe with explicit image groups,
lineages and splits. Keep related photographs in the same group/split when that
relationship is known. Group names do not establish person-level independence.
The dlib adapter accepts all 68 parts per box, then applies the existing
reference8 index mapping. It does not invent occlusion labels or establish that
the teacher's anatomical convention exactly matches MediaPipe's vertices.
Do not tune splits, margins, alpha or model settings on evaluation outcomes and
then relabel that run as the original experiment.

Successful pipeline directories are immutable, self-contained transfer units.
`verify RUN` checks byte inventories, original-source conversions, stage bindings
and stored replay equality. `replay RUN --out NEW_REPLAY` recomputes the fitted
model and evaluation from retained observations under the original code and
numerical environment. It performs no native re-inference. A fresh native run
is a new experiment; compare its arrays explicitly rather than assuming bitwise
determinism. Python source files and new retained evidence use Git byte-preserving
attributes so checkout line-ending conversion cannot change their fingerprints.

Failures keep their actual partial files and journals. A failed pipeline has
`failure.json` and `failure.events.jsonl` and no valid success manifest. Nested
validation/finalization faults retain uniquely named `*.failure-*` directories.
Inspect these before choosing recovery. Existing acquisition sessions can use
`study-run` for the inherited receipt-aware resume: started cases are not
silently retried. Re-run the complete pipeline only into a new destination.

The release gate supports the real pipeline as well as synthetic controls:

```shell
python scripts/validate_current.py --out runs/release-001 \
  --native-dataset evidence/native-pipeline-20261002/pilot/dataset \
  --evidence evidence/native-pipeline-20261002 \
  --retain-evidence evidence/new-release-preseal
```

This repeats full tests, `milestone-lab`, verify/replay, installed-wheel import
and byte checks, and actual native acquisition from source, installed wheel and
extracted archive. It records commands, environments, return codes and raw logs.
An executed Windows run does not establish Linux/macOS native compatibility;
the hosted matrix records those platforms separately when actually executed.

The 0.11.1 local Ubuntu/WSL2 pilot completes all 43 observations and its same-host
replay. Its measured coordinate and augmentation arrays equal the Windows run
byte-for-byte. A Linux bundle replayed on Windows nevertheless fails the strict
qualification comparison: derived simultaneous scores differ in their final
floating-point bits, while every saved numerical array remains byte-identical.
Package version equality alone is therefore insufficient for cross-OS exact
replay. Preserve the original bundle and runtime; the rejected recomputation is a
distinct diagnostic result. No tolerance or saved result was changed to admit it.
The measured Linux-to-Windows control has 12 identical saved arrays and a maximum
absolute derived-score difference of `7.815970093361102e-14`. This value diagnoses
that one run; it is not a new permissible replay tolerance.
