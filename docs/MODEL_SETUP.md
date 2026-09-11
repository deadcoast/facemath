# Optional live model setup

FACEMATH does not implicitly download a model. The live adapter requires an installed compatible MediaPipe runtime and a local FaceLandmarker `.task` file. The supplied source tests include API fixtures; those do not establish actual inference.

From the source project on PowerShell:

```powershell
python -m pip install -e ".[dev,observer]"
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile models/face_landmarker.task
Get-FileHash models/face_landmarker.task -Algorithm SHA256
python -m facemath doctor --model models/face_landmarker.task
```

The upstream `latest` download can change. Retain the actual returned digest and pass it via `--model-sha256` during observation. Do not substitute a digest copied from an unexecuted example. Keep the runtime version and environment alongside the model:

```powershell
python -m pip freeze --all > environment.lock.txt
$env:FACEMATH_MODEL = (Resolve-Path models/face_landmarker.task).Path
$env:FACEMATH_FACE = (Resolve-Path portraits/independent_source.png).Path
python -m pytest -q -m live_observer
```

Use an authorized, original single-face image. A zero-face or multiple-face outcome is a legitimate failed acquisition, not a reason to fabricate fallback points. Passing this live test establishes that the runtime actually produced the expected observation. It still does not establish anatomical accuracy: run `validate-observer` against independently collected reference annotations afterward.

Release environment status: MediaPipe/model acquisition did not succeed; the live test remained skipped. Direct package acquisition and installation attempt logs are retained in `evidence/wave4/`. No model weights or third-party face datasets are shipped.

The HTML workbench/report are self-contained and make no remote requests. This is separate from upstream runtime network/telemetry policy, which FACEMATH has not audited or certified. Install pinned versions and enforce your own network policy where needed.

Official Python API and model setup:
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
