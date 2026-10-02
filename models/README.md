# Optional FaceLandmarker model

The automatic provider needs a compatible MediaPipe Python installation and a
local `face_landmarker.task`. The instrument never downloads one implicitly.
The 0.11 native pilot retains an explicitly downloaded model under
`evidence/native-pipeline-20261002/pilot/dataset/assets/face_landmarker.task`.
The `dataset fetch examples/native-dlib-pilot.json --out NEW_DIRECTORY` command
can retrieve the pinned assets explicitly on another machine.

Official documentation:
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker

The documented version-1 model artifact can be obtained explicitly by the operator:

```sh
curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o models/face_landmarker.task
```

In PowerShell:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "models/face_landmarker.task"
Get-FileHash "models/face_landmarker.task" -Algorithm SHA256
```

The retrieved version-1 artifact has SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
`--model-sha256` compares against the expected digest. A hash verifies byte
consistency; it is not an external authenticity attestation.

Historical Wave10 development could not retrieve the model. The 0.11 pipeline
actually executed it with locked MediaPipe 1.0.1 on Windows/Python 3.11 and
retained all 43 native observations. See the native-pipeline evidence for actual
process records. To run the single-image integration after installation:

```powershell
$env:FACEMATH_MODEL = (Resolve-Path "models/face_landmarker.task").Path
$env:FACEMATH_FACE = (Resolve-Path "face_a.png").Path
python -m pytest -q -m live_observer
```

Use images for which you have an appropriate right to perform this geometric analysis. The provider does not perform identity recognition.
