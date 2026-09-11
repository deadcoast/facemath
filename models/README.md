# Optional FaceLandmarker model

No pretrained model is bundled and the instrument never downloads one implicitly. The automatic provider needs a compatible MediaPipe Python installation and a local `face_landmarker.task`.

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

Record the SHA-256 you obtained. `--model-sha256` compares against an operator-supplied expected digest; this release does not invent a known-good model hash because the artifact was not retrieved here. A hash alone verifies consistency with your expected artifact, not the provenance of an untrusted download.

The development environment could not resolve/download the dependency/model, so real-model inference was not run. A mocked API test is not a replacement. To run the actual integration after installation, supply a legitimate single-face test image and the model:

```powershell
$env:FACEMATH_MODEL = (Resolve-Path "models/face_landmarker.task").Path
$env:FACEMATH_FACE = (Resolve-Path "face_a.png").Path
python -m pytest -q -m live_observer
```

Use images for which you have an appropriate right to perform this geometric analysis. The provider does not perform identity recognition.
