param(
    [string]$Python = "$PSScriptRoot/runs/enrollment-registry-20261003/env311/Scripts/python.exe",
    [string]$Model = "$PSScriptRoot/evidence/workspace-20261004/runtime/face_landmarker.task",
    [string]$Output = "$PSScriptRoot/runs/comparisons",
    [int]$Port = 8812
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Supply -Python with your Python 3.11+ executable that has facemath and its observer dependencies installed.'
}
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw 'Supply -Model with your existing MediaPipe Face Landmarker .task file.'
}
$priorPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "$PSScriptRoot/src"
    & $Python -m facemath app --model $Model --out $Output --port $Port
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $priorPath
}
