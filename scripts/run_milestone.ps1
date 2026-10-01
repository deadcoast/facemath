param([string]$Output = "runs/wave10")
$ErrorActionPreference = "Stop"
python -m facemath milestone-lab --out $Output
if ($LASTEXITCODE -ne 0) { throw "Milestone laboratory failed; retain its artifacts." }
python -m facemath overview "$Output/positive/qualification"
if ($LASTEXITCODE -ne 0) { throw "Positive qualification verification failed." }
python -m facemath overview "$Output/negative/qualification"
if ($LASTEXITCODE -ne 0) { throw "Negative qualification verification failed." }
python -m facemath overview "$Output/admitted_pair"
if ($LASTEXITCODE -ne 0) { throw "Pair verification failed." }
