$env:PYTHONPATH = (Resolve-Path '.').Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& '..\runs\enrollment-registry-20261003\env311\Scripts\python.exe' -m pytest -q ../tests/test_annotation_coordinates.py ../tests/test_terminal_encoding.py --basetemp=.pacer/run/annotation20
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '..\runs\structural-comparison-20261004\browser-env\Scripts\python.exe' ../scripts/browser_qa_annotation.py ../evidence/annotation-coordinates-20261004/workbench.html --out .pacer/run/browser20 --browser 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
exit $LASTEXITCODE
