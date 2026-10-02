"""One retained Linux invocation keeps its temporary environment alive."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/mnt/c/Users/ryanf/github/_active/facemath')
run = root/'runs/source-cohort-20261002'
prior = root/'runs/native-portability-20261002'
local = Path('/tmp/facemath-source-cohort-final-20261002')
local.mkdir(exist_ok=False)
env = dict(os.environ, UV_CACHE_DIR=str(prior/'linux-cache'),
    UV_PROJECT_ENVIRONMENT=str(local/'env'), PYTHONPATH=str(root/'src'),
    MPLCONFIGDIR=str(local/'mpl'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
    FACEMATH_MODEL=str(root/'evidence/source-cohort-20261002/dataset/assets/face_landmarker.task'),
    FACEMATH_FACE=str(root/'evidence/source-cohort-20261002/dataset/inputs/case_00000.png'))
python = local/'env/bin/python'
commands = [
    ('linux-final-install', [prior/'linux-tools/uv', 'sync', '--locked', '--offline', '--python',
                           sys.executable, '--extra', 'dev', '--extra', 'observer'], 0),
    ('linux-final-tests', [python, '-m', 'pytest', '-q', '-ra', f'--basetemp={local}/tests',
                         '-o', f'cache_dir={local}/cache', f'--junitxml={run}/linux-final.xml'], 0),
    ('linux-final-native', [python, '-m', 'facemath', 'dataset', 'run',
        root/'evidence/source-cohort-20261002/dataset', '--out', run/'linux-final-native'], 0),
    ('linux-final-verify', [python, '-m', 'facemath', 'verify', run/'linux-final-native'], 0),
    ('linux-final-runtime', [python, '-m', 'facemath', 'runtime-profile', '--out', run/'linux-final-profile-v2.json'], 0),

]
for name, args, expected in commands:
    argv = list(map(str, args))
    start = datetime.now(timezone.utc).isoformat()
    with (run/(name+'.log')).open('x', encoding='utf-8') as stream:
        result = subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(command=argv, cwd=str(root), started_utc=start, returncode=result.returncode,
        expected_returncode=expected, finished_utc=datetime.now(timezone.utc).isoformat(),
        environment={k: env[k] for k in ('PYTHONPATH', 'UV_CACHE_DIR', 'UV_PROJECT_ENVIRONMENT',
                     'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'FACEMATH_MODEL', 'FACEMATH_FACE')})
    with (run/(name+'.json')).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(name, result.returncode, flush=True)
    if result.returncode != expected:
        raise SystemExit(result.returncode or 1)
