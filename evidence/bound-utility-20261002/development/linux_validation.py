"""Fresh retained Linux execution under the final source fingerprint."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/mnt/c/Users/ryanf/github/_active/facemath')
run = root/'runs/bound-utility-20261002'
prior = root/'runs/native-portability-20261002'
local = Path('/tmp/facemath-bound-utility-20261002')
local.mkdir(exist_ok=False)
env = dict(os.environ, UV_CACHE_DIR=str(prior/'linux-cache'),
    UV_PROJECT_ENVIRONMENT=str(local/'env'), PYTHONPATH=str(root/'src'),
    MPLCONFIGDIR=str(local/'mpl'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
    FACEMATH_MODEL=str(root/'evidence/bound-utility-20261002/dataset/assets/face_landmarker.task'),
    FACEMATH_FACE=str(root/'evidence/bound-utility-20261002/dataset/inputs/case_00000.png'))
python = local/'env/bin/python'
commands = [
    ('linux-install', [prior/'linux-tools/uv', 'sync', '--locked', '--offline', '--python',
                       sys.executable, '--extra', 'dev', '--extra', 'observer']),
    ('linux-tests', [python, '-m', 'pytest', '-q', '-ra', f'--basetemp={local}/tests',
                    '-o', f'cache_dir={local}/cache', f'--junitxml={run}/linux.xml']),
    ('linux-native', [python, '-m', 'facemath', 'dataset', 'run',
                     root/'evidence/bound-utility-20261002/dataset', '--out', run/'linux-native']),
    ('linux-verify', [python, '-m', 'facemath', 'verify', run/'linux-native']),
]
for name, command in commands:
    argv = list(map(str, command))
    start = datetime.now(timezone.utc).isoformat()
    with (run/(name+'.log')).open('x', encoding='utf-8') as stream:
        result = subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(command=argv, cwd=str(root), started_utc=start, returncode=result.returncode,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        environment={k: env[k] for k in ('PYTHONPATH', 'UV_CACHE_DIR', 'UV_PROJECT_ENVIRONMENT',
                                       'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'FACEMATH_MODEL', 'FACEMATH_FACE')})
    with (run/(name+'.json')).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
