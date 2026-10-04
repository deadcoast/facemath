"""Confirm the final code refuses exact Linux-to-Windows numerical replay."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

run = Path(__file__).resolve().parent
root = run.parents[1]
env = dict(os.environ, PYTHONPATH=str(root/'src'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
commands = [
    ('cross-host-profile', [sys.executable,'-m','facemath','runtime-profile','--compare',
        str(run/'linux-native/evaluation/environment.json'),'--out',str(run/'cross-host-profile.result.json')]),
    ('cross-host-replay', [sys.executable,'-m','facemath','replay',str(run/'linux-native/evaluation'),
                          '--out',str(run/'cross-host-refused-replay')]),
]
records = []
for name, command in commands:
    started = datetime.now(timezone.utc).isoformat()
    with (run/(name+'.log')).open('x',encoding='utf-8') as log:
        proc = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    record = dict(command=command,cwd=str(root),started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),returncode=proc.returncode,
        environment={k:env[k] for k in ('PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')})
    with (run/(name+'.json')).open('x') as f:
        json.dump(record,f,indent=2)
    records.append(record)
    assert proc.returncode == 2, record
comparison = json.loads((run/'cross-host-profile.result.json').read_text())['comparison']
assert comparison['compatible'] is False
assert 'requires matching numerical runtime' in (run/'cross-host-replay.log').read_text()
result = dict(expected_refusal=True, exact_cross_os_replay=False, commands=records,
    differing_fields=[v['path'] for v in comparison['differences']],
    limit='Numerical-runtime mismatch is retained; no evidence bytes, tolerances or model parameters changed')
with (run/'cross-host.result.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(dict(expected_refusal=True, returncodes=[r['returncode'] for r in records],
    differing_fields=len(result['differing_fields'])),indent=2))
