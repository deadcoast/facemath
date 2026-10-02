"""Verify transferred bytes, then require actual preflight rejection on Linux."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from facemath.observation.deployment import unpack_snapshot
from facemath.observation.study_io import verify_qualification_bundle

run = Path('/mnt/c/Users/ryanf/github/_active/facemath/runs/numerical-runtime-20261002')
root = run.parents[1]
receipt = json.loads((run/'windows-transfer.json').read_text())
raw = (run/'windows-qualification.zip').read_bytes()
assert sha256(raw).hexdigest() == receipt['sha256']
local = Path('/tmp/facemath-numeric-replay-negative-20261002')
local.mkdir(exist_ok=False)
source = local/'qualification'; source.mkdir()
unpack_snapshot(raw, source)
verification = verify_qualification_bundle(source)
assert sha256((source/'manifest.json').read_bytes()).hexdigest() == receipt['source_manifest_sha256']
output = local/'recomputed'
command = [sys.executable, '-m', 'facemath', 'replay', str(source), '--out', str(output)]
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH=str(root/'src'))
started = datetime.now(timezone.utc).isoformat()
result = subprocess.run(command, cwd=local, env=env, capture_output=True, text=True)
record = dict(command=command, cwd=str(local), started_utc=started,
    finished_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
    stdout=result.stdout, stderr=result.stderr, verification=verification,
    archive_sha256=receipt['sha256'], destination_created=output.exists(),
    source_manifest_unchanged=sha256((source/'manifest.json').read_bytes()).hexdigest() == receipt['source_manifest_sha256'],
    scope='actual transferred byte verification and incompatible-runtime preflight; not successful numerical replay')
with (run/'transferred-replay-control.result.json').open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record, indent=2))
assert result.returncode == 2 and 'requires matching numerical runtime' in result.stderr
assert not output.exists() and record['source_manifest_unchanged']
