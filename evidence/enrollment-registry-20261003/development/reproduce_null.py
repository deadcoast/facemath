"""Retain the pre-fix explicit-null registry bypass without altering old evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from facemath.observation.study_lab import make_fixture_study
from facemath.observation.study import write_json_new, read_json

run = Path(__file__).resolve().parent
spec, executor, _ = make_fixture_study(run/'null-control',development=6,calibration=1,evaluation=1)
write_json_new(run/'null-registry.json', None)
command = [sys.executable,'-m','facemath','study-freeze',str(spec),'--registry',str(run/'null-registry.json'),
           '--out',str(run/'null-bypass-lock')]
start = datetime.now(timezone.utc).isoformat()
result = subprocess.run(command,capture_output=True,text=True)
record = dict(command=command,started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),
    returncode=result.returncode,expected_returncode=2,stdout=result.stdout,stderr=result.stderr,
    observer_calls=executor.calls, defect='Explicit JSON null registry was silently treated as absent')
write_json_new(run/'null-bypass.result.json',record)
print(json.dumps(record,indent=2))
assert result.returncode == 2, 'An explicitly requested malformed registry must not be ignored'
