import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base = Path(__file__).resolve().parent
name, cwd, *command = sys.argv[1:]
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
clock = time.monotonic()
with (base / (name + '.log')).open('x', encoding='utf-8') as log:
    result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
record = dict(command=command, cwd=cwd, started_utc=start,
              finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              elapsed_seconds=time.monotonic()-clock, returncode=result.returncode,
              environment={k:env.get(k) for k in ['PYTHONPATH','UV_CACHE_DIR','UV_PROJECT_ENVIRONMENT','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
with (base / (name + '.json')).open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record))
print((base / (name + '.log')).read_text(encoding='utf-8')[-7000:])
sys.exit(result.returncode)
