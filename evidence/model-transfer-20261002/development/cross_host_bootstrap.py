from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/mnt/c/Users/ryanf/github/_active/facemath')
run=root/'runs/model-transfer-20261002'
prior=root/'runs/native-portability-20261002'
local=Path('/tmp/facemath-transfer-cross-host-20261002')
local.mkdir(exist_ok=False)
env=dict(os.environ,UV_CACHE_DIR=str(prior/'linux-cache'),UV_PROJECT_ENVIRONMENT=str(local/'env'),
         PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
commands=[('cross-host-install',[prior/'linux-tools/uv','sync','--locked','--offline','--python',sys.executable,'--extra','dev','--extra','observer']),
          ('cross-host-check',[local/'env/bin/python',run/'cross_host_control.py',run/'portable-windows-evaluation'])]
for name,cmd in commands:
    command=list(map(str,cmd)); start=datetime.now(timezone.utc).isoformat()
    with (run/(name+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    with (run/(name+'.json')).open('x') as stream:
        json.dump(dict(command=command,cwd=str(root),started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),
                       returncode=result.returncode,environment={k:env[k] for k in ('UV_CACHE_DIR','UV_PROJECT_ENVIRONMENT','PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}),stream,indent=2)
    print(name,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
