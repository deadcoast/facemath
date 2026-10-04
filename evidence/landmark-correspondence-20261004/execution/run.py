from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys, time
root=Path(__file__).resolve().parents[2]; out=Path(__file__).resolve().parent
name=sys.argv[1]; cwd=Path(sys.argv[2]).resolve(); cmd=sys.argv[3:]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONIOENCODING='utf-8',UV_CACHE_DIR=str(root/'.uv-cache'))
env['PYTHONPATH']=str(cwd/'src')
r={'command':cmd,'cwd':str(cwd),'started_utc':datetime.now(timezone.utc).isoformat(),'environment':{k:env.get(k) for k in ('PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','UV_CACHE_DIR','FACEMATH_MODEL','FACEMATH_FACE')}}
start=time.monotonic()
with (out/(name+'.log')).open('x',encoding='utf-8') as f: result=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
r.update(returncode=result.returncode,seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
with (out/(name+'.json')).open('x',encoding='utf-8') as f: json.dump(r,f,indent=2)
print(json.dumps(r,indent=2)); sys.exit(result.returncode)
