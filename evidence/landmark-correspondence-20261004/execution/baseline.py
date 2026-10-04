from pathlib import Path
from hashlib import file_digest
from datetime import datetime, timezone
import json, os, subprocess, sys, zipfile
root=Path.cwd(); out=root/'runs/landmark-repair-20261004'
parent=root/'packages/facemath-0.16.0-enrollment-registry-20261003.zip'
record={'parent':str(parent),'started':datetime.now(timezone.utc).isoformat()}
with parent.open('rb') as f: record['sha256']=file_digest(f,'sha256').hexdigest()
(out/'parent.json').write_text(json.dumps(record,indent=2))
print('parent sha256 '+record['sha256'],flush=True)
with zipfile.ZipFile(parent) as z: z.extractall(out/'parent')
base=out/'parent/facemath-0.16.0'
env=dict(os.environ,PYTHONPATH=str(base/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
cmd=[sys.executable,'-m','pytest','-q','-ra',f'--basetemp={out / "parent-temp"}',f'--junitxml={out / "parent.xml"}']
record.update(command=cmd,cwd=str(base),environment={k:env.get(k) for k in ('PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')})
with (out/'parent-tests.log').open('x',encoding='utf-8') as f: result=subprocess.run(cmd,cwd=base,env=env,stdout=f,stderr=subprocess.STDOUT)
record.update(returncode=result.returncode,finished=datetime.now(timezone.utc).isoformat())
(out/'parent.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
sys.exit(result.returncode)
