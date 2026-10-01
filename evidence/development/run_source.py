import subprocess,time,json,os
from pathlib import Path
out=Path('/mnt/data/wave9_work');t=time.time();cmd=['python','-m','pytest','-q'];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with (out/'source_full_first.log').open('w') as f:r=subprocess.run(cmd,cwd='/mnt/data/facemath_wave_09',env=env,stdout=f,stderr=subprocess.STDOUT)
(out/'source_full_first_execution.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode,'elapsed_seconds':time.time()-t},indent=2))
