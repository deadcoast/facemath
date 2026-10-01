import subprocess,os,json,time
from pathlib import Path
p=Path('/mnt/data/wave9_work')
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
r=[]
for name,cmd in [('parent_install',['python','-m','pip','install','--no-deps','--no-build-isolation','-e','.']),('parent_tests',['python','-m','pytest','-q'])]:
 start=time.time()
 with (p/(name+'.log')).open('w') as f: result=subprocess.run(cmd,cwd=p/'parent/facemath_wave_08',stdout=f,stderr=subprocess.STDOUT,env=env)
 r.append({'command':cmd,'returncode':result.returncode,'seconds':time.time()-start,'log':name+'.log'})
 (p/'parent_execution.json').write_text(json.dumps(r,indent=2))
 if result.returncode:break
(p/'parent_done').write_text('done')
