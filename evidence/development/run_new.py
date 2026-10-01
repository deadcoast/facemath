import subprocess,os,json,time
from pathlib import Path
root=Path('/mnt/data/facemath_wave_09');out=Path('/mnt/data/wave9_work');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
cmd=['python','-m','pytest','-q','tests/test_wave9_acquisition.py','tests/test_wave9_integration.py','tests/test_wave9_study.py','tests/test_wave3_artifacts.py']
t=time.time()
with (out/'new_tests_first.log').open('w') as f:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
(out/'new_tests_first_execution.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode,'elapsed_seconds':time.time()-t},indent=2))
