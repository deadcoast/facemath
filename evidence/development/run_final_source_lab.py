from pathlib import Path
import subprocess,json,os,time
r=Path('/mnt/data/facemath_wave_09');e=r/'evidence/wave9';e.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');records=[]
for name,args in [('source_tests',['python','-m','pytest','-q']),('measured_lab',['python','-m','facemath','measured-lab','--out',str(e/'laboratory')])]:
 t=time.time()
 with (e/(name+'.log')).open('w') as f:p=subprocess.run(args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
 records.append({'name':name,'command':args,'returncode':p.returncode,'elapsed_seconds':time.time()-t})
 (e/'source_execution.json').write_text(json.dumps(records,indent=2))
 if p.returncode:break
