"""Distinct continuation after the retained CRLF metadata tooling failure."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
gates=base/'verified314';out=base/'release-final';out.mkdir(exist_ok=False)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONIOENCODING='utf-8',UV_OFFLINE='1')
env['PYTHONPATH']=str(root/'src')
records=[]
def run(name,command,cwd=root,source=root):
    current=dict(env,PYTHONPATH=str(source/'src'))
    record={'command':list(map(str,command)),'cwd':str(cwd),'started_utc':datetime.now(timezone.utc).isoformat(),
            'environment':{k:current.get(k) for k in ('PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','UV_OFFLINE')}}
    with (out/f'{name}.log').open('x',encoding='utf-8') as log:
        p=subprocess.run(record['command'],cwd=cwd,env=current,stdout=log,stderr=subprocess.STDOUT)
    record.update(name=name,returncode=p.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
    records.append(record)
    (out/'commands.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(name,p.returncode,flush=True)
    if p.returncode:raise SystemExit(p.returncode)

assert all(c['returncode']==0 for c in json.loads((gates/'commands.json').read_text()))
from facemath.observation.journal import code_fingerprint
fingerprint=code_fingerprint()
assert fingerprint==json.loads((gates/'environment.log').read_text())['engine_code_sha256']
wheel=next((gates/'wheels').glob('*.whl'));archive=out/'facemath-0.10.1.zip'
pack=[sys.executable,root/'scripts/package_release.py','--wheel',wheel,'--stage',out/'stage',
      '--output',archive,'--evidence','evidence/review-remediation-20261001']
run('selection',[*pack,'--dry-run'])
run('package',pack)
run('archive-byte-verify',[sys.executable,root/'scripts/verify_release.py',archive])
with zipfile.ZipFile(archive) as z:z.extractall(out/'extracted')
extracted=out/'extracted/facemath-0.10.1'
run('extracted-byte-verify',[sys.executable,extracted/'scripts/verify_release.py',extracted],extracted,extracted)
run('archive-import',[sys.executable,'-c','import facemath,json; from pathlib import Path; '
    'from facemath.observation.journal import code_fingerprint; '
    'assert Path(facemath.__file__).resolve().is_relative_to(Path.cwd()/"src"); '
    'print(json.dumps({"import":facemath.__file__,"code_sha256":code_fingerprint()}))'],extracted,extracted)
run('archive-tests',[sys.executable,'-m','pytest','-q','-ra',f'--junitxml={out / "archive.xml"}',
                     f'--basetemp={out / "archive-temp"}'],extracted,extracted)
lab=out/'archive-lab'
run('archive-milestone',[sys.executable,'-m','facemath','milestone-lab','--out',lab],extracted,extracted)
for kind,relative in [('qualification','positive/qualification'),('pair','admitted_pair')]:
    run(f'archive-{kind}-verify',[sys.executable,'-m','facemath','verify',lab/relative],extracted,extracted)
    run(f'archive-{kind}-replay',[sys.executable,'-m','facemath','replay',lab/relative,'--out',out/f'{kind}-replay'],extracted,extracted)
result={'completed':True,'engine_code_sha256':fingerprint,'commands':records,
        'prior_source_wheel_gates':str(gates/'commands.json'),
        'prior_failed_wrapper':str(base/'validation314-final.json'),
        'continuation_reason':'Release metadata parser corrected for actual CRLF wheel headers; application source and wheel bytes unchanged.',
        'packaging_regressions':['packaging-final311.json','packaging-final314.json'],
        'archive':json.loads((out/'archive-byte-verify.log').read_text())}
(out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result['archive']),flush=True)
