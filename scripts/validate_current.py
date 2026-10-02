"""Execute source, installed-wheel and extracted-archive gates in a fresh run.

Run with the locked development interpreter. No historical evidence is modified.
Parent baseline, native-model and human-reference checks are separate gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile

from package_release import package
from verify_release import verify


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--evidence',action='append',default=[])
    parser.add_argument('--retain-evidence',type=Path,help='new directory below repository evidence/ for pre-seal execution records')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    env.pop('PYTHONPATH',None)
    env['UV_CACHE_DIR']=str(root/'.uv-cache')
    records=[]

    def run(name,command,cwd=root,source=None):
        current=dict(env)
        if source is not None:
            current['PYTHONPATH']=str(source/'src')
        record={'name':name,'command':list(map(str,command)),'cwd':str(cwd),
                'started_utc':datetime.now(timezone.utc).isoformat(),
                'environment':{k:current.get(k) for k in ('PYTHONPATH','UV_CACHE_DIR','UV_OFFLINE',
                    'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','PYTHONIOENCODING')}}
        with (out/f'{name}.log').open('x',encoding='utf-8') as log:
            result=subprocess.run(record['command'],cwd=cwd,env=current,stdout=log,stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
        records.append(record)
        (out/'commands.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(f'{name}: exit {result.returncode}',flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    def workflow(prefix,cwd,source):
        lab=out/f'{prefix}-lab'
        run(f'{prefix}-milestone',[sys.executable,'-m','facemath','milestone-lab','--out',lab],cwd,source)
        for kind,relative in [('qualification','positive/qualification'),('pair','admitted_pair')]:
            bundle=lab/relative
            run(f'{prefix}-{kind}-verify',[sys.executable,'-m','facemath','verify',bundle],cwd,source)
            run(f'{prefix}-{kind}-replay',[sys.executable,'-m','facemath','replay',bundle,'--out',out/f'{prefix}-{kind}-replay'],cwd,source)
        return lab

    run('environment',[sys.executable,'-c',
        'import json; from facemath.observation.journal import environment; print(json.dumps(environment(),indent=2))'],source=root)
    run('lock-check',[shutil.which('uv') or 'uv','lock','--check'])
    run('source-tests',[sys.executable,'-m','pytest','-q','-ra',f'--junitxml={out / "source.xml"}',
                        f'--basetemp={out / "source-temp"}'],source=root)
    lab=workflow('source',root,root)
    run('wheel-build',[shutil.which('uv') or 'uv','build','--wheel','--out-dir',out/'wheels'])
    wheels=list((out/'wheels').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('expected exactly one current wheel')
    run('wheel-check',[sys.executable,root/'scripts/check_wheel.py',wheels[0],
        '--tests',root/'tests','--bundle',lab/'positive/qualification',
        '--pair-bundle',lab/'admitted_pair','--out',out/'wheel-check','--milestone'])
    if args.retain_evidence is not None:
        retained=(root/args.retain_evidence).resolve()
        if retained==root/'evidence' or not retained.is_relative_to(root/'evidence'):
            raise ValueError('retained evidence must be a new directory below repository evidence/')
        retained.mkdir(parents=True,exist_ok=False)
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix in {'.log','.json','.xml'}:
                shutil.copyfile(path,retained/path.name)
        shutil.copytree(lab,retained/'laboratory')
        installed=retained/'wheel-check';installed.mkdir()
        for path in (out/'wheel-check').iterdir():
            if path.is_file() and path.suffix in {'.json','.xml'}:
                shutil.copyfile(path,installed/path.name)
        if not any(retained.is_relative_to((root/name).resolve()) for name in args.evidence):
            args.evidence.append(retained.relative_to(root).as_posix())
    version=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    archive=out/f'facemath-{version}.zip'
    plan=package(root,wheels[0],out/'stage',archive,args.evidence,dry_run=True)
    (out/'selection.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    sealed=package(root,wheels[0],out/'stage',archive,args.evidence)
    with zipfile.ZipFile(archive) as z:z.extractall(out/'extracted')
    extracted=out/'extracted'/f'facemath-{version}'
    byte_check={'archive':sealed,'extracted':verify(extracted)}
    (out/'package-bytes.json').write_text(json.dumps(byte_check,indent=2),encoding='utf-8')
    run('archive-tests',[sys.executable,'-m','pytest','-q','-ra',f'--junitxml={out / "archive.xml"}',
                         f'--basetemp={out / "archive-temp"}'],extracted,extracted)
    workflow('archive',extracted,extracted)
    # Post-extraction execution belongs in an adjacent result, never in the
    # already sealed inventory. The artifact's digest remains stable.
    result={'completed':True,'archive':verify(archive),'commands':records,
            'limits':['synthetic controls, not human-reference validation',
                      'installed wheel shares interpreter dependencies',
                      'only this executed host/runtime is validated']}
    (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['archive'],indent=2))


if __name__=='__main__':main()
