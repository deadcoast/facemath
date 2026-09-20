"""Run the exact release and preserve commands, output, timing and exit codes."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence'/'wave7'


def run(label: str, argv: list[str]) -> None:
    path=OUT/f'{label}.json'
    if path.exists():raise FileExistsError(path)
    start=datetime.now(timezone.utc).isoformat()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    result=subprocess.run(argv,cwd=ROOT,env=env,text=True,capture_output=True)
    record={'command':argv,'cwd':str(ROOT),'started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),
            'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    path.write_text(json.dumps(record,indent=2))
    (OUT/f'{label}.txt').write_text(result.stdout+'\n'+result.stderr)
    print(label,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    run('final-doctor',[sys.executable,'-m','facemath','doctor'])
    run('final-source-tests',[sys.executable,'-m','pytest','-q',f'--junitxml={OUT/"final-source-tests.xml"}'])
    run('final-laboratory',[sys.executable,'-m','facemath','joint-lab','--out',str(OUT/'laboratory')])
    run('final-pair-replay',[sys.executable,'-m','facemath','replay',str(OUT/'laboratory/pair_02_mouth_8px'),'--out',str(OUT/'source-pair-replay')])
    run('final-study-replay',[sys.executable,'-m','facemath','replay',str(OUT/'laboratory/qualification'),'--out',str(OUT/'source-study-replay')])
    run('final-browser',[sys.executable,str(ROOT/'scripts/browser_qa_wave7.py'),str(OUT/'laboratory/pair_02_mouth_8px/report.html'),'--out',str(OUT/'browser')])
    run('final-inherited-browser',[sys.executable,str(ROOT/'scripts/browser_qa_wave6.py'),str(OUT/'laboratory/pair_02_mouth_8px/report.html'),'--out',str(OUT/'inherited-browser')])
    run('final-wheel-build',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',str(ROOT/'dist'),str(ROOT)])
    run('final-wheel-validation',[sys.executable,str(ROOT/'scripts/check_wheel.py'),str(ROOT/'dist/facemath-0.7.0-py3-none-any.whl'),
         '--tests',str(ROOT/'tests'),'--bundle',str(OUT/'laboratory/qualification'),
         '--pair-bundle',str(OUT/'laboratory/pair_02_mouth_8px'),'--out',str(OUT/'wheel-check')])


if __name__=='__main__':main()
