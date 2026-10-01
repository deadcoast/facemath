"""Create-only execution records for FACEMATH Wave 09 release validation."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json, os, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence'/'wave9'

def run(name,command):
    target=OUT/(name+'.execution.json')
    if target.exists():raise FileExistsError(target)
    start=datetime.now(timezone.utc).isoformat()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    completed=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    record={'command':command,'cwd':str(ROOT),'started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),
            'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr}
    target.write_text(json.dumps(record,indent=2))
    (OUT/(name+'.log')).write_text(completed.stdout+'\n'+completed.stderr)
    print(name,completed.returncode,flush=True)
    if completed.returncode:raise SystemExit(completed.returncode)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lab=OUT/'laboratory';pair=lab/'pair_02_mouth_8px'
    # Source-suite and complete lab are separately recorded by source_execution.json.
    if not(lab/'laboratory.json').exists():raise RuntimeError('execute measured-lab and retain source tests first')
    run('doctor',[sys.executable,'-m','facemath','doctor'])
    run('source_group_verify',[sys.executable,'-m','facemath','verify',str(lab/'query_group_b_02')])
    run('source_group_replay',[sys.executable,'-m','facemath','replay',str(lab/'query_group_b_02'),'--out',str(OUT/'source-group-replay')])
    run('source_pair_replay',[sys.executable,'-m','facemath','replay',str(pair),'--out',str(OUT/'source-pair-replay')])
    run('source_study_replay',[sys.executable,'-m','facemath','replay',str(lab/'qualification'),'--out',str(OUT/'source-study-replay')])
    run('browser',[sys.executable,str(ROOT/'scripts/browser_qa_wave9.py'),str(lab/'qualification/report.html'),str(pair/'report.html'),'--out',str(OUT/'browser')])
    run('joint_browser',[sys.executable,str(ROOT/'scripts/browser_qa_wave7.py'),str(pair/'report.html'),'--out',str(OUT/'joint-browser')])
    run('relation_browser',[sys.executable,str(ROOT/'scripts/browser_qa_wave6.py'),str(pair/'report.html'),'--out',str(OUT/'relation-browser')])
    run('wheel_build',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',str(ROOT/'dist'),str(ROOT)])
    run('wheel_check',[sys.executable,str(ROOT/'scripts/check_wheel.py'),str(ROOT/'dist/facemath-0.9.0-py3-none-any.whl'),
        '--tests',str(ROOT/'tests'),'--bundle',str(lab/'qualification'),'--pair-bundle',str(pair),'--out',str(OUT/'wheel-check')])

if __name__=='__main__':main()
