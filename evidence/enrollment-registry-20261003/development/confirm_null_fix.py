from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
from facemath.observation.study import read_json, write_json_new, freeze_study
from facemath.observation.journal import code_fingerprint, environment

run = Path(__file__).resolve().parent
root = run.parents[1]
evidence = root/'evidence/enrollment-registry-20261003'
invalid = run/'null-registry.json'
commands = [
    ('null-freeze-fixed', ['study-freeze',run/'null-control/study.json','--registry',invalid,'--out',run/'null-fixed-lock']),
    ('null-prior-fixed', ['enrollment','build','--receipt',evidence/'exports/pilot.json','--scope','malformed prior control',
        '--prior',invalid,'--out',run/'null-fixed-registry.json']),
    ('null-pipeline-fixed', ['dataset','run',evidence/'dataset','--registry',invalid,'--out',run/'null-fixed-pipeline']),
    ('null-transfer-fixed', ['dataset','run',evidence/'dataset','--frozen-model',evidence/'helen-model.json',
        '--registry',invalid,'--out',run/'null-fixed-transfer']),
]
records = []
for name,args in commands:
    command = [sys.executable,'-m','facemath',*map(str,args)]
    start = datetime.now(timezone.utc).isoformat()
    with (run/(name+'.log')).open('x',encoding='utf-8') as stream:
        result = subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
    record = dict(command=command,started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),
        returncode=result.returncode,expected_returncode=2)
    write_json_new(run/(name+'.json'),record); records.append(record)
    assert result.returncode == 2 and 'registry must be a snapshot object' in (run/(name+'.log')).read_text()
assert not (run/'null-fixed-lock').exists() and not (run/'null-fixed-registry.json').exists()
assert not (run/'null-fixed-pipeline/session').exists() and not (run/'null-fixed-transfer/session').exists()
write_json_new(run/'null-fixed.result.json',dict(all_routes_refused=True,observer_sessions_created=0,commands=records))
old = read_json(evidence/'prospective-freeze.json')
study = freeze_study(evidence/'dataset/study.spec.json',run/'prospective-lock-02',
    frozen_model=read_json(evidence/'helen-model.json'),enrollment_registry=read_json(evidence/'registry.json'))
assert old['check'] == study.data['enrollment_registry']['check']
record = {**old,'frozen_utc':datetime.now(timezone.utc).isoformat(),'engine_code_sha256':code_fingerprint(),
    'study_sha256':study.digest,'environment':environment(),
    'original_plan_freeze':'prospective-freeze.json', 'original_study_sha256':old['study_sha256'],
    'first_native_outcomes_already_known':True,
    'purpose':'Implementation requalification of exactly the original 12-source plan after explicit-null input repair; not new unseen enrollment; model, selection, registry and policy unchanged'}
write_json_new(evidence/'prospective-freeze-02.json',record)
print(json.dumps(dict(all_routes_refused=True,study_sha256=study.digest,engine_code_sha256=code_fingerprint()),indent=2))
