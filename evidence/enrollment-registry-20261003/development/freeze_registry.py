"""Register five retained source plans, then freeze a distinct integration window."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

from facemath.observation.dataset import verify_dataset
from facemath.observation.enrollment import load_registry
from facemath.observation.journal import code_fingerprint, environment
from facemath.observation.source_cohort import select_members
from facemath.observation.study import freeze_study, read_json, write_json_new
from facemath.observation.transfer import load_capsule

run = Path(__file__).resolve().parent
root = run.parents[1]
evidence = root/'evidence/enrollment-registry-20261003'
exports = evidence/'exports'; exports.mkdir(exist_ok=False)
sources = [('pilot', 'native-pipeline-20261002'), ('lfpw', 'source-cohort-20261002'),
           ('helen-donor', 'bound-utility-20261002'), ('ibug-transfer', 'model-transfer-20261002'),
           ('helen-holdout', 'source-window-20261003')]
records = []

def cli(name, *args, expected=0):
    command = [sys.executable, '-m', 'facemath', *map(str, args)]
    start = datetime.now(timezone.utc).isoformat()
    with (run/(name+'.log')).open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    write_json_new(run/(name+'.json'), dict(command=command, cwd=str(root), started_utc=start,
        finished_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode, expected_returncode=expected))
    assert result.returncode == expected, name

for name, directory in sources:
    study = root/'evidence'/directory/'preseal/native-pipeline/study'
    cli('export-'+name, 'enrollment', 'export', study, '--out', exports/(name+'.json'))
    receipt = read_json(exports/(name+'.json'))
    cases = json.loads(receipt['study_json'])['cases']
    records.append(dict(name=name, source=study.relative_to(root).as_posix(),
        study_sha256=receipt['manifest']['study_sha256'], receipt_sha256=receipt['content_sha256'],
        enrolled=len(cases), eligible=sum(c['eligibility']=='included' for c in cases)))

scope = ('The retained dlib pilot, LFPW original-PTS cohort, HELEN donor, IBUG frozen transfer and '
         'HELEN roots 60-119 holdout; one complete source lock per plan, not independent replicas. '
         'Other studies, unknown person aliases and native training enrollment are outside this snapshot.')
args = [x for name,_ in sources[:4] for x in ('--receipt', exports/(name+'.json'))]
cli('registry-initial', 'enrollment', 'build', *args, '--scope', scope, '--out', evidence/'prior-registry.json')
cli('registry-extend', 'enrollment', 'build', '--receipt', exports/'helen-holdout.json',
    '--scope', scope, '--prior', evidence/'prior-registry.json', '--out', evidence/'registry.json')
registry = read_json(evidence/'registry.json')
rows = load_registry(registry)
assert len(rows) == sum(r['enrolled'] for r in records)
cli('reject-prior-check', 'enrollment', 'check', evidence/'registry.json',
    root/'evidence/source-window-20261003/preseal/native-pipeline/study',
    '--out', evidence/'prior-overlap.json', expected=2)
cli('reject-prior-pipeline', 'dataset', 'run', root/'evidence/source-window-20261003/dataset',
    '--frozen-model', root/'evidence/source-window-20261003/helen-model.json',
    '--registry', evidence/'registry.json', '--out', run/'refused-prior-pipeline', expected=2)
assert not (run/'refused-prior-pipeline/session').exists()
assert 'registry overlaps' in read_json(run/'refused-prior-pipeline/failure.json')['detail']

shutil.copytree(run/'intake/dataset', evidence/'dataset')
dataset = evidence/'dataset'
verified = verify_dataset(dataset)
assert verified['cases'] == 12
shutil.copyfile(root/'evidence/source-window-20261003/helen-model.json', evidence/'helen-model.json')
capsule = read_json(evidence/'helen-model.json'); model = load_capsule(capsule)
recipe = read_json(dataset/'recipe.json'); plan = recipe['cohort']['plan']
prior_recipe = read_json(root/'evidence/source-window-20261003/dataset/recipe.json')
prior_plan = prior_recipe['cohort']['plan']
assert all(plan[k] == prior_plan[k] for k in ('archive','candidate_prefix','selection','seed','model','coordinate_origin','box_margin'))
assert select_members(prior_plan, recipe['cohort']['inventory']) == prior_recipe['cohort']['selection']
assert plan['selection_offset'] == 120 and plan['split_counts'] == dict(development=0, calibration=0, evaluation=12)
study = freeze_study(dataset/'study.spec.json', run/'prospective-lock', frozen_model=capsule, enrollment_registry=registry)
cli('check-new-window', 'enrollment', 'check', evidence/'registry.json', study.root,
    '--out', evidence/'new-window-check.json')
record = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), engine_code_sha256=code_fingerprint(),
    study_sha256=study.digest, registry_sha256=registry['content_sha256'], registry_scope=scope,
    registered_sources=records, registered_cases=len(rows), check=study.data['enrollment_registry']['check'],
    capsule_sha256=capsule['content_sha256'], calibration_sha256=model.digest,
    model_bytes_sha256=sha256(capsule['model_json'].encode()).hexdigest(), radius=model.quantile,
    plan_sha256=sha256((root/'examples/helen-registry-window.json').read_bytes()).hexdigest(),
    selection_window=recipe['cohort']['selection_window'], selection=recipe['cohort']['selection'],
    inference_executed=False, fit_performed=False, empirical_admission=False, environment=environment(),
    purpose='prospective registry/native integration; no new scientific acceptance threshold or independent sample count from repeated executions')
write_json_new(evidence/'prospective-freeze.json', record)
print(json.dumps({k:v for k,v in record.items() if k not in ('environment','selection')}, indent=2))
