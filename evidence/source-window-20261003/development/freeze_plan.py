from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

from facemath.observation.dataset import verify_dataset
from facemath.observation.journal import code_fingerprint, environment
from facemath.observation.source_cohort import select_members
from facemath.observation.study import freeze_study, read_json
from facemath.observation.transfer import load_capsule

run = Path(__file__).resolve().parent
root = run.parents[1]
evidence = root/'evidence/source-window-20261003'
dataset = evidence/'dataset'
shutil.copytree(run/'intake/dataset', dataset)
verified = verify_dataset(dataset)
assert verified['cases'] == 60
shutil.copyfile(root/'evidence/model-transfer-20261002/helen-model.json', evidence/'helen-model.json')
capsule = read_json(evidence/'helen-model.json')
model = load_capsule(capsule)
recipe = read_json(dataset/'recipe.json')
plan = recipe['cohort']['plan']
prior_recipe = read_json(root/'evidence/bound-utility-20261002/dataset/recipe.json')
prior_plan = prior_recipe['cohort']['plan']
assert select_members(prior_plan, recipe['cohort']['inventory']) == prior_recipe['cohort']['selection']
assert all(plan[k] == prior_plan[k] for k in ('archive','candidate_prefix','selection','seed','model','coordinate_origin','box_margin'))
full_plan = json.loads(json.dumps(plan))
full_plan['selection_offset'] = 0
full_plan['split_counts']['evaluation'] = 120
full_selection = select_members(full_plan, recipe['cohort']['inventory'])
assert [r['image'] for r in full_selection[:60]] == [r['image'] for r in prior_recipe['cohort']['selection']]
assert full_selection[60:] == recipe['cohort']['selection']

def tokens(path):
    conversions = read_json(path/'conversions.json')
    return {v[key] for v in conversions for key in ('source_encoded_sha256','source_raster_sha256')}

checks = []
new_groups = {r['group_id'] for r in recipe['images']}
for relative in ('bound-utility-20261002/dataset','source-cohort-20261002/dataset',
                 'model-transfer-20261002/dataset','native-pipeline-20261002/pilot/dataset'):
    prior = root/'evidence'/relative
    overlap = tokens(prior) & tokens(dataset)
    prior_groups = {r['group_id'] for r in read_json(prior/'recipe.json')['images']}
    assert not overlap and not new_groups & prior_groups
    checks.append({'dataset': str(prior.relative_to(root)), 'source_hash_overlap': sorted(overlap),
                   'group_overlap': sorted(new_groups & prior_groups)})
study = freeze_study(dataset/'study.spec.json', run/'prospective-lock', frozen_model=capsule)
record = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), engine_code_sha256=code_fingerprint(),
    study_sha256=study.digest, capsule_sha256=capsule['content_sha256'], calibration_sha256=model.digest,
    model_bytes_sha256=sha256(capsule['model_json'].encode()).hexdigest(), radius=model.quantile,
    enrolled=len(study.cases), eligible=len(study.included), split_counts=plan['split_counts'],
    selection_window=recipe['cohort']['selection_window'], selection=recipe['cohort']['selection'],
    source_overlap_checks=checks, donor_first_window_reproduced=True,
    prior_plan_sha256=sha256(json.dumps(prior_plan,sort_keys=True).encode()).hexdigest(),
    plan_sha256=sha256((root/'examples/helen-transfer-window.json').read_bytes()).hexdigest(),
    inference_executed=False, fit_performed=False, empirical_admission=False, environment=environment(),
    limit='source separation and published-label agreement do not establish person independence or anatomical accuracy')
with (run/'prospective-freeze.json').open('x') as stream:
    json.dump(record, stream, indent=2)
shutil.copyfile(run/'prospective-freeze.json', evidence/'prospective-freeze.json')
print(json.dumps({k:v for k,v in record.items() if k not in ('environment','selection')}, indent=2))
