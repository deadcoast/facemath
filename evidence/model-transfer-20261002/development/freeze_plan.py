from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

from facemath.observation.dataset import verify_dataset
from facemath.observation.journal import code_fingerprint, environment
from facemath.observation.study import freeze_study, read_json
from facemath.observation.transfer import load_capsule

run=Path(__file__).resolve().parent
root=run.parents[1]
evidence=root/'evidence/model-transfer-20261002'
dataset=evidence/'dataset'
shutil.copytree(run/'intake-grouped/dataset',dataset)
assert verify_dataset(dataset)['eligible']==20
capsule=read_json(evidence/'helen-model.json')
model=load_capsule(capsule)
study=freeze_study(dataset/'study.spec.json',run/'prospective-lock',frozen_model=capsule)

def source_hashes(path):
    conversions=read_json(path/'conversions.json')
    return {v[key] for v in conversions for key in ('source_encoded_sha256','source_raster_sha256')}

checks=[]
for prior in ('bound-utility','source-cohort'):
    donor=root/f'evidence/{prior}-20261002/dataset'
    assert not source_hashes(donor)&source_hashes(dataset)
    checks.append(dict(dataset=str(donor.relative_to(root)),original_encoded_and_decoded_aliases=0))
recipe=read_json(dataset/'recipe.json')
record=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),engine_code_sha256=code_fingerprint(),
    study_sha256=study.digest,capsule_sha256=capsule['content_sha256'],calibration_sha256=model.digest,
    radius=model.quantile,enrolled=20,split_counts=dict(development=0,calibration=0,evaluation=20),
    candidate_annotations=recipe['cohort']['selection'][0]['candidate_count'],
    candidate_filename_roots=recipe['cohort']['selection'][0]['candidate_groups'],
    selection=recipe['cohort']['selection'],source_overlap_checks=checks,
    plan_sha256=sha256((root/'examples/ibug-transfer-cohort.json').read_bytes()).hexdigest(),
    inference_executed=False,fit_performed=False,environment=environment(),
    limit='source disjointness checks do not establish person independence or absence from native model training')
with (run/'prospective-freeze.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k not in ('environment','selection')},indent=2))
