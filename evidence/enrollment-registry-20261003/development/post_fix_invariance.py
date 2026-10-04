from hashlib import sha256
import json
from pathlib import Path
from facemath.observation.contracts import Observation, array_digest
from facemath.observation.study import read_json, write_json_new

run = Path(__file__).resolve().parent
old = run/'release/source-native'
new = run/'release-03/source-native'
rows = []
for path in sorted((old/'session/cases').glob('*/observation.json')):
    current = new/'session/cases'/path.parent.name/path.name
    before = Observation.from_dict(read_json(path))
    after = Observation.from_dict(read_json(current))
    arrays = {k:dict(before=array_digest(getattr(before,k)),after=array_digest(getattr(after,k))) for k in ('xy','samples')}
    assert all(v['before'] == v['after'] for v in arrays.values())
    rows.append(dict(case=path.parent.name,arrays=arrays,observation_json_bytes_equal=path.read_bytes()==current.read_bytes()))
assert len(rows) == 12
model_before = (old/'evaluation/localization_calibration.json').read_bytes()
model_after = (new/'evaluation/localization_calibration.json').read_bytes()
assert model_before == model_after
result = dict(initial_source_study=read_json(old/'study/manifest.json')['study_sha256'],
    final_source_study=read_json(new/'study/manifest.json')['study_sha256'],
    same_enrollment=True,model_bytes_unchanged=True,model_sha256=sha256(model_after).hexdigest(),
    cases=rows,arrays_checked=24,all_original_observation_arrays_unchanged=True,
    interpretation='same planned sources re-executed after input-boundary repair; not new independent samples')
write_json_new(run/'post-fix-invariance.result.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
