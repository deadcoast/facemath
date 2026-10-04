from pathlib import Path
from hashlib import sha256
import json, shutil
from facemath.observation.contracts import Observation
from facemath.observation.reference import observer_recipe
root=Path.cwd(); out=root/'evidence/landmark-correspondence-20261004'
diag=json.loads((root/'LANDMARK_DIAGNOSIS.json').read_text())
rows=[]
base=root/'evidence/bound-utility-20261002/preseal/native-pipeline/session/cases'
for row in diag['cases']:
    if row['status']!='observed': continue
    data=(base/row['case_id']/'observation.json').read_bytes()
    assert sha256(data).hexdigest()==row['observation_file_sha256']
    record=json.loads(data); obs=Observation.from_dict(record)
    assert obs.to_dict()==record
    assert observer_recipe(obs)=='d805baa53707cce9795a34b3e0e88347021888c1b708ee62a073d9da4a735c37'
    rows.append(row['case_id'])
assert (root/'LANDMARK_DIAGNOSIS.json').read_bytes()==(out/'diagnosis.json').read_bytes()
result=dict(historical_observations_unchanged=len(rows),legacy_roundtrips_exact=True,legacy_recipe_unchanged=True,diagnosis_sha256=sha256((root/'LANDMARK_DIAGNOSIS.json').read_bytes()).hexdigest(),diagnosis_html_sha256=sha256((root/'LANDMARK_DIAGNOSIS.html').read_bytes()).hexdigest())
with (out/'historical-preservation.json').open('x') as f: json.dump(result,f,indent=2)
logs=out/'execution';logs.mkdir()
for path in (root/'runs/landmark-repair-20261004').iterdir():
    if path.is_file() and path.suffix in ('.json','.log','.py') and path.name!='release-01.log':
        shutil.copyfile(path,logs/path.name)
shutil.copyfile(root/'runs/landmark-repair-20261004/prospective-study/study.lock.json',out/'prospective-study.json')
print(json.dumps(result,indent=2))
