from pathlib import Path
from hashlib import file_digest
import json, shutil

root=Path(__file__).resolve().parents[2]
evidence=root/'evidence/core-features-20261004'
evidence.mkdir(exist_ok=False)
logs=evidence/'development-logs'; logs.mkdir()
for path in Path(__file__).parent.iterdir():
    if path.is_file() and path.suffix in ('.log','.json','.py'):
        shutil.copyfile(path,logs/path.name)
inputs=evidence/'frozen-inputs'; inputs.mkdir()
source=root/'evidence/structural-comparison-20261004'
shutil.copyfile(source/'frozen-inputs/model.json',inputs/'model.json')
shutil.copyfile(source/'next-registry.json',inputs/'registry.json')
plan=json.loads((root/'examples/helen-structure-cohort.json').read_text())
plan['name']='helen-core-v19-frozen-evaluation-20'
plan['selection_offset']=212
plan['grouping_limit']=('One face per HELEN filename photograph root; mirrors excluded and exact aliases rejected. '
    'Filename roots do not prove person independence or native-training exclusion. Window [212,232) excludes '
    'previously enrolled HELEN sources and is checked against the retained 335-case registry. '
    'The v0.17 model remains frozen; no fitting, calibration, replacement or tuning on these outcomes.')
with (root/'examples/helen-core-cohort.json').open('x',encoding='utf-8') as stream:
    json.dump(plan,stream,indent=2); stream.write('\n')
hashes={}
for path in inputs.iterdir():
    with path.open('rb') as stream:hashes[path.name]=file_digest(stream,'sha256').hexdigest()
with (evidence/'frozen-inputs.json').open('x') as stream:json.dump(hashes,stream,indent=2)
