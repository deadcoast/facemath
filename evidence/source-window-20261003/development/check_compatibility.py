"""Exercise the actual archived parent reader and current legacy recomputation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from facemath.observation.source_cohort import make_recipe

run = Path(__file__).resolve().parent
root = run.parents[1]
parent = run/'parent/facemath-0.14.0'
snippet = '''
import json, sys
from pathlib import Path
import facemath
from facemath.observation.source_cohort import validate_plan
from facemath.observation.dataset import _recipe
from facemath.observation.contracts import ObservationError
records = []
for kind, path in [('plan',sys.argv[1]),('recipe',sys.argv[2])]:
    try:
        validate_plan(json.loads(Path(path).read_text())) if kind == 'plan' else _recipe(Path(path))
    except ObservationError as exc:
        records.append(dict(kind=kind, refused=True, reason=str(exc)))
    else:
        raise AssertionError('old reader silently accepted a source window')
print(json.dumps(dict(import_path=facemath.__file__, version=facemath.__version__, checks=records)))
'''
command = [sys.executable, '-c', snippet, str(root/'examples/helen-transfer-window.json'),
           str(root/'evidence/source-window-20261003/dataset/recipe.json')]
env = dict(os.environ, PYTHONPATH=str(parent/'src'))
proc = subprocess.run(command, cwd=parent, env=env, text=True, capture_output=True)
record = dict(command=command, cwd=str(parent), returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
assert proc.returncode == 0, record
result = json.loads(proc.stdout)
assert Path(result['import_path']).is_relative_to(parent/'src') and result['version'] == '0.14.0'
record['legacy_recipes'] = []
for cohort in ('source-cohort','bound-utility','model-transfer'):
    path = root/f'evidence/{cohort}-20261002/dataset/recipe.json'
    recipe = json.loads(path.read_text())
    assert make_recipe(recipe['cohort']['plan'], recipe['cohort']['inventory']) == recipe
    record['legacy_recipes'].append(dict(path=str(path.relative_to(root)), exact_recomputed=True))
with (run/'compatibility.result.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(result, indent=2))
print('Current code exactly recomputed all three retained legacy cohort recipes.')
