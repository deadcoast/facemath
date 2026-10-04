"""Execute current and extracted-parent readers on the new bound study."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from facemath.observation.study import load_study, read_json, write_json_new
from facemath.observation.source_cohort import make_recipe
from facemath.observation.transfer import load_capsule

run = Path(__file__).resolve().parent
root = run.parents[1]
study = load_study(run/'prospective-lock-03')
assert study.data['format'] == 'facemath.prospective_study.v16'
parent = run/'parent/facemath-0.15.0'
env = dict(os.environ, PYTHONPATH=str(parent/'src'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
commands = [('old-study-reader-03', [sys.executable, '-m', 'facemath', 'verify', str(study.root)])]
# The native transfer intentionally keeps the legacy fitted model. Exercise the
# new model-export format using the explicitly synthetic integration-test bundle.
capsules = list((run/'release-03/source-temp').glob('test_registry_survives_final*/model.json'))
assert len(capsules) == 1
load_capsule(read_json(capsules[0]))
script = ('import sys; from facemath.observation.study import read_json; '
          'from facemath.observation.transfer import load_capsule; '
          'from facemath.observation.contracts import ObservationError\n'
          'try: load_capsule(read_json(sys.argv[1]))\n'
          'except ObservationError as e: print(str(e)); sys.exit(2)\n')
commands.append(('old-capsule-reader-03', [sys.executable, '-c', script, str(capsules[0])]))
records = []
for name, command in commands:
    start = datetime.now(timezone.utc).isoformat()
    with (run/(name+'.log')).open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=parent, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(command=command, cwd=str(parent), environment={'PYTHONPATH':env['PYTHONPATH']},
        started_utc=start, finished_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
        expected_returncode=2, synthetic_capsule=(name=='old-capsule-reader-03'))
    write_json_new(run/(name+'.json'), record); records.append(record)
    assert result.returncode == 2
recipes = []
for cohort, date in [('source-cohort','20261002'), ('bound-utility','20261002'),
                      ('model-transfer','20261002'), ('source-window','20261003')]:
    path = root/f'evidence/{cohort}-{date}/dataset/recipe.json'
    recipe = read_json(path)
    assert make_recipe(recipe['cohort']['plan'], recipe['cohort']['inventory']) == recipe
    recipes.append(path.relative_to(root).as_posix())
result = dict(current_reader_verified=True, study_sha256=study.digest, old_reader_refusals=records,
              legacy_recipes_recomputed_unchanged=recipes)
write_json_new(run/'compatibility-03.result.json', result)
print(json.dumps(result, indent=2))
