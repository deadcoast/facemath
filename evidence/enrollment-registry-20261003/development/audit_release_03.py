"""Read-only final source/package/model/observation checks; no evidence repairs."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile
import numpy as np
from facemath.observation.contracts import Observation, array_digest
from facemath.observation.journal import code_fingerprint
from facemath.observation.source_cohort import make_recipe
from facemath.observation.transfer_io import verify_transfer_pipeline

run = Path(__file__).resolve().parent
root = run.parents[1]
release = run/'release-03'
paths = dict(source=release/'source-native', installed=release/'wheel-check/native-pipeline',
             archive=release/'archive-native', linux=run/'linux-native-03')
capsule = json.loads((root/'evidence/enrollment-registry-20261003/helen-model.json').read_text())
model_bytes = capsule['model_json'].encode()
assert (root/'evidence/enrollment-registry-20261003/helen-model.json').read_bytes() == (root/'evidence/model-transfer-20261002/helen-model.json').read_bytes()
frozen = json.loads((root/'evidence/enrollment-registry-20261003/prospective-freeze-03.json').read_text())
assert code_fingerprint() == frozen['engine_code_sha256']
result = dict(engine_code_sha256=code_fingerprint(), pipelines={}, comparisons={},
              model_bytes_sha256=sha256(model_bytes).hexdigest(), legacy_recipes={})
for cohort in ('source-cohort','bound-utility','model-transfer'):
    path = root/f'evidence/{cohort}-20261002/dataset/recipe.json'
    recipe = json.loads(path.read_text())
    assert make_recipe(recipe['cohort']['plan'], recipe['cohort']['inventory']) == recipe
    result['legacy_recipes'][str(path.relative_to(root))] = {'unchanged_recomputed_recipe': True,
                                                           'sha256': sha256(path.read_bytes()).hexdigest()}
observations, reports, outcomes = {}, {}, {}
for label, path in paths.items():
    verified = verify_transfer_pipeline(path)
    observations[label], outcomes[label] = {}, {}
    checked = []
    for case in sorted((path/'session/cases').iterdir()):
        receipt = json.loads((case/'receipt.json').read_text())
        outcomes[label][case.name] = receipt['status']
        if receipt['status'] != 'observed':
            checked.append(dict(case=case.name, status=receipt['status'], raw_arrays=0))
            continue
        copies = [case/'observation.json', path/'evaluation/acquisition/cases'/case.name/'observation.json',
                  path/'replay/acquisition/cases'/case.name/'observation.json']
        raw = [p.read_bytes() for p in copies]
        assert raw[0] == raw[1] == raw[2]
        obs = Observation.from_dict(json.loads(raw[0]))
        observations[label][case.name] = (obs, raw[0])
        checked.append(dict(case=case.name, status='observed', raw_arrays=2,
                            xy_sha256=array_digest(obs.xy), samples_sha256=array_digest(obs.samples)))
    for stage in ('evaluation','replay'):
        assert (path/stage/'localization_calibration.json').read_bytes() == model_bytes
    reports[label] = json.loads((path/'evaluation/transfer.json').read_text())
    assert reports[label]['study_sha256'] == frozen['study_sha256']
    assert reports[label]['enrolled_cases'] == frozen['check']['enrolled']
    assert all(reports[label][k] is False for k in ('fit_performed','raw_coordinates_changed','empirical_admission'))
    result['pipelines'][label] = dict(verification=verified, cases=checked,
        raw_arrays_unchanged=sum(c['raw_arrays'] for c in checked), model_bytes_unchanged=True,
        outcomes=outcomes[label])
for label in ('installed','archive','linux'):
    cases = []
    for name in sorted(set(observations['source']) | set(observations[label])):
        if name not in observations['source'] or name not in observations[label]:
            cases.append(dict(case=name, same_outcome=False))
            continue
        left, raw = observations['source'][name]
        right, other = observations[label][name]
        arrays = {field: dict(bytes_equal=array_digest(getattr(left,field)) == array_digest(getattr(right,field)),
            maximum_absolute_delta_px=float(np.max(np.abs(getattr(left,field)-getattr(right,field)))))
                  for field in ('xy','samples')}
        cases.append(dict(case=name, same_outcome=True, observation_bytes_equal=raw == other, arrays=arrays))
    with np.load(paths['source']/'evaluation/raw_validation/arrays.npz', allow_pickle=False) as a, np.load(paths[label]/'evaluation/raw_validation/arrays.npz', allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        arrays = {k: dict(bytes_equal=array_digest(a[k]) == array_digest(b[k]),
            maximum_absolute_delta=float(np.max(np.abs(a[k].astype(float)-b[k].astype(float))))) for k in a.files}
    same_report = reports['source'] == reports[label]
    if label != 'linux':
        assert same_report and outcomes['source'] == outcomes[label]
        assert all(c['observation_bytes_equal'] for c in cases)
        assert all(v['bytes_equal'] for v in arrays.values())
    result['comparisons'][label] = dict(cases=cases, raw_arrays=arrays, report_exact=same_report,
        all_outcomes_equal=outcomes['source'] == outcomes[label],
        limit='observed repeatability of the same enrolled portraits, not new independent enrollment or universal determinism')
archive = release/'facemath-0.16.0.zip'
checked = []
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        relative = name.split('/',1)[1]
        if relative.startswith(('src/facemath/','tests/','docs/','scripts/','examples/','.github/')) or relative in (
            'README.md','AGENTS.md','CHANGELOG.md','REPOSITORY_REVIEW.json','pyproject.toml','uv.lock','.gitattributes','.gitignore'):
            assert z.read(name) == (root/relative).read_bytes(), relative
            checked.append(relative)
result.update(archive_sha256=sha256(archive.read_bytes()).hexdigest(), source_files_exact=checked,
              archive_bytes=archive.stat().st_size)
with (run/'release-audit-03.result.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(dict(archive_sha256=result['archive_sha256'], source_files=len(checked),
    raw_arrays_unchanged={k:v['raw_arrays_unchanged'] for k,v in result['pipelines'].items()},
    reports_equal={k:v['report_exact'] for k,v in result['comparisons'].items()}),indent=2))

