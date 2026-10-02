"""Collect completed release records without modifying any sealed run/archive."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from facemath.observation.journal import code_fingerprint

root = Path(sys.argv[1]).resolve()
run = root / 'runs/native-pipeline-20261002'
release = run / 'release-check'
evidence = root / 'evidence/native-pipeline-20261002'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def copy_new(source, target):
    with source.open('rb') as source_file, target.open('xb') as target_file:
        shutil.copyfileobj(source_file, target_file)


def suite(path):
    xml = ET.parse(path).getroot()
    suites = list(xml.iter('testsuite'))
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'errors', 'failures', 'skipped')}
    counts['passed'] = counts['tests'] - counts['errors'] - counts['failures'] - counts['skipped']
    counts['skips'] = [dict(test=c.attrib['name'], reason=c.find('skipped').attrib.get('message',''))
                      for c in xml.iter('testcase') if c.find('skipped') is not None]
    return counts


validation = read(release / 'validation.json')
assert validation['completed'] and validation['native_pipeline_executed']
assert all(c['returncode'] == 0 for c in validation['commands'])
post = evidence / 'postseal'; post.mkdir(exist_ok=False)
for path in release.iterdir():
    if path.is_file() and path.suffix in ('.json', '.log', '.xml'):
        copy_new(path, post / path.name)
shutil.copytree(release / 'archive-native', post / 'archive-native')
shutil.copytree(release / 'wheel-check/native-pipeline', post / 'installed-native')
commands = post / 'installed-commands'; commands.mkdir()
for path in (release / 'wheel-check').iterdir():
    if path.is_file() and path.suffix in ('.json', '.xml'):
        copy_new(path, commands / path.name)

pipelines = {
    'pilot': evidence / 'pilot', 'source': release / 'source-native',
    'installed_wheel': release / 'wheel-check/native-pipeline', 'extracted_archive': release / 'archive-native',
}
reference = pipelines['pilot']
comparison = {}
for name, path in pipelines.items():
    summary = read(path / 'pipeline.json')
    assert summary['completed'] and summary['replay_verified'] and not summary['human_accuracy_validated']
    assert summary['environment']['engine_code_sha256'] == code_fingerprint()
    records = sorted((path / 'session/cases').glob('*/observation.json'))
    equal, maximum = 0, 0.0
    for observation in records:
        previous = read(reference / 'session/cases' / observation.parent.name / 'observation.json')
        current = read(observation)
        equal += previous == current
        maximum = max(maximum, float(np.max(np.abs(np.asarray(previous['xy']) - np.asarray(current['xy'])))))
    comparison[name] = dict(observations=len(records), exact_observation_objects_equal_to_pilot=equal,
        max_xy_difference_px=maximum, outcomes=summary['outcomes'], fingerprint=summary['environment']['engine_code_sha256'])

controls = {}
for name, path in [('source',release/'source-lab'),('installed_wheel',release/'wheel-check/milestone'),
                   ('extracted_archive',release/'archive-lab')]:
    lab = read(path/'laboratory.json')
    assert lab['admitted_pair']['raw_arrays_unchanged']
    assert lab['positive_control']['admission']['passed'] and not lab['negative_control']['admission']['passed']
    assert lab['withheld_query']['observer_calls'] == 0
    controls[name] = dict(raw_arrays_checked=lab['admitted_pair']['raw_arrays_checked'],
                          raw_arrays_unchanged=True, positive_admitted=True, negative_withheld=True)

target = root / 'packages/facemath-0.11.0-native-20261002.zip'
copy_new(release/'facemath-0.11.0.zip', target)
copy_new(release/'facemath-0.11.0.release.json', target.with_suffix('.release.json'))
copy_new(release/'validation.json', target.with_suffix('.validation.json'))

result = {
    'schema':'facemath.native_pipeline_validation.v11', 'version':'0.11.0',
    'created_utc':datetime.now(timezone.utc).isoformat(), 'engine_code_sha256':code_fingerprint(),
    'parent':read(evidence/'development/parent-archive.json'),
    'suites':{'parent_0_10_1':suite(run/'parent.xml'),'source_python311':suite(release/'source.xml'),
              'source_python314':suite(run/'source314.xml'),'installed_wheel':suite(release/'wheel-check/pytest.xml'),
              'extracted_archive':suite(release/'archive.xml')},
    'native_repetitions':comparison, 'milestone_controls':controls,
    'pilot':read(evidence/'pilot/pipeline.json'),
    'archive':dict(path=target.relative_to(root).as_posix(),bytes=target.stat().st_size,
                   sha256=sha256(target.read_bytes()).hexdigest(),validation=validation['archive']),
    'review_progress':{
        'I16':{'status':'implemented_and_tested','files':['src/facemath/observation/journal.py',
            'src/facemath/observation/study_io.py','src/facemath/observation/validation_io.py',
            'tests/test_failed_study_evidence.py'],
            'result':'Partial validation/finalization bytes and actual journals survive injected faults; repeated failures create distinct runs.'},
        'P01':{'status':'native_execution_completed_on_tested_host','files':['examples/native-dlib-pilot.json',
            'evidence/native-pipeline-20261002/pilot/session','models/README.md'],
            'result':'Pinned real model executed on Windows/Python 3.11; all 43 enrolled crops observed.',
            'next_action':'Assess the separate hosted native jobs before claiming other-platform compatibility.'},
        'D01':{'status':'automated_machine_reference_pipeline_completed','files':['src/facemath/observation/dataset.py',
            'tests/test_native_dataset.py','docs/NATIVE_PIPELINE_OPERATIONS.md'],
            'result':'Sourced data, native observations, trained joint error model and exact replay are retained and transferable.',
            'next_action':'Enroll a larger diverse source-disjoint dataset before fitting; retain label origin and evaluate error-bound utility without retuning evaluation.'},
        'D03':{'status':'partial','files':['uv.lock','examples/native-dlib-pilot.json','.github/workflows/milestone.yml'],
            'result':'Locked MediaPipe 1.0.1 executed on Python 3.11; Python 3.14 source regressions run separately.',
            'next_action':'Retain completed native compatibility results for each intended host/runtime.'},
        'D07':{'status':'partial','files':['src/facemath/observation/study_runner.py'],
            'result':'Windows workers launch hidden and real success paths were executed.',
            'next_action':'Qualify timeout/interruption descendant cleanup with actual native processes.'}},
    'limits':['Teacher-model agreement is not independent anatomical accuracy.',
        'The 90% radius is unbounded: rank five exceeds four calibration groups. No evaluation retuning occurred.',
        'Nine source-photo groups are not externally verified person-independent samples.',
        'Bitwise agreement of these repeated runs does not prove universal native determinism.',
        'Installed-wheel tests use the same dependency environment; hosted CI results are recorded separately.',
        'Post-seal evidence is adjacent to the immutable archive, not retrospectively inserted.'],
    'records':{'preseal':'evidence/native-pipeline-20261002/preseal',
               'postseal':'evidence/native-pipeline-20261002/postseal',
               'development':'evidence/native-pipeline-20261002/development'},
}
with (evidence/'VALIDATION.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps({'archive':result['archive'],'suites':result['suites'],
                  'native_repetitions':comparison,'milestone_controls':controls},indent=2))
