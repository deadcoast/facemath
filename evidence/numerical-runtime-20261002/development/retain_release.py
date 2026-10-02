"""Retain actual completed gates; never modify an earlier execution."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
release = run/'release-check'
evidence = root/'evidence/numerical-runtime-20261002'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def retain_files(source, target, suffixes=('.json', '.log', '.xml', '.py')):
    target.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.iterdir()):
        if source == run and path.name.startswith('retain-release.'):
            continue
        if path.is_file() and path.suffix in suffixes:
            shutil.copyfile(path, target/path.name)


def suite(path):
    data = ET.parse(path).getroot()
    suites = [data] if data.tag == 'testsuite' else list(data)
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
    counts['skip_reasons'] = sorted({s.attrib.get('message', '') for s in data.iter('skipped')})
    assert counts['errors'] == counts['failures'] == 0, counts
    return counts


assert read(release/'validation.json')['completed']
assert read(run/'linux-validation.json')['returncode'] == 0
assert read(run/'python314-tests.json')['returncode'] == 0
for name in ('windows-rejects-linux.json', 'linux-rejects-windows.json'):
    assert read(run/name)['comparison']['compatible'] is False
for name in ('cross-runtime-windows.json', 'cross-runtime-linux.json'):
    assert read(run/name)['returncode'] == 2
transfer = read(run/'transferred-replay-control.result.json')
assert transfer['returncode'] == 2 and transfer['verification']['verified']
assert not transfer['destination_created'] and transfer['source_manifest_unchanged']

retain_files(run, evidence/'development')
shutil.copytree(run/'linux-native', evidence/'development/linux-native')
post = evidence/'postseal'
retain_files(release, post)
retain_files(release/'wheel-check', post/'wheel-check', ('.json', '.xml'))
shutil.copytree(release/'wheel-check/native-pipeline', post/'installed-native')
shutil.copytree(release/'archive-native', post/'archive-native')
labs = {}
for name, path in [('source', release/'source-lab'), ('installed', release/'wheel-check/milestone'),
                   ('archive', release/'archive-lab')]:
    data = read(path/'laboratory.json')
    assert data['admitted_pair']['raw_arrays_unchanged'] and data['withheld_query']['observer_calls'] == 0
    labs[name] = {'positive': data['positive_control']['admission']['status'],
        'negative': data['negative_control']['admission']['status'],
        'raw_arrays_checked': data['admitted_pair']['raw_arrays_checked'],
        'raw_arrays_unchanged': True, 'withheld_observer_calls': 0}
    shutil.copyfile(path/'laboratory.json', post/(name+'-laboratory.json'))

archive_source = release/'facemath-0.11.2.zip'
archive = root/'packages/facemath-0.11.2-numerics-20261002.zip'
for source, target in [(archive_source, archive),
    (archive_source.with_suffix('.release.json'), archive.with_suffix('.release.json')),
    (release/'validation.json', archive.with_suffix('.validation.json'))]:
    assert not target.exists()
    shutil.copyfile(source, target)

result = {'schema': 'facemath.numeric_runtime_validation.v1', 'version': '0.11.2',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'engine_code_sha256': read(release/'environment.log')['engine_code_sha256'],
    'review_sha256': sha256((root/'REPOSITORY_REVIEW.json').read_bytes()).hexdigest(),
    'parent': {'archive': 'packages/facemath-0.11.1-portability-20261002.zip',
        'sha256': '417bb4e81b9725c2ede1dad6d020c3af4658c85f82f768044ce606652a77c6d2',
        'suite': suite(run/'parent.xml'), 'commands': ['development/parent-archive.json',
            'development/parent-extracted.json', 'development/parent-tests.json']},
    'suites': {'windows_source_311': suite(release/'source.xml'),
        'installed_wheel': suite(release/'wheel-check/pytest.xml'),
        'extracted_archive': suite(release/'archive.xml'),
        'windows_314': suite(run/'python314.xml'), 'linux_311': suite(run/'linux.xml')},
    'milestone_controls': labs,
    'native_runs': {name: read(path/'pipeline.json') for name, path in [
        ('source', release/'source-native'), ('installed', release/'wheel-check/native-pipeline'),
        ('archive', release/'archive-native')]},
    'native_comparisons': {p.name: read(p) for p in run.glob('compare-*.result.json')},
    'numerical_diagnosis': {'trace': read(run/'trace-comparison.json'),
        'windows_swapped_cholesky': read(run/'windows-swap-output.json'),
        'original_inputs': 'evidence/native-portability-20261002/preseal/native-pipeline/qualification',
        'changes_to_mathematics_or_tolerances': False,
        'interpretation': 'identical residuals; differences arise in Cholesky and linear solve; changing norm alone does not fix them'},
    'runtime_controls': {'windows_to_linux': 'development/linux-rejects-windows.json',
        'linux_to_windows': 'development/windows-rejects-linux.json',
        'expected_exit_status': 2, 'incompatible_replay_authorized': False,
        'actual_transferred_qualification_replay': transfer,
        'new_linux_native_smoke': 'development/linux-native',
        'linux_native_scope': 'isolated pinned inference smoke; the prior complete Linux pilot remains historical evidence'},
    'preserved_failures': [{'record': 'development/focused-tests.json',
        'reason': 'first focused run used old installed package metadata 0.11.1 with editable 0.11.2 source',
        'result': '55 passed, 1 skipped, 2 version assertions failed',
        'resolution': 'fresh isolated locked 0.11.2 environment; focused and full suites rerun'},
        {'record': 'evidence/native-portability-20261002/VALIDATION.json',
         'reason': 'prior cross-host exact numerical replay failures and macOS native abort remain retained'}],
    'package': read(release/'validation.json')['archive'],
    'installed_bytes': 'postseal/wheel-check/installed-byte-check.json',
    'archive_source_parity': 'development/archive-source-parity.result.json',
    'completion_boundary': {'implemented': 'versioned numerical identity and preflight with field differences',
        'exact_replay': 'same compatible runtime only, still checked against stored outputs',
        'transfer': 'byte verification remains independent of this host runtime',
        'independent_anatomical_accuracy': False, 'cross_os_exact_replay': False,
        'human_testing_dependency': False},
    'next_actions': [
        {'priority': 'P1', 'task': 'larger prospectively frozen source-disjoint cohort',
         'references': ['docs/NUMERICAL_REPLAY_DESIGN.md', 'src/facemath/observation/dataset.py'],
         'acceptance': 'pinned original annotations/images, explicit origin/visibility/grouping, enough calibration groups frozen before inference, retained outcomes without retuning'},
        {'priority': 'P1', 'task': 'hosted macOS native qualification after jobs can execute',
         'references': ['.github/workflows/milestone.yml', 'scripts/check_native_runtime.py'],
         'acceptance': 'actual isolated native output and complete pipeline; scheduler refusal is not execution evidence'},
        {'priority': 'P2', 'task': 'portable numerical implementation only under an independently specified arithmetic contract',
         'references': ['src/facemath/observation/joint.py', 'docs/NUMERICAL_REPLAY_DESIGN.md'],
         'acceptance': 'new contract version, independently justified error/conditioning tests, cross-host trials; no tolerance fitted to old evaluation differences'}],
}
with (evidence/'VALIDATION.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({'archive': result['package'], 'suites': result['suites'], 'milestone': labs}, indent=2))
