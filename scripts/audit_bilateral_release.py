"""Retain executed bilateral comparisons and summarize the separate release run."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from facemath.observation.journal import code_fingerprint


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = root/'runs/bilateral-20261004'
    feature = root/'evidence/bilateral-20261004'
    if args.collect:
        feature = feature/'final'
        feature.mkdir()
        browser = read(scratch/'browser-02/browser.json')
        assert browser['passed'] and browser['bilateral_numeric_components_compared'] == 30
        shutil.copytree(scratch/'browser-02', feature/'browser', ignore=shutil.ignore_patterns('comparison.zip'))
        shutil.copytree(scratch/'pair-02', feature/'pair')
        for key in ('run_id', 'no_face_run', 'cancelled_run'):
            shutil.copytree(scratch/'app'/browser[key], feature/'app'/browser[key],
                            ignore=shutil.ignore_patterns('runtime-cache', 'comparison.zip'))
        native = feature/'app'/browser['run_id']/'result'
        assert read(native/'environment.json')['engine_code_sha256'] == code_fingerprint()
        for name in ('targeted-03.log', 'ui-audit.json', 'browser-02.log', 'pair-02.log'):
            shutil.copyfile(scratch/name, feature/name)
        parent = feature/'parent'; parent.mkdir()
        for name in ('archive-tests.log', 'archive.xml', 'commands.json'):
            shutil.copyfile(root/'runs/workspace-20261004/release-01'/name, parent/name)
        parent_archive = root/'packages/facemath-0.22.0-local-application-20261004.zip'
        assert digest(parent_archive) == '7c965167f1c9658d86c86818adc73a8e4837faa0a0d381c47d386387a5fea6fe'
        # Compare the inherited observed arrays against the same retained native
        # observation pair. This is distinct from new-browser inference.
        keys = read(scratch/'pair-02/comparison.json')['raw_array_sha256']
        with np.load(root/'evidence/edge-comparison-20261004/pair-02/nominal/arrays.npz') as before, np.load(feature/'pair/nominal/arrays.npz') as after:
            common = sorted(set(before.files) & set(keys))
            for key in common:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        data = {'version': '0.23.0', 'code_sha256': code_fingerprint(), 'browser': browser,
                'native_pair_audit': read(feature/'pair/bilateral-audit.json'),
                'parent_archive_sha256': digest(parent_archive), 'inherited_observed_arrays_unchanged': len(common),
                'parent_baseline': 'Reused completed actual 0.22 extracted-archive run; not a new execution',
                'development_failure': {'source': 'tool transcript', 'command': 'pytest targeted --basetemp=runs/bilateral-20261004/tests-01',
                    'result': '21 passed, 3 setup errors: basetemp parent directory absent; created parent and retained successful fresh tests-02 log'},
                'scope': 'Observed bilateral configuration; no new cohort, fitting or independent anatomical reference validation'}
        write(feature/'workflow.json', data)
        print(json.dumps({k: data[k] for k in ('code_sha256', 'inherited_observed_arrays_unchanged', 'scope')}, indent=2))
        return
    release = args.release.resolve()
    validation = read(release/'validation.json')
    assert validation['completed'] and all(c['returncode'] == 0 for c in validation['commands'])
    counts, milestones = {}, {}
    for tier, location, lab in (('source', 'source.xml', 'source-lab'),
                                ('wheel', 'wheel-check/pytest.xml', 'wheel-check/milestone'),
                                ('archive', 'archive.xml', 'archive-lab')):
        suite = ET.parse(release/location).getroot().find('testsuite')
        result = {k: int(suite.attrib[k]) for k in ('tests', 'failures', 'errors', 'skipped')}
        assert not result['failures'] and not result['errors']
        result['passed'] = result['tests'] - result['skipped']
        result['skip_reasons'] = sorted({s.attrib.get('message', '') for s in suite.iter('skipped')})
        live = [c for c in suite.iter('testcase') if c.attrib.get('name') == 'test_live_mediapipe_provider']
        assert live and all(c.find('skipped') is None for c in live)
        result['live_provider_passed'] = True
        counts[tier] = result
        control = read(release/lab/'laboratory.json')['admitted_pair']
        assert control['raw_arrays_unchanged'] and control['qualification_snapshot_replayed']
        milestones[tier] = control
    post = root/'evidence/bilateral-release-20261004'; post.mkdir()
    for path in release.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log', '.xml'):
            shutil.copyfile(path, post/path.name)
    shutil.copytree(release/'wheel-check', post/'wheel-check',
                    ignore=shutil.ignore_patterns('replayed', 'pair-replayed', 'milestone'))
    archive = root/'packages/facemath-0.23.0-bilateral-comparison-20261004.zip'
    if archive.exists():
        raise FileExistsError(archive)
    shutil.copyfile(release/'facemath-0.23.0.zip', archive)
    workflow = read(feature/'final/workflow.json')
    assert workflow['code_sha256'] == code_fingerprint()
    write(root/'BILATERAL_COMPARISON.json', {'version': '0.23.0', 'code_sha256': code_fingerprint(),
        'tests': counts, 'milestones': milestones, 'workflow': workflow,
        'installed_bytes': read(release/'wheel-check/installed-byte-check.json'),
        'archive': {'path': archive.relative_to(root).as_posix(), 'sha256': digest(archive), 'files': validation['archive']['files']},
        'evidence': post.relative_to(root).as_posix()})
    print(json.dumps({'tests': counts, 'archive_sha256': digest(archive)}, indent=2))


if __name__ == '__main__':
    main()
