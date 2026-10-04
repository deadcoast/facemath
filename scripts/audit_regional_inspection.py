"""Retain actual regional-inspector execution and seal its release record."""
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
    scratch = root/'runs/regional-inspection-20261004'
    feature = root/'evidence/regional-inspection-20261004'
    if args.collect:
        browser = read(scratch/'browser-01/browser.json')
        assert browser['passed'] and browser['regional_keyboard_disclosures'] == 7
        shutil.copytree(scratch/'browser-01', feature/'browser', ignore=shutil.ignore_patterns('comparison.zip'))
        shutil.copytree(scratch/'pair-01', feature/'pair')
        for key in ('run_id', 'no_face_run', 'cancelled_run'):
            shutil.copytree(scratch/'app'/browser[key], feature/'app'/browser[key],
                            ignore=shutil.ignore_patterns('runtime-cache', 'comparison.zip'))
        native = feature/'app'/browser['run_id']/'result'
        assert read(native/'environment.json')['engine_code_sha256'] == code_fingerprint()
        for name in ('targeted-01.log', 'targeted-02.log', 'targeted-03.log', 'ui-audit.json', 'browser-01.log', 'pair-01.log'):
            shutil.copyfile(scratch/name, feature/name)
        keys = read(scratch/'pair-01/comparison.json')['raw_array_sha256']
        with np.load(root/'evidence/bilateral-20261004/final/pair/nominal/arrays.npz') as before, np.load(feature/'pair/nominal/arrays.npz') as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            count = len(before.files)
        write(feature/'workflow.json', {'version': '0.24.0', 'code_sha256': code_fingerprint(), 'browser': browser,
            'native_pair_audit': read(feature/'pair/regional-audit.json'),
            'parent_baseline': read(feature/'parent/baseline.json'),
            'inherited_arrays_unchanged': count, 'observed_arrays_calibration_unchanged': len(keys),
            'development_failure': 'targeted-01: 69 passed, 1 failed: scalar versus batched Frobenius norm differed by one ULP; analytic comparison uses 1e-14 tolerance. No production calculation changed for this failure.',
            'scope': 'Inspect existing observed geometry; no new cohort, fit or independent anatomical accuracy validation'})
        print(f'Collected native browser and pair evidence; {count} inherited arrays unchanged.', flush=True)
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
    post = root/'evidence/regional-inspection-release-20261004'; post.mkdir()
    for path in release.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log', '.xml'):
            shutil.copyfile(path, post/path.name)
    shutil.copytree(release/'wheel-check', post/'wheel-check',
                    ignore=shutil.ignore_patterns('replayed', 'pair-replayed', 'milestone'))
    archive = root/'packages/facemath-0.24.0-regional-inspection-20261004.zip'
    if archive.exists():
        raise FileExistsError(archive)
    shutil.copyfile(release/'facemath-0.24.0.zip', archive)
    workflow = read(feature/'workflow.json')
    assert workflow['code_sha256'] == code_fingerprint()
    write(root/'REGIONAL_INSPECTION.json', {'version': '0.24.0', 'code_sha256': code_fingerprint(),
        'tests': counts, 'milestones': milestones, 'workflow': workflow,
        'installed_bytes': read(release/'wheel-check/installed-byte-check.json'),
        'archive': {'path': archive.relative_to(root).as_posix(), 'sha256': digest(archive), 'files': validation['archive']['files']},
        'evidence': post.relative_to(root).as_posix()})
    print(json.dumps({'tests': counts, 'archive_sha256': digest(archive)}, indent=2))


if __name__ == '__main__':
    main()
