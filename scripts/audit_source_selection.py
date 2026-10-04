"""Retain actual source-selection execution and seal its local release record."""
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
    scratch = root/'runs/source-selection-20261004'
    feature = root/'evidence/source-selection-20261004'
    if args.collect:
        crop = read(scratch/'browser-02/browser.json')
        whole = read(scratch/'whole-image-01/browser.json')
        portable = read(scratch/'portable-01/portable.json')
        assert crop['passed'] and whole['passed'] and portable['passed']
        for directory in ('browser-01', 'browser-02', 'whole-image-01', 'fixtures', 'pair-01'):
            shutil.copytree(scratch/directory, feature/directory,
                            ignore=shutil.ignore_patterns('comparison.zip'))
        (feature/'portable-01').mkdir()
        for name in ('portable.json', 'selection.png'):
            shutil.copyfile(scratch/'portable-01'/name, feature/'portable-01'/name)
        ids = {crop['full_image_multiple_faces_run'], crop['cropped_native_run'],
               whole['run_id'], whole['no_face_run'], whole['cancelled_run'],
               'f67872652e0d46b9ba6bf6e751d8b6ec', 'bb8a4a21644948919ba2f3fad915a6af'}
        for run_id in sorted(ids):
            shutil.copytree(scratch/'app'/run_id, feature/'app'/run_id,
                            ignore=shutil.ignore_patterns('runtime-cache', 'comparison.zip'))
        for run_id in (crop['cropped_native_run'], whole['run_id']):
            assert read(feature/'app'/run_id/'result/environment.json')['engine_code_sha256'] == code_fingerprint()
        for name in ('targeted-01.log', 'ui-audit.json', 'browser-01.log', 'browser-02.log',
                     'whole-image-01.log', 'pair-01.log', 'portable-01.log', 'fixtures.log'):
            shutil.copyfile(scratch/name, feature/name)
        with np.load(root/'evidence/regional-inspection-20261004/pair/nominal/arrays.npz') as before, np.load(feature/'pair-01/nominal/arrays.npz') as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            count = len(before.files)
        write(feature/'workflow.json', {'version': '0.25.0', 'code_sha256': code_fingerprint(),
            'crop_browser': crop, 'whole_image_browser': whole, 'portable_download': portable,
            'native_pair_audit': read(feature/'pair-01/regional-audit.json'),
            'parent_baseline': read(feature/'parent/baseline.json'), 'inherited_arrays_unchanged': count,
            'development_failure': {'path': 'browser-01.log',
                'cause': 'Retry retained the previous failed status marker while preparing the upload; browser wait returned before the new run.',
                'repair': 'UI clears terminal state during preparation/loading; QA also waits for a distinct new run ID.',
                'prefinal_runs': ['f67872652e0d46b9ba6bf6e751d8b6ec', 'bb8a4a21644948919ba2f3fad915a6af'],
                'prefinal_result': 'First native cropped analysis completed, but the browser assertion failed before observing it. Distinct final-code runs are retained.'},
            'scope': 'Exact input preparation and native workflow on an explicitly duplicated portrait control; no new cohort, fit or independent anatomical validation'})
        print(f'Collected native cropping, whole-image and offline download; {count} inherited arrays unchanged.', flush=True)
        return
    if args.release is None:
        parser.error('choose --collect or --release')
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
    post = root/'evidence/source-selection-release-20261004'; post.mkdir()
    for path in release.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log', '.xml'):
            shutil.copyfile(path, post/path.name)
    shutil.copytree(release/'wheel-check', post/'wheel-check',
                    ignore=shutil.ignore_patterns('replayed', 'pair-replayed', 'milestone'))
    archive = root/'packages/facemath-0.25.0-source-selection-20261004.zip'
    if archive.exists():
        raise FileExistsError(archive)
    shutil.copyfile(release/'facemath-0.25.0.zip', archive)
    workflow = read(feature/'workflow.json')
    assert workflow['code_sha256'] == code_fingerprint()
    write(root/'SOURCE_SELECTION.json', {'version': '0.25.0', 'code_sha256': code_fingerprint(),
        'tests': counts, 'milestones': milestones, 'workflow': workflow,
        'installed_bytes': read(release/'wheel-check/installed-byte-check.json'),
        'archive': {'path': archive.relative_to(root).as_posix(), 'sha256': digest(archive), 'files': validation['archive']['files']},
        'evidence': post.relative_to(root).as_posix()})
    print(json.dumps({'tests': counts, 'archive_sha256': digest(archive)}, indent=2))


if __name__ == '__main__':
    main()
