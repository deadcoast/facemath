"""Retain completed 0.18 execution evidence and compare all repeated native arrays."""
from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from facemath.observation.contracts import Observation, array_digest
from facemath.observation.journal import code_fingerprint
from facemath.observation.study import read_json, write_json_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    release, evidence = args.release.resolve(), args.evidence.resolve()
    completion = read_json(release/'validation.json')
    assert completion['completed'] and all(c['returncode'] == 0 for c in completion['commands'])
    project = Path(__file__).resolve().parents[1]
    pair = read_json(project/'evidence/structural-comparison-20261004/pair-01/comparison.json')
    assert pair['raw_arrays_unchanged'] and pair['raw_arrays_checked'] == 146
    evidence.mkdir(parents=True, exist_ok=False)
    tests = {}
    for tier, path in [('source', release/'source.xml'), ('wheel', release/'wheel-check/pytest.xml'), ('archive', release/'archive.xml')]:
        suites = ET.parse(path).getroot()
        suite = suites if suites.tag == 'testsuite' else suites.find('testsuite')
        counts = {name: int(suite.attrib[name]) for name in ('tests', 'errors', 'failures', 'skipped')}
        assert counts['errors'] == counts['failures'] == 0
        counts['passed'] = counts['tests']-counts['skipped']
        tests[tier] = counts
    pipelines = {'source': release/'source-native', 'wheel': release/'wheel-check/native-pipeline',
                 'archive': release/'archive-native'}
    base = read_json(pipelines['source']/'study/study.lock.json')
    baseline = {}
    for i, case in enumerate(base['cases']):
        path = Path('session/cases')/f'case_{i:05d}'/'observation.json'
        observation = Observation.from_dict(read_json(pipelines['source']/path))
        baseline[path] = observation
    native = {}
    for tier, root in pipelines.items():
        report = read_json(root/'pipeline.json')
        transfer = read_json(root/'evaluation/transfer.json')
        assert report['completed'] and report['fit_performed'] is False and report['empirical_admission'] is False
        assert report['outcomes']['observed'] == len(base['cases']) == 20
        assert report['environment']['engine_code_sha256'] == code_fingerprint()
        array_hashes = {}
        for path, reference in baseline.items():
            observation = Observation.from_dict(read_json(root/path))
            assert observation.image_sha256 == reference.image_sha256
            for name in ('xy', 'samples', 'covariance', 'in_bounds'):
                actual, expected = getattr(observation, name), getattr(reference, name)
                np.testing.assert_array_equal(actual, expected, err_msg=f'{tier}/{path}/{name}')
                array_hashes[f'{path.parent.name}/{name}'] = array_digest(actual)
        native[tier] = {'observed': 20, 'enrolled': len(base['cases']), 'fit_performed': False,
                        'covered_groups': transfer['coverage']['covered_groups'],
                        'calibration_sha256': transfer['calibration_sha256'],
                        'radius': transfer['radius'], 'raw_array_sha256': array_hashes,
                        'raw_array_comparisons': len(array_hashes)}
    assert len({v['calibration_sha256'] for v in native.values()}) == 1
    assert len({v['radius'] for v in native.values()}) == 1
    for path in release.iterdir():
        if path.is_file() and path.suffix in ('.log', '.json', '.xml'):
            shutil.copyfile(path, evidence/path.name)
    for name in ('archive-lab', 'archive-native', 'archive-qualification-replay', 'archive-pair-replay'):
        shutil.copytree(release/name, evidence/name)
    wheel = evidence/'wheel-check'; wheel.mkdir()
    for path in (release/'wheel-check').iterdir():
        if path.is_file():
            shutil.copyfile(path, wheel/path.name)
    write_json_new(evidence/'native-array-parity.json', native)
    archive = release/'facemath-0.18.0.zip'
    with archive.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    result = {'format': 'facemath.structural_release.v18', 'version': '0.18.0',
              'code_sha256': code_fingerprint(), 'tests': tests,
              'parent': {'version': '0.17.0', 'archive_sha256': 'ea5fec7d4f162696d2c61390801147f7d38f86adbb37a5ead7f9ccd676b1d2a2',
                         'synchronized_commit': '26c1fdc24827a8fbeb308cbb82b74304273c1460', 'baseline': '1177 passed, 1 Windows symlink skip'},
              'archive': {'path': 'packages/facemath-0.18.0-structure-20261004.zip', 'sha256': digest, 'bytes': archive.stat().st_size,
                          'files': completion['archive']['files']},
              'implemented': ['Fourteen named structural measurements in the production pair result, CLI, terminal overview and existing HTML report',
                              'Reference bounds from the same source graph, with per-source radii, uncertain normalization and frozen missingness denominators',
                              'All seven published artifact families inherit parent directory access',
                              'Registry-bearing frozen model transfer receives decoded image dimensions before overlap checks'],
              'verification': {'commands': str(evidence/'commands.json'),
                               'installed_wheel_bytes': str(evidence/'wheel-check/installed-byte-check.json'),
                               'native_array_parity': str(evidence/'native-array-parity.json'),
                               'native_pair': 'evidence/structural-comparison-20261004/pair-01/comparison.json',
                               'pair_raw_arrays_unchanged': pair['raw_arrays_checked'],
                               'browser': 'evidence/structural-comparison-20261004/browser-03/browser.json',
                               'ordinary_user_access': 'evidence/structural-comparison-20261004/development-logs/ordinary-user-read.json',
                               'ordinary_user_full_verification': 'evidence/structural-comparison-20261004/ordinary-user-verify.json',
                               'static_ui_audit_review': 'evidence/structural-comparison-20261004/ui-audit-review.json'},
              'native_evaluation': {'sources': 20, 'helen_window': [192, 212], 'fit_performed': False,
                                    'covered_groups': native['source']['covered_groups'], 'model_unchanged': True,
                                    'domain_declaration': 'The recorded domain string is inherited from the donor and retains its cohort wording; matching this label does not prove transfer validity. The sealed target selection is [192,212).',
                                    'published_label_measurements': 'evidence/structural-comparison-20261004/published-label-agreement.json'},
              'next_registry': 'evidence/structural-comparison-20261004/next-registry.json', 'registered_cases': 335,
              'failures_retained': 'evidence/structural-comparison-20261004/failure-history.json',
              'open': ['All fourteen reference comparisons for the declared native pair remain unresolved under the broad frozen model',
                       'Independent anatomical and visibility validation remains unperformed; published PTS agreement is not empirical admission',
                       'Person independence and native model training overlap are unknown',
                       'Validation executed on this Windows runtime; installed wheel shares dependencies',
                       'Rendered static UI audit reports three actionless-button heuristic findings; actual registered handlers and all three downloads were exercised'],
              'next_work': 'Independent anatomical/visibility references and prospectively separate development/calibration if the observer or model changes; do not retune on this evaluation.'}
    write_json_new(args.out, result)
    print(json.dumps({'archive': result['archive'], 'tests': tests, 'native_arrays_per_tier': 80}, indent=2))


if __name__ == '__main__':
    main()
