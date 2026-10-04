"""Retain completed core-function release checks and exact native-array parity."""
from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import tomllib
import xml.etree.ElementTree as ET

import numpy as np

from facemath.observation.contracts import Observation, array_digest
from facemath.observation.journal import code_fingerprint
from facemath.observation.study import read_json, write_json_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    release, features, evidence = args.release.resolve(), args.features.resolve(), args.evidence.resolve()
    completion = read_json(release/'validation.json')
    assert completion['completed'] and all(c['returncode'] == 0 for c in completion['commands'])
    project = Path(__file__).resolve().parents[1]
    version = tomllib.loads((project/'pyproject.toml').read_text())['project']['version']
    pair = read_json(features/'pair-01/regional-audit.json')
    assert pair['raw_arrays_unchanged'] and pair['raw_arrays_checked'] == 158
    assert pair['same_components_before_after_calibration_and_replay']
    tests = {}
    for tier, path in [('source', release/'source.xml'), ('wheel', release/'wheel-check/pytest.xml'), ('archive', release/'archive.xml')]:
        suites = ET.parse(path).getroot()
        suite = suites if suites.tag == 'testsuite' else suites.find('testsuite')
        counts = {name: int(suite.attrib[name]) for name in ('tests', 'errors', 'failures', 'skipped')}
        assert counts['errors'] == counts['failures'] == 0
        counts['passed'] = counts['tests']-counts['skipped']
        tests[tier] = counts
    pipelines = {'source': release/'source-native', 'wheel': release/'wheel-check/native-pipeline', 'archive': release/'archive-native'}
    original = read_json(pipelines['source']/'study/study.lock.json')
    baseline = {}
    for i, _ in enumerate(original['cases']):
        path = Path('session/cases')/f'case_{i:05d}'/'observation.json'
        baseline[path] = Observation.from_dict(read_json(pipelines['source']/path))
    native = {}
    for tier, root in pipelines.items():
        report = read_json(root/'pipeline.json')
        transfer = read_json(root/'evaluation/transfer.json')
        assert report['completed'] and report['fit_performed'] is False and report['empirical_admission'] is False
        assert report['outcomes']['observed'] == len(original['cases']) == 20
        assert report['environment']['engine_code_sha256'] == code_fingerprint()
        hashes = {}
        for path, reference in baseline.items():
            actual = Observation.from_dict(read_json(root/path))
            assert actual.image_sha256 == reference.image_sha256
            for name in ('xy', 'samples', 'covariance', 'in_bounds'):
                np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name), err_msg=f'{tier}/{path}/{name}')
                hashes[f'{path.parent.name}/{name}'] = array_digest(getattr(actual, name))
        native[tier] = {'observed': 20, 'enrolled': 20, 'fit_performed': False,
            'covered_groups': transfer['coverage']['covered_groups'], 'radius': transfer['radius'],
            'calibration_sha256': transfer['calibration_sha256'], 'raw_array_sha256': hashes}
    assert len({x['calibration_sha256'] for x in native.values()}) == len({x['radius'] for x in native.values()}) == 1
    evidence.mkdir(parents=True, exist_ok=False)
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
    archive = release/f'facemath-{version}.zip'
    with archive.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    parent = read_json(features/'development-logs/parent.json')
    result = {'format': 'facemath.core_feature_release.v19', 'version': version,
        'code_sha256': code_fingerprint(), 'parent': parent, 'tests': tests,
        'archive': {'path': f'packages/facemath-{version}-core-20261004.zip', 'sha256': digest,
                    'bytes': archive.stat().st_size, 'files': completion['archive']['files']},
        'functions': {'displacement_components': 'regional shift, centered mismatch, exact RMS decomposition and frozen support denominators',
                      'polar_components': 'proper rotation, right stretch, logarithmic area and deviatoric/shear components of the existing maps',
                      'deformation_components': 'supported A-domain pixel aggregation with explicit unsupported output',
                      'compare_regions': 'seven fixed regions from the single canonical state; no second registration'},
        'fixed': 'optional holdout registration no longer discards a valid primary comparison; failed sensitivity refits retain slots and withhold intervals',
        'pair': {'report': str(features/'pair-01/calibrated/report.html'), 'audit': str(features/'pair-01/regional-audit.json'),
                 'raw_arrays_unchanged': 158, 'supported_triangles': pair['supported_triangles']},
        'native': {'selection_window': [212,232], 'sources': 20, 'fit_performed': False,
                   'calibration_sha256': native['source']['calibration_sha256'], 'radius': native['source']['radius'],
                   'covered_groups': native['source']['covered_groups'], 'array_parity': str(evidence/'native-array-parity.json')},
        'verification': {'commands': str(evidence/'commands.json'), 'wheel_bytes': str(evidence/'wheel-check/installed-byte-check.json'),
                         'browser': str(features/'browser-01/browser.json'), 'ordinary_user_access': str(features/'ordinary-user-verify.json')},
        'next_registry': str(features/'next-registry.json'), 'registered_cases': 355,
        'failures_retained': str(features/'failure-history.json'),
        'open': ['Full-mesh regional components have no independently established anatomical accuracy or propagated localization intervals',
                 'Deformation is directional and weighted by supported A-domain pixels; not an exchange-symmetric surface statistic',
                 'Published PTS labels and filename grouping do not establish independent human visibility labels, person independence or native-training exclusion',
                 'Frozen transfer retains the donor domain label; that declaration alone is not proof of target transfer validity',
                 'Validation is on this Windows runtime; the installed wheel shares interpreter dependencies']}
    write_json_new(args.out, result)
    print(json.dumps({'archive': result['archive'], 'tests': tests, 'native_arrays_per_tier': 80}, indent=2))


if __name__ == '__main__':
    main()
