"""Create release metadata from actual local artifacts, never synthetic run counts."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from facemath.observation.artifacts import verify_bundle
from facemath.observation.journal import code_fingerprint, environment


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_counts(path: Path) -> dict:
    suites = ET.parse(path).getroot().findall('testsuite')
    total = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ('tests', 'failures', 'errors', 'skipped')}
    total['passed'] = total['tests'] - total['failures'] - total['errors'] - total['skipped']
    total['seconds'] = sum(float(s.attrib.get('time', 0)) for s in suites)
    return total


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit('usage: release_summary.py PARENT_ZIP EXTRACTED_PARENT_ROOT')
    parent_zip, parent_root = Path(sys.argv[1]), Path(sys.argv[2])
    source = ROOT / 'src' / 'facemath'
    old_source = parent_root / 'src' / 'facemath'
    old = {str(p.relative_to(old_source)): digest(p) for p in old_source.rglob('*.py')}
    new = {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')}
    lineage = {'parent_archive': parent_zip.name, 'parent_archive_sha256': digest(parent_zip),
               'new_python_modules': sorted(set(new) - set(old)),
               'modified_python_modules': sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
               'unchanged_python_modules': sorted(k for k in old.keys() & new.keys() if old[k] == new[k]),
               'removed_python_modules': sorted(set(old) - set(new)),
               'source_files': new}
    coverage = json.loads((ROOT / 'evidence/coverage.json').read_text())
    entries = [v['summary'] for k, v in coverage['files'].items() if k.startswith('src/facemath/observation/')]
    sums = {k: sum(row[k] for row in entries) for k in ('covered_lines', 'num_statements', 'covered_branches', 'num_branches')}
    sums['combined_line_branch_percent'] = 100 * (sums['covered_lines'] + sums['covered_branches']) / (sums['num_statements'] + sums['num_branches'])
    wheel = ROOT / 'dist/facemath-0.3.0-py3-none-any.whl'
    data = {
        'format': 'facemath.release-validation.v3', 'version': '0.3.0',
        'scope': 'actual captured tool-side executions; not detector accuracy or external attestation',
        'environment': environment(), 'code_sha256': code_fingerprint(),
        'lineage': lineage,
        'tests': {name: test_counts(ROOT / path) for name, path in (
            ('coverage_run', 'evidence/pytest.xml'),
            ('source_release_run', 'evidence/release-run/pytest.xml'),
            ('fresh_installed_wheel', 'evidence/installed-wheel/pytest.xml'))},
        'coverage': {'whole_package': coverage['totals'], 'active_observation_subpackage': sums,
                     'note': 'code execution coverage is not mathematical equivalence or detector accuracy'},
        'bundle_checks': {name: verify_bundle(ROOT / path) for name, path in (
            ('demo', 'evidence/release-run/demo'), ('source_replay', 'evidence/release-run/replay'),
            ('installed_wheel_replay', 'evidence/installed-wheel/replayed'))},
        'laboratory': json.loads((ROOT / 'evidence/release-run/laboratory.json').read_text()),
        'browser': json.loads((ROOT / 'evidence/browser-qa-verified/browser-check.json').read_text()),
        'wheel': {'filename': wheel.name, 'bytes': wheel.stat().st_size, 'sha256': digest(wheel)},
        'explicit_limits': [
            'Real MediaPipe model inference not executed; one live integration test skipped.',
            'Provider API assembly/remapping tested with fixtures, not learned-model weights.',
            'No human reference population or independently annotated localization-accuracy study included.',
            '2D projected anatomy; no 3D neutralization or identity inference.',
            'Inherited formula registry is not proof of complete mathematical implementation.',
            'Configured GitHub Actions matrix has not been executed here.',
        ],
    }
    for name, counts in data['tests'].items():
        assert counts['failures'] == counts['errors'] == 0, name
    target = ROOT / 'evidence/VALIDATION.json'
    target.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'validation': str(target), 'tests': data['tests'],
                      'new_modules': len(lineage['new_python_modules']),
                      'modified_modules': len(lineage['modified_python_modules']),
                      'unchanged_modules': len(lineage['unchanged_python_modules']),
                      'code_sha256': data['code_sha256']}, indent=2))


if __name__ == '__main__':
    main()
