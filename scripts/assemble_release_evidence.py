"""Assemble a release summary from saved execution records; never rerun or invent them.

This is the release engineer's consolidation step, not an execution trace producer.
Pass the actual work directory whose logs and wheel-check records were retained.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def read(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def inventory(p: Path, root: Path) -> dict:
    return {'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
            'sha256': sha256(p.read_bytes()).hexdigest()}


def suite(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = list(root.iter('testsuite'))
    total = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
             for key in ('tests', 'failures', 'errors', 'skipped')}
    total['passed'] = total['tests'] - total['failures'] - total['errors'] - total['skipped']
    total['seconds'] = sum(float(s.attrib['time']) for s in suites)
    total['junit_timestamps'] = [s.attrib.get('timestamp') for s in suites]
    total['skipped_cases'] = [{'name': c.attrib['name'], 'reason': c.find('skipped').attrib.get('message')}
                            for c in root.iter('testcase') if c.find('skipped') is not None]
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    a = parser.parse_args(); project = a.project.resolve(); work = a.work.resolve()
    output = project/'evidence/wave10/execution'; output.mkdir(parents=True, exist_ok=True)
    raw = []
    for sub in ('logs', 'wheel-check'):
        dest = output/sub; dest.mkdir(exist_ok=True)
        for p in sorted((work/sub).iterdir()):
            if p.is_file() and p.suffix in ('.txt', '.xml', '.json'):
                shutil.copy2(p, dest/p.name); raw.append(inventory(dest/p.name, project))
    commands = []
    for p in sorted((output/'logs').glob('*.result.json')):
        commands.append({'record': p.relative_to(project).as_posix(), **read(p)})
    wheel_commands = [dict(record=p.relative_to(project).as_posix(), **read(p))
                      for p in sorted((output/'wheel-check').glob('command-*.json'))]
    from facemath.observation.journal import environment, code_fingerprint
    lab = read(project/'evidence/wave10/laboratory/laboratory.json')
    browsers = {kind: read(project/f'evidence/wave10/{kind}/browser_validation.json')
                for kind in ('browser', 'joint-browser', 'relation-browser')}
    lineage = read(project/'evidence/wave10/lineage_and_wheel_equivalence.json')
    out = {
        'format': 'facemath.release-validation.v10', 'version': '0.10.0',
        'summary_created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'summary of actual retained executions; raw logs separate; synthetic controls are not human accuracy or native model inference',
        'environment': environment(), 'code_sha256': code_fingerprint(),
        'parent': {'archive': 'facemath_wave_09.zip',
                   'sha256': '690568754812a5d34fff61cb41be634617d2078a89caf0fcfd1ec7c758baef1f',
                   'reproduced_tests': suite(output/'logs/parent.xml')},
        'lineage_and_wheel_equivalence': lineage,
        'source_tests': suite(output/'logs/source_final.xml'),
        'installed_wheel_tests': suite(output/'wheel-check/pytest.xml'),
        'development_failure_suite': suite(output/'logs/source_first.xml'),
        'laboratory_summary': {
            'kind': lab['kind'], 'seed': lab['seed'],
            'record': 'evidence/wave10/laboratory/laboratory.json',
            'outcomes': {name: {k:v for k,v in lab[name].items() if k != 'admission'}
                         for name in ('positive_control', 'negative_control')},
            'admission': {name: {k:lab[name]['admission'][k]
                                for k in ('status','passed','policy','tail_probability_per_endpoint','multiplicity','categories')}
                          for name in ('positive_control','negative_control')},
            'additional_fields': {k:v for k,v in lab.items()
                                  if k not in ('kind','seed','format','positive_control','negative_control')}
        },
        'native_smoke': read(project/'evidence/wave10/native_smoke/smoke.json'),
        'native_transport_probe': read(project/'evidence/wave10/native_transport_probe.json'),
        'native_api_fixture': {
            'record': 'evidence/wave10/execution/logs/native_api_control.txt',
            'execution': read(output/'logs/native_api_control.result.json'),
            'native_inference_executed': False,
            'scope': 'patched Python API fixture only; not MediaPipe, pretrained weights, or human reference validation'},
        'consolidated_numeric_controls': read(project/'evidence/wave10/numerical_controls/controls.json'),
        'browser_checks': browsers,
        'source_replay': {key: read(output/f'logs/{key}.result.json') for key in
                          ('source_study_replay','source_pair_replay','research_replay')},
        'actual_command_records': commands, 'installed_wheel_commands': wheel_commands,
        'retained_raw_execution_files': raw,
        'logging_boundary': 'Some early wrappers recorded argv/cwd/returncode/elapsed only. Missing UTC fields are not retroactively reconstructed. JUnit and later native/wheel records retain their own original timestamps.',
        'final_archive_validation': 'After sealing the archive, extracted-suite and replay outcomes will be stored in the adjacent facemath_wave_10_release.json, not retroactively inserted in this sealed record.',
        'limits': [
            'No live MediaPipe inference or independently annotated real-portrait accuracy study was completed.',
            'The positive/negative acceptance workflows are engineered deterministic synthetic controls, not exchangeable population experiments.',
            'Sampling checks of analytic outer bounds are not formal interval arithmetic proofs or exact extrema.',
            'Held-out acceptance requires the declared sampling assumptions; IDs do not prove independence or blinding; adaptive repeated studies are not controlled.',
            'Byte verification and numerical replay establish consistency with retained inputs, not external authenticity or real-world truth.',
            'All Python executions used one Linux numerical dependency stack; a target wheel installation is not a fresh dependency stack.',
            'The multi-OS CI matrix is supplied configuration, not executed Windows/macOS validation.',
            'Browser file navigation was blocked; actual emitted HTML was tested in memory and paths inspected separately.',
            'Historical formula registry remains an inventory, not proof that all earlier mathematical analogies are implemented.',
            'No new original-video face score, identity analysis, or historical neural telemetry is claimed.']
    }
    for key in ('source_tests','installed_wheel_tests'):
        assert out[key]['passed']==942 and out[key]['skipped']==1 and out[key]['failures']==out[key]['errors']==0
    target=project/'evidence/WAVE10_VALIDATION.json'
    target.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(inventory(target,project),indent=2))

if __name__=='__main__': main()
