"""Read executed records; do not recreate execution traces or invent statuses."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile


def pytest_counts(path: Path) -> dict:
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    counts['skips']=[{'test':c.attrib.get('name'),'reason':c.find('skipped').attrib.get('message','')}
                     for s in suites for c in s.findall('testcase') if c.find('skipped') is not None]
    return counts


def main():
    root=Path(__file__).resolve().parents[1];out=root/'evidence/wave7'
    parent=Path(sys.argv[1]);parent_archive=Path(sys.argv[2])
    actual={p.relative_to(root/'src/facemath').as_posix():sha256(p.read_bytes()).hexdigest()
            for p in (root/'src/facemath').rglob('*.py')}
    old={p.relative_to(parent/'src/facemath').as_posix():sha256(p.read_bytes()).hexdigest()
         for p in (parent/'src/facemath').rglob('*.py')}
    wheel=root/'dist/facemath-0.7.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as z:
        inside={n.removeprefix('facemath/'):sha256(z.read(n)).hexdigest() for n in z.namelist()
                if n.startswith('facemath/') and n.endswith('.py')}
    assert inside==actual,'wheel/source code mismatch'
    labels=['final-doctor','final-source-tests','final-laboratory','final-pair-replay','final-study-replay',
            'final-browser','final-inherited-browser','final-wheel-build','final-wheel-validation']
    executions={name:json.loads((out/(name+'.json')).read_text()) for name in labels}
    assert all(v['returncode']==0 for v in executions.values()),'an executed validation failed'
    wheel_commands=[json.loads(p.read_text()) for p in sorted((out/'wheel-check').glob('command-*.json'))]
    assert all(c['returncode']==0 for c in wheel_commands)
    source=pytest_counts(out/'final-source-tests.xml');installed=pytest_counts(out/'wheel-check/pytest.xml')
    assert source['passed']==installed['passed'] and source['skipped']==installed['skipped']
    lab=json.loads((out/'laboratory/laboratory.json').read_text())
    data={'format':'facemath.release_validation.v7','version':'0.7.0','assembled_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'actual executed commands and artifacts; self-consistency, not external attestation',
        'parent':{'file':parent_archive.name,'sha256':sha256(parent_archive.read_bytes()).hexdigest()},
        'lineage':{'new':sorted(actual.keys()-old.keys()),'modified':sorted(k for k in actual.keys()&old.keys() if actual[k]!=old[k]),
                   'unchanged':sorted(k for k in actual.keys()&old.keys() if actual[k]==old[k]),'removed':sorted(old.keys()-actual.keys()),
                   'source_sha256':actual,'wheel_source_byte_equality':True},
        'source_tests':source,'installed_wheel_tests':installed,
        'wheel':{'path':str(wheel.relative_to(root)),'bytes':wheel.stat().st_size,'sha256':sha256(wheel.read_bytes()).hexdigest()},
        'execution_records':{k:{**v,'record':str((out/(k+'.json')).relative_to(root))} for k,v in executions.items()},
        'installed_wheel_commands':wheel_commands,
        'numerical_controls':{'scalar_checks':lab['set_controls']['scalar_relation_checks'],
          'relation_violations':lab['set_controls']['sampled_relation_violations'],
          'shape_violations':lab['set_controls']['sampled_shape_violations'],
          'queries':lab['queries'],'heldout_coverage':lab['selected_error_model']['coverage'],
          'calls':[lab['calls_first_batch'],lab['calls_complete'],lab['calls_after_resume']]},
        'native_environment':json.loads((out/'native_environment.json').read_text()),
        'limitations':['MediaPipe live inference unexecuted; no human detector validation or original-face similarity score',
                       'synthetic controls are explicit; no population calibration',
                       'source and installed-wheel tests reuse this numerical dependency stack; not clean dependency or Windows validation',
                       'report browsers loaded exact emitted HTML in memory after local file navigation was blocked',
                       'bounds are conservative float calculations, not formal interval arithmetic or exact constrained extrema',
                       'replay uses captured observations, not another detector invocation',
                       'same-set marginal comparison loses the shared norm budget as well as cross-point covariance']}
    (root/'evidence/WAVE7_VALIDATION.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'source':source,'wheel':installed,'modules':len(actual),
        'new':len(data['lineage']['new']),'modified':len(data['lineage']['modified']),
        'unchanged':len(data['lineage']['unchanged'])},indent=2))

if __name__=='__main__':main()
