"""Summarize only executed checks; final ZIP testing is recorded beside the ZIP."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
import json
import xml.etree.ElementTree as ET
import zipfile

from facemath.observation.journal import environment, code_fingerprint

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/wave5'


def hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


def junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = list(root.iter('testsuite'))
    result = {name:sum(int(s.attrib.get(name,0)) for s in suites)
              for name in ('tests','failures','errors','skipped')}
    result['passed'] = result['tests']-result['failures']-result['errors']-result['skipped']
    assert result['failures']==result['errors']==0
    return result


def main() -> None:
    parent_archive=ROOT.parent/'facemath_wave_04.zip'
    parent=ROOT.parent/'wave5_base/facemath_wave_04/src/facemath'
    old,new=hashes(parent),hashes(ROOT/'src/facemath')
    wheel=ROOT/'dist/facemath-0.5.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as z:
        wheel_sources={name.removeprefix('facemath/'):sha256(z.read(name)).hexdigest()
                       for name in z.namelist() if name.startswith('facemath/') and name.endswith('.py')}
    assert wheel_sources==new
    commands={p.name:json.loads(p.read_text()) for p in EVIDENCE.glob('final-*.json')}
    assert commands and all(r['returncode']==0 for r in commands.values())
    tests={'source':junit(EVIDENCE/'final-source-tests.xml'),'installed_wheel':junit(EVIDENCE/'wheel-check/pytest.xml')}
    assert tests['source']==tests['installed_wheel']
    lab=json.loads((EVIDENCE/'laboratory/laboratory.json').read_text())
    assert all(lab['checks'].values())
    browser=json.loads((EVIDENCE/'browser/browser_validation.json').read_text())
    assert browser['passed']
    record={
        'format':'facemath.release_validation.v5','version':'0.5.0',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'actual captured executions; no live human-detector validation or external attestation',
        'environment':environment(),'code_sha256':code_fingerprint(),
        'lineage':{'parent_archive':parent_archive.name,'parent_sha256':sha256(parent_archive.read_bytes()).hexdigest(),
                   'new_modules':sorted(set(new)-set(old)),
                   'modified_modules':sorted(k for k in old if old[k]!=new[k]),
                   'unchanged_modules':sorted(k for k in old if old[k]==new[k]),
                   'removed_modules':sorted(set(old)-set(new))},
        'source_files':new,'wheel_sha256':sha256(wheel.read_bytes()).hexdigest(),
        'wheel_source_equivalence':{'files':len(new),'byte_identical':True},
        'tests':tests,'commands':commands,
        'replay':{'source_qualification':True,'source_pair':True,'wheel_qualification':True,'wheel_pair':True,
                  'scope':'captured observations and source rasters; no detector reinvocation'},
        'laboratory':lab,'browser':browser,
        'native_environment':json.loads((EVIDENCE/'native_environment.json').read_text()),
        'native_failure_control':json.loads((EVIDENCE/'native_failure_control/control.json').read_text()),
        'limits':['one live MediaPipe integration test remains skipped',
                  'synthetic controls are not human localization accuracy or a resemblance score',
                  'wheel checks reuse installed numerical dependencies; not a fresh dependency-stack install',
                  'browser used in-memory HTML after file navigation was blocked; evidence paths checked on disk',
                  'final ZIP extraction/test record is produced beside the ZIP after packaging']}
    (ROOT/'evidence/WAVE5_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'tests':tests,'source_files':len(new),'code_sha256':code_fingerprint()},indent=2))


if __name__=='__main__':main()
