"""Summarize executed checks; never manufacture a validation result."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import zipfile

from facemath.observation.journal import environment, code_fingerprint

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'evidence/wave6'


def junit(path: Path, require_pass: bool=True) -> dict:
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    out={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    out['passed']=out['tests']-out['failures']-out['errors']-out['skipped']
    out['elapsed_seconds']=sum(float(s.attrib.get('time',0)) for s in suites)
    if require_pass:assert out['failures']==out['errors']==0,out
    return out


def main() -> None:
    parent=ROOT.parent/'facemath_wave_05.zip'
    wheel=ROOT/'dist/facemath-0.6.0-py3-none-any.whl'
    with zipfile.ZipFile(parent) as z:
        old={n.split('/src/facemath/',1)[1]:sha256(z.read(n)).hexdigest() for n in z.namelist()
             if '/src/facemath/' in n and n.endswith('.py')}
    new={p.relative_to(ROOT/'src/facemath').as_posix():sha256(p.read_bytes()).hexdigest()
         for p in sorted((ROOT/'src/facemath').rglob('*.py'))}
    with zipfile.ZipFile(wheel) as z:
        built={n.removeprefix('facemath/'):sha256(z.read(n)).hexdigest() for n in z.namelist()
               if n.startswith('facemath/') and n.endswith('.py')}
    assert new==built
    commands={p.name:json.loads(p.read_text()) for p in sorted(EV.glob('final-*.json'))}
    expected={'final-doctor.json','final-source-tests.json','final-laboratory.json','final-pair-replay.json',
              'final-study-replay.json','final-browser.json','final-wheel-build.json','final-wheel-validation.json'}
    assert set(commands)==expected
    assert all(r['returncode']==0 for r in commands.values())
    tests={'source':junit(EV/'final-source-tests.xml'),'installed_wheel':junit(EV/'wheel-check/pytest.xml')}
    assert {k:v for k,v in tests['source'].items() if k!='elapsed_seconds'}=={k:v for k,v in tests['installed_wheel'].items() if k!='elapsed_seconds'}
    laboratory=json.loads((EV/'laboratory/laboratory.json').read_text())
    assert laboratory['set_controls']['sampled_relation_violations']==0
    assert laboratory['set_controls']['sampled_refit_violations']==0
    assert all(r['raw_arrays_unchanged'] for r in laboratory['queries'])
    browser=json.loads((EV/'browser/browser_validation.json').read_text())
    assert browser['passed']
    record={
        'format':'facemath.release_validation.v6','version':'0.6.0',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'actual executed source/wheel tests, frozen-observation replay and synthetic controls; not live human validation or external attestation',
        'environment':environment(),'code_sha256':code_fingerprint(),
        'lineage':{'parent_archive':parent.name,'parent_archive_sha256':sha256(parent.read_bytes()).hexdigest(),
            'new_modules':sorted(set(new)-set(old)),
            'modified_modules':sorted(k for k in old if k in new and old[k]!=new[k]),
            'unchanged_modules':sorted(k for k in old if k in new and old[k]==new[k]),
            'removed_modules':sorted(set(old)-set(new))},
        'source_files':new,'wheel_sha256':sha256(wheel.read_bytes()).hexdigest(),
        'wheel_source_equivalence':{'files':len(new),'byte_identical':True},
        'tests':tests,'baseline_tests':junit(EV/'inherited-installed.xml'),
        'retained_development_failures':{
            'uninstalled_parent':{'result':junit(EV/'inherited.xml',False),
                'cause':'six subprocess imports failed before editable installation; source parent then passed 370 tests'},
            'unsupported_chin':{'log':'wave6/intrinsic-first.txt',
                'cause':'off-image chin influenced canonical orientation; supported-anchor fallback fixed'},
            'old_version_assertions':{'result':junit(EV/'integration-first.xml',False),
                'cause':'two assertions expected old 0.5.0; version expectations updated without weakening math checks'},
            'unbounded_error_gate':{'log':'wave6/intrinsic-v3.txt',
                'cause':'new test expected processing despite inherited unbounded-calibration gate; revised test verifies refusal and low-level bound-state separately'}},
        'commands':commands,'laboratory':laboratory,'browser':browser,
        'native_environment':json.loads((EV/'native_environment.json').read_text()),
        'replay':{'source_qualification':True,'source_pair':True,'wheel_qualification':True,'wheel_pair':True,
                  'scope':'recomputed from saved observations and original rasters; not detector reinvocation'},
        'limits':['live MediaPipe integration remains skipped; model not obtained',
                  'known-coordinate synthetic controls are not human accuracy or resemblance estimates',
                  'source uncertainty assumptions remain necessary; domain-shift undercoverage retained',
                  'conservative float64 analytic enclosures, not formal interval arithmetic or exact extrema',
                  'wheel tests reuse installed numerical dependencies, not a fresh dependency-stack or Windows test',
                  'browser rendering used in-memory HTML after local navigation failed',
                  'superseded development bundles are separate from the final evidence to avoid mixed code fingerprints',
                  'fresh archive extraction results are recorded beside the ZIP after packaging']}
    (ROOT/'evidence/WAVE6_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'tests':tests,'new_modules':record['lineage']['new_modules'],
        'modified_modules':record['lineage']['modified_modules'],'python_sources':len(new),'code_sha256':code_fingerprint()},indent=2))


if __name__=='__main__':main()
