"""Summarize already executed release records and compare actual distribution bytes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:return sha256(path.read_bytes()).hexdigest()
def modules(root: Path) -> dict[str,str]:return {p.relative_to(root).as_posix():digest(p)for p in sorted(root.rglob('*.py'))}
def counts(path: Path) -> dict:
    cases=list(ET.parse(path).getroot().iter('testcase'))
    skipped=sum(c.find('skipped')is not None for c in cases)
    failures=sum(c.find('failure')is not None for c in cases)
    errors=sum(c.find('error')is not None for c in cases)
    return {'total':len(cases),'passed':len(cases)-skipped-failures-errors,'skipped':skipped,'failures':failures,'errors':errors,
            'skip_reasons':[c.find('skipped').get('message') for c in cases if c.find('skipped')is not None]}


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);ap.add_argument('--parent-archive',required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];parent=Path(a.parent).resolve();e=root/'evidence/wave8'
    source=modules(root/'src/facemath');old=modules(parent/'src/facemath')
    wheel=root/'dist/facemath-0.8.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel)as z:
        packed={n[len('facemath/'):]:sha256(z.read(n)).hexdigest()for n in z.namelist()if n.startswith('facemath/')and n.endswith('.py')}
    assert source==packed, 'source/wheel Python bytes differ'
    original_tests=parent/'tests';modified=[]
    for f in original_tests.glob('*.py'):
        target=root/'tests'/f.name;assert target.exists()
        if target.read_bytes()!=f.read_bytes():
            modified.append(f.name)
            assert f.name=='test_wave3_cli.py'and target.read_text()==f.read_text().replace('0.7.0','0.8.0')
    records={p.stem:json.loads(p.read_text())for p in e.glob('final-*.json')}
    assert len(records)==10 and all(r['returncode']==0 for r in records.values())
    source_counts=counts(e/'final-source-tests.xml');wheel_counts=counts(e/'wheel-check/pytest.xml')
    assert source_counts==wheel_counts and source_counts['failures']==source_counts['errors']==0
    installed={p.stem:json.loads(p.read_text())for p in (e/'wheel-check').glob('command-*.json')}
    assert all(r['returncode']==0 for r in installed.values())
    imported=json.loads(installed['command-01']['stdout'])
    lab=json.loads((e/'laboratory/laboratory.json').read_text())
    checks={name:json.loads((e/name/'browser_validation.json').read_text())for name in ('browser','joint-browser','relation-browser')}
    assert all(x['passed']for x in checks.values())
    value={'format':'facemath.release_validation.v8','version':'0.8.0','generated_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'executed source and independently installed-wheel tests/replay; fresh-final-ZIP checks recorded in external release JSON',
        'environment':json.loads((e/'native_environment.json').read_text()),
        'source_tests':source_counts,'installed_wheel_tests':wheel_counts,
        'distribution_equivalence':{'package_python_files':len(source),'source_equals_wheel':True,'source_hashes':source,
            'wheel_sha256':digest(wheel),'installed_import':imported},
        'lineage':{'parent_archive':Path(a.parent_archive).name,'parent_archive_sha256':digest(Path(a.parent_archive)),
            'new_modules':[k for k in source if k not in old], 'modified_modules':[k for k in old if k in source and old[k]!=source[k]],
            'unchanged_modules':[k for k in old if source.get(k)==old[k]],'removed_modules':[k for k in old if k not in source],
            'inherited_tests_modified':modified,'test_change':'only CLI release-version assertions; all inherited numerical assertions retained',
            'parent_python_hashes':old},
        'executed_commands':records,'installed_wheel_commands':installed,
        'laboratory':{'acquisition':lab['acquisition'],'radii':[{k:r[k]for k in('stratum','quantile','calibration_groups','rank','status')}for r in lab['category_radii']],
            'hidden_shift':lab['hidden_within_clear_shift'],'queries':lab['queries'],'set_controls':lab['set_controls'],
            'unsupported_queries':lab['unsupported_queries']},'browser':checks,
        'limits':['synthetic observations; no live MediaPipe inference or independently labeled human performance',
            'source/wheel tests reuse this numerical dependency stack, not a clean dependency installation or Windows execution',
            'numerical replay uses captured observations, not detector reexecution',
            'category declarations do not detect hidden shift or establish exchangeability',
            'hashes establish self-consistency, not external provenance or historical model telemetry']}
    path=root/'evidence/WAVE8_VALIDATION.json';path.write_text(json.dumps(value,indent=2));print(path)

if __name__=='__main__':main()
