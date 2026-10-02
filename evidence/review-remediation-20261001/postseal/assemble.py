"""Collect already executed outcomes without rewriting a sealed result."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime,timezone

root=Path(__file__).resolve().parents[2];run=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'scripts'))
from verify_release import verify

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def suite(path):
    xml=ET.parse(path).getroot();suites=list(xml.iter('testsuite'))
    value={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    value['passed']=value['tests']-value['failures']-value['errors']-value['skipped']
    value['seconds']=sum(float(s.get('time',0)) for s in suites)
    value['skips']=[{'test':c.get('name'),'reason':c.find('skipped').get('message')} for c in xml.iter('testcase') if c.find('skipped') is not None]
    return value

gates=run/'verified314';final=run/'release-final';validation=read(final/'validation.json')
assert validation['completed']
out=root/'evidence/review-remediation-20261001/postseal';out.mkdir(exist_ok=False)
for path in final.iterdir():
    if path.is_file() and path.suffix in {'.json','.xml','.log'}:shutil.copyfile(path,out/path.name)
for name in ('source311-final.log','source311-final.json','source311-final.xml','validation314-final.log','validation314-final.json',
             'release-final-driver.log','release-final-driver.json','shipped-checks.log','shipped-checks.json','shipped_checks.py'):
    shutil.copyfile(run/name,out/name)
shutil.copyfile(__file__,out/'assemble.py')
for name in ('laboratory.json',):shutil.copyfile(final/'archive-lab'/name,out/('archive-'+name))
archive=final/'facemath-0.10.1.zip';published=root/'packages/facemath-0.10.1-review-20261001.zip'
with archive.open('rb') as source,published.open('xb') as target:shutil.copyfileobj(source,target)
assert digest(archive)==digest(published)
checks={
    'parent':suite(run/'parent.xml'),
    'first_patched_python314':suite(run/'final314/source.xml'),
    'first_patched_python311':suite(run/'source311.xml'),
    'source_python314':suite(gates/'source.xml'),
    'source_python311':suite(run/'source311-final.xml'),
    'installed_wheel_python314':suite(gates/'wheel-check/pytest.xml'),
    'extracted_archive_python314':suite(final/'archive.xml'),
}
for name in ('source_python314','source_python311','installed_wheel_python314','extracted_archive_python314'):
    assert checks[name]['failures']==checks[name]['errors']==0
lab=read(gates/'source-lab/laboratory.json')
assert read(run/'shipped-checks.json')['returncode']==0
record={
    'schema':'facemath.review_remediation.v1','version':'0.10.1','created_utc':datetime.now(timezone.utc).isoformat(),
    'baseline_review_sha256':digest(root/'REPOSITORY_REVIEW.json'),
    'parent':read(run/'parent-result.json'),
    'engine_code_sha256':read(gates/'environment.log')['engine_code_sha256'],
    'environment':read(gates/'environment.log'),
    'suites':checks,'archive':dict(verify(published),path=published.relative_to(root).as_posix()),
    'wheel_bytes':read(gates/'wheel-check/installed-byte-check.json'),
    'packaging_followup':{'reason':'Actual Windows wheel metadata exposed CRLF parser bug after source/wheel suites.',
                          'regressions':'6 passed on both Python 3.11 and 3.14 after parser correction',
                          'suite_count_difference':'Final archive adds one packaging regression; application source and wheel bytes unchanged.',
                          'failed_attempt':'evidence/review-remediation-20261001/packaging-correction/validation314-final.json',
                          'final_distinct_run':validation},
    'milestone':{'positive_accepted':lab['positive_control']['admission']['passed'],
                 'negative_withheld':not lab['negative_control']['admission']['passed'],
                 'raw_arrays_checked':lab['admitted_pair']['raw_arrays_checked'],
                 'raw_arrays_unchanged':lab['admitted_pair']['raw_arrays_unchanged'],
                 'scope':'engineered synthetic controls, not native human-reference validation'},
    'implemented_findings':[
        {'id':'I01','change':'Locked stable acquisition snapshot with recreated ephemeral sentinel','files':['src/facemath/observation/study_io.py','tests/test_wave5_study.py']},
        {'id':'I02','change':'OpenCV 4 constraint and explicit Haar API/asset failures','files':['pyproject.toml','src/facemath/observation/provider.py','tests/test_wave3_provider.py']},
        {'id':'I03','change':'Original decoded ZIP names and complete path inventory checked before extraction','files':['src/facemath/observation/deployment.py','tests/test_wave10_snapshot.py']},
        {'id':'I04','change':'Python 3.11-compatible report expression; actual 3.11 suite executed','files':['src/facemath/observation/stratified_report.py']},
        {'id':'I05','change':'0.10.1 project/lock/wheel/installed identity; generated egg-info no longer tracked','files':['pyproject.toml','uv.lock','src/facemath/__init__.py','tests/test_wave3_cli.py']},
        {'id':'I06','change':'Explicit release staging, wheel parity, create-only versioned archive and sidecar','files':['scripts/package_release.py','scripts/verify_release.py','tests/test_release_packaging.py']},
        {'id':'I07','change':'Malformed ZIP fixture verifies local/central header bytes and original decoded names','files':['tests/test_wave10_snapshot.py']},
        {'id':'I08','change':'Held-out support preserves declared anatomical totals in JSON and HTML','files':['src/facemath/observation/measure.py','tests/test_wave3_instrument.py']},
    ],
    'development_progress':{
        'D02':'Windows source 3.11/3.14 and installed/extracted 3.14 gates executed; hosted Linux/macOS and other Python matrix jobs not executed here',
        'D03':'Lock refreshed and two isolated locked core environments tested; native-model/runtime compatibility remains unqualified',
        'D13':'Driver source/wheel gates passed, then exposed CRLF packaging failure. Corrected packaging and archive gates executed in a distinct continuation; no fresh full single-driver rerun is claimed. Parent baseline and all failed attempts retained separately.',
        'D14':'Historical root manifest distinguished from new staged manifests in README and remediation design',
    },
    'remaining_findings':[f['id'] for key in ('1','2','3') for f in read(root/'REPOSITORY_REVIEW.json')[0][key]['items'] if f['id'] not in {f'I{i:02}' for i in range(1,9)}],
    'limits':['No pinned native model/runtime or live independent-reference study was executed; models/ contains only its README.',
              'No browser pointer/annotation-coordinate acceptance was executed; D05/D06 remain open.',
              'I09-I16 and the remaining research/operational items are not claimed fixed.',
              'The two skipped tests require real native assets and Windows directory-symlink permission respectively.',
              'Installed-wheel testing uses the validated 3.14 numerical stack; it is not another independent dependency resolution.',
              'Byte hashes and replay establish internal consistency, not external authenticity or human accuracy.'],
    'records':{'preseal':'evidence/review-remediation-20261001/preseal',
               'baseline_and_development':'evidence/review-remediation-20261001/baseline-and-development',
               'postseal':'evidence/review-remediation-20261001/postseal'},
}
for dest in (root/'REPOSITORY_REMEDIATION.json',out/'SUMMARY.json',published.with_suffix('.validation.json')):
    with dest.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps({'archive':record['archive'],'suites':checks,'milestone':record['milestone']},indent=2))
