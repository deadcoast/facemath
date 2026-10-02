"""Retain completed release executions without editing any original run."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
release = run/'release-check'
evidence = root/'evidence/native-portability-20261002'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def files(source, target, suffixes=('.log','.json','.xml','.py')):
    target.mkdir(parents=True, exist_ok=False)
    for p in sorted(source.iterdir()):
        if source==run and p.name.startswith('retain-release.'):
            continue  # This command's own log is still open; retain it after exit.
        if p.is_file() and p.suffix in suffixes:
            shutil.copyfile(p,target/p.name)

def suite(path):
    cases=ET.parse(path).getroot().findall('.//testcase')
    errors=sum(c.find('error') is not None for c in cases)
    failures=sum(c.find('failure') is not None for c in cases)
    skips=[{'test':c.attrib['name'],'reason':c.find('skipped').attrib.get('message','')} for c in cases if c.find('skipped') is not None]
    return dict(tests=len(cases),passed=len(cases)-errors-failures-len(skips),errors=errors,failures=failures,skips=skips)

assert read(release/'validation.json')['completed']
assert read(run/'linux-tests-local.json')['returncode']==0
assert read(run/'linux-pipeline.json')['returncode']==0
files(run,evidence/'development')
for name in ('parent-ubuntu','parent-macos','inherited-mixed-env-refusal','actual-native-timeout','linux-native-smoke'):
    shutil.copytree(run/name,evidence/'development'/name)
shutil.copytree(run/'linux-pipeline',evidence/'linux-pipeline')
post=evidence/'postseal'
files(release,post)
files(release/'wheel-check',post/'wheel-check',('.json','.xml'))
for name in ('archive-native',):
    shutil.copytree(release/name,post/name)
shutil.copytree(release/'wheel-check/native-pipeline',post/'installed-native')
for name in ('cross-linux-to-windows','cross-windows-to-linux-transferred'):
    if (run/name).exists():shutil.copytree(run/name,post/name)
labs={}
for name,path in [('source',release/'source-lab'),('installed',release/'wheel-check/milestone'),('archive',release/'archive-lab')]:
    data=read(path/'laboratory.json')
    assert data['admitted_pair']['raw_arrays_unchanged'] and data['withheld_query']['observer_calls']==0
    labs[name]={'positive':data['positive_control']['admission']['status'],
                'negative':data['negative_control']['admission']['status'],
                'raw_arrays_checked':data['admitted_pair']['raw_arrays_checked'],
                'raw_arrays_unchanged':True,'withheld_query_observer_calls':0}
    shutil.copyfile(path/'laboratory.json',post/(name+'-laboratory.json'))

archive_source=release/'facemath-0.11.1.zip'
archive=root/'packages/facemath-0.11.1-portability-20261002.zip'
for source,target in [(archive_source,archive),(archive_source.with_suffix('.release.json'),archive.with_suffix('.release.json')),
                      (release/'validation.json',archive.with_suffix('.validation.json'))]:
    assert not target.exists()
    shutil.copyfile(source,target)

result={'schema':'facemath.native_portability_validation.v1','version':'0.11.1',
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'engine_code_sha256':read(release/'environment.log')['engine_code_sha256'],
    'baseline_review_sha256':sha256((root/'REPOSITORY_REVIEW.json').read_bytes()).hexdigest(),
    'source_archive_parity':'development/archive-source-parity.result.json',
    'installed_bytes':'postseal/wheel-check/installed-byte-check.json',
    'parent':{'path':'packages/facemath-0.11.0-native-20261002.zip',
        'sha256':'bf181b9e8719b99f370bd1e0e17e1f4a187d053e397a7cce5413fb883791bfd4',
        'baseline':{'passed':980,'skipped':2,'record':'development/parent-tests.json'},
        'byte_checks':['development/parent-byte-check.json','development/parent-extracted-byte-check.json']},
    'suites':{'windows_source':suite(release/'source.xml'),'installed_wheel':suite(release/'wheel-check/pytest.xml'),
        'extracted_archive':suite(release/'archive.xml'),'linux_wsl':suite(run/'linux-local.xml'),
        'windows_python314':{'passed':989,'skipped':2,'additional_process_controls_passed':2,
            'records':['development/source314.json','development/source314-added-controls.json']}},
    'milestone_controls':labs,
    'native_runs':{name:read(path/'pipeline.json') for name,path in [
        ('windows_source',release/'source-native'),('installed_wheel',release/'wheel-check/native-pipeline'),
        ('archive',release/'archive-native'),('linux_wsl',run/'linux-pipeline')]},
    'comparisons':{p.name:read(p) for p in run.glob('compare-*.result.json')},
    'comparison_summary':{p.name:{'cases':len(d['cases']),
        'identical_observation_files':d['identical_observations'],
        'exact_coordinate_arrays':sum(c.get('arrays',{}).get('xy',{}).get('bytes_equal',False) for c in d['cases']),
        'exact_sample_arrays':sum(c.get('arrays',{}).get('samples',{}).get('bytes_equal',False) for c in d['cases'])}
        for p in run.glob('compare-*.result.json') for d in [read(p)]},
    'cross_host_replay':{p.name:read(p) for p in run.glob('cross-*.json')},
    'numerical_kernel_records':['development/windows-kernels.log','development/linux-kernels.log'],
    'transfer':{'archive_receipt':'development/windows-native-transfer.json',
        'results':'development/cross-windows-to-linux-transferred.results.json',
        'meaning':'Linux verification of transferred Windows bytes passed; strict numerical replay remained rejected'},
    'negative_controls':{'mixed_environment':read(run/'inherited-mixed-env-refusal/smoke.json'),
        'actual_native_timeout':read(run/'actual-native-timeout/smoke.json'),
        'child_process_faults':'development/process-fault-controls.json'},
    'hosted':{'parent':'development/parent-ci.json','parent_failures':'development/parent-ci-failed.log',
        'diagnostic_run':'development/diagnostic-ci.json','scheduler_reason':'development/diagnostic-ci-annotations.json',
        'status':'new jobs did not start: GitHub account billing or spending-limit condition',
        'macos_native':'unresolved original constructor abort; isolated diagnostic awaits a runnable macOS host'},
    'archive':{'path':archive.relative_to(root).as_posix(),'bytes':archive.stat().st_size,
        'sha256':sha256(archive.read_bytes()).hexdigest(),'validation':'postseal/validation.json'},
    'review_progress':[
        {'ids':['I02','D03'],'change':'one OpenCV contrib 4 wheel; explicit native binding guard; exact installed/source byte checks',
         'remaining':'portable numerical-runtime contract: cross-OS exact replay rejects last-bit derived-score differences; hosted matrix after scheduler condition resolves'},
        {'ids':['D07'],'change':'isolated live tests and retained real child timeout/exit plus actual native timeout',
         'remaining':'multi-descendant and operator-interruption recovery qualification'},
        {'ids':['D01'],'change':'retained same frozen real-image pipeline on Windows and Linux; explicit observation comparisons',
         'remaining':'larger source-disjoint cohort frozen before model evaluation; finite useful calibration bounds'}],
    'limits':['machine-reference agreement is not independently established anatomical accuracy',
        'the nine-photo pilot retains an unbounded radius; alpha, splits and outputs were not retuned',
        'local Linux evidence uses WSL2, not a hosted Ubuntu runner',
        'cross-OS exact qualification replay is not achieved: native coordinates and saved arrays match, but derived scores differ in final floating-point bits',
        'macOS native remains unqualified; core parent matrix is historical evidence',
        'installed wheel is outside the source tree but shares interpreter dependencies',
        'postseal evidence is adjacent to the archive; it was not retroactively inserted into sealed bytes',
        'observed repeatability is not a guarantee for all inputs, hosts or future runtimes']}
with (evidence/'VALIDATION.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps({'archive':result['archive'],'suites':result['suites'],'milestone_controls':labs},indent=2))
