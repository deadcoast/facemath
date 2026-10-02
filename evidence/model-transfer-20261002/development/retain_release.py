"""Retain completed validation and package evidence without replacing prior runs."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

run=Path(__file__).resolve().parent
root=run.parents[1]
release=run/'release'
evidence=root/'evidence/model-transfer-20261002'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def retain_files(source,target,existing=False,suffixes=('.json','.log','.xml','.py')):
    target.mkdir(parents=True,exist_ok=existing)
    for path in sorted(source.iterdir()):
        if source==run and path.name.startswith('retain-release.'):
            continue
        if path.is_file() and path.suffix in suffixes:
            dest=target/path.name
            if dest.exists():
                assert dest.read_bytes()==path.read_bytes(),str(dest)
            else:shutil.copyfile(path,dest)

def suite(path):
    data=ET.parse(path).getroot()
    nodes=[data] if data.tag=='testsuite' else list(data)
    values={k:sum(int(n.attrib.get(k,0)) for n in nodes) for k in ('tests','failures','errors','skipped')}
    values['passed']=values['tests']-values['failures']-values['errors']-values['skipped']
    values['skip_reasons']=sorted({s.attrib.get('message','') for s in data.iter('skipped')})
    assert values['failures']==values['errors']==0
    return values

def benchmark(path):
    pipeline=read(path/'pipeline.json'); report=read(path/'evaluation/transfer.json')
    assert report['fit_performed'] is False and report['raw_coordinates_changed'] is False
    assert report['empirical_admission'] is False
    return dict(report=report, exact_retained_replay=pipeline['replay_verified'],
        native_environment=pipeline['environment'],grouping_limit=pipeline['grouping_limit'],
        human_accuracy_validated=False,evaluation_used_to_retune=False)

assert read(release/'validation.json')['completed']
assert read(run/'linux.json')['returncode']==0
assert read(run/'cross-host.result.json')['expected_refusal']
audit=read(run/'release-audit.result.json')
retain_files(run,evidence/'development',existing=True)
shutil.copytree(run/'linux-native',evidence/'development/linux-native')
post=evidence/'postseal';retain_files(release,post)
retain_files(release/'wheel-check',post/'wheel-check',suffixes=('.json','.xml'))
shutil.copytree(release/'wheel-check/native-pipeline',post/'installed-native')
shutil.copytree(release/'archive-native',post/'archive-native')
labs={}
for name,path in [('source',release/'source-lab'),('installed',release/'wheel-check/milestone'),('archive',release/'archive-lab')]:
    data=read(path/'laboratory.json')
    assert data['admitted_pair']['raw_arrays_unchanged'] and data['withheld_query']['observer_calls']==0
    labs[name]=dict(positive=data['positive_control']['admission']['status'],negative=data['negative_control']['admission']['status'],
        raw_arrays_checked=data['admitted_pair']['raw_arrays_checked'],raw_arrays_unchanged=True,withheld_observer_calls=0)
    shutil.copyfile(path/'laboratory.json',post/(name+'-laboratory.json'))
archive_source=release/'facemath-0.14.0.zip';archive=root/'packages/facemath-0.14.0-transfer-20261002.zip'
for source,target in [(archive_source,archive),(archive_source.with_suffix('.release.json'),archive.with_suffix('.release.json')),
                      (release/'validation.json',archive.with_suffix('.validation.json'))]:
    assert not target.exists();shutil.copyfile(source,target)
result=dict(schema='facemath.model_transfer_validation.v1',version='0.14.0',created_utc=datetime.now(timezone.utc).isoformat(),
    engine_code_sha256=read(release/'environment.log')['engine_code_sha256'],
    source_python_modules=len(list((root/'src/facemath').rglob('*.py'))),
    parent=read(run/'parent-baseline.json'),
    parent_commit='97d31a37d947689c9df7e0df20f32c0b2883470b',
    historical_review_sha256=sha256((root/'REPOSITORY_REVIEW.json').read_bytes()).hexdigest(),
    historical_review_scope='Unchanged historical 79-module review, not a refreshed source review',
    prospective_plan=read(run/'prospective-freeze.json'),
    source_provenance=dict(archive=read(root/'examples/ibug-transfer-cohort.json')['archive'],
        public_source='https://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz',
        label_source='Original full-precision PTS in public dlib 300W repack; published semi-automatic annotations',
        candidate_annotations=135,candidate_filename_roots=114,selected_roots=20,all_evaluation=True,
        grouping='Filename-root rule consolidates identical photograph bytes with separate face annotations; not external identity validation',
        terms=read(root/'examples/ibug-transfer-cohort.json')['image_terms'],selection_outcome_blind=True),
    frozen_model=dict(capsule='helen-model.json',donor='evidence/bound-utility-20261002/preseal/native-pipeline/qualification',
        donor_enrolled_cases=60,donor_model_bytes_sha256=audit['model_bytes_sha256'],
        unchanged_in_all_four_runs=True,refit_performed=False),
    suites=dict(windows_source_311=suite(release/'source.xml'),installed_wheel=suite(release/'wheel-check/pytest.xml'),
                extracted_archive=suite(release/'archive.xml'),linux_311=suite(run/'linux.xml')),
    milestone_controls=labs,
    native_runs={name:benchmark(path) for name,path in [('windows_source',release/'source-native'),
        ('windows_installed',release/'wheel-check/native-pipeline'),('windows_archive',release/'archive-native'),('linux_source',run/'linux-native')]},
    repeatability=audit['comparisons'],native_raw_invariance=audit['pipelines'],
    cross_host_negative_control=read(run/'cross-host.result.json'),
    preserved_failures=[
        dict(record='development/focused.json',reason='99 passed; one expected-error regex named raters while reference consensus rejected unavailable eye corners',resolution='Corrected assertion to existing reference-observable contract; no production behavior weakened'),
        dict(record='development/intake.json',reason='IBUG full candidate inventory includes identical photograph bytes with separate face annotations',resolution='No inference ran; retained alias list; one face per explicit filename root, reject cross-root byte aliases'),
        dict(record='tool transcript',reason='Initial inline Python command to copy the Linux helper failed PowerShell quoting before execution',resolution='Used literal PowerShell string replacement; no source or input data changed'),
        dict(record='development/cross-host.json',reason='WSL could not read original Windows secure temporary-directory owner ACL',resolution='Created and byte-verified a distinct portable copy; original bytes and permissions unchanged'),
        dict(record='development/cross-host-portable.json',reason='WSL ephemeral /tmp environment disappeared after the completed validation process exited',resolution='Bootstrapped an offline locked environment and ran the negative control in one invocation; expected numerical-runtime refusal confirmed')],
    package=read(release/'validation.json')['archive'],archive_source_parity={k:v for k,v in audit.items() if k not in ('pipelines','comparisons')},
    installed_bytes='postseal/wheel-check/installed-byte-check.json',
    completion_boundary=dict(automated_sourcing=True,new_human_testing_required=False,
        empirical_query_admission=False,person_independence_established=False,visibility_known=False,
        native_training_overlap_known=False,cross_os_exact_replay=False,
        stratified_model_transfer=False,
        hosted_execution='Prior scheduler refused jobs before steps due to billing/spending limit; current push checked separately',
        macos_native='Previous constructor abort unresolved'),
    next_actions=[
        dict(priority='P1',task='Freeze useful-width and domain criteria for a distinct development/calibration study with a new evaluation cohort',
             references=['docs/MODEL_TRANSFER_DESIGN.md','docs/BOUND_UTILITY_DESIGN.md'],
             acceptance='Retain this 20-group transfer as fixed evidence; no retuning or relabeling its evaluation as fresh acceptance'),
        dict(priority='P1',task='Track all prior cohort enrollment across model versions and transfer studies',
             references=['src/facemath/observation/transfer.py','src/facemath/observation/source_cohort.py'],
             acceptance='A prospective registry binds prior evaluation source/group/lineage exclusions; donor capsule alone does not establish unseen status across unrelated studies'),
        dict(priority='P1',task='Define stratified transfer routing and category-complete denominators',
             references=['src/facemath/observation/stratified.py','docs/MODEL_TRANSFER_DESIGN.md'],
             acceptance='Preserve every category radius/fit/protocol; no flattening or pooled fallback; new frozen domain contract'),
        dict(priority='P1',task='Resolve actual macOS native constructor failure and execute hosted matrix when scheduler starts',
             references=['.github/workflows/milestone.yml','scripts/check_native_runtime.py'],
             acceptance='Actual isolated native outputs and full regression; billing refusal is not execution')])
with (evidence/'VALIDATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(package=result['package'],suites=result['suites'],milestones=labs),indent=2))
