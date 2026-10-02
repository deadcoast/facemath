"""Retain actual final evidence and failure receipts without replacing old runs."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

run=Path(__file__).resolve().parent
root=run.parents[1]
release=run/'release'
evidence=root/'evidence/bound-utility-20261002'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def retain_files(source,target,suffixes=('.json','.log','.xml','.py')):
    target.mkdir(parents=True,exist_ok=False)
    for path in sorted(source.iterdir()):
        if source==run and path.name.startswith('retain-release.'):
            continue
        if path.is_file() and path.suffix in suffixes:
            shutil.copyfile(path,target/path.name)

def suite(path):
    data=ET.parse(path).getroot()
    nodes=[data] if data.tag=='testsuite' else list(data)
    values={k:sum(int(n.attrib.get(k,0)) for n in nodes) for k in ('tests','failures','errors','skipped')}
    values['passed']=values['tests']-values['failures']-values['errors']-values['skipped']
    values['skip_reasons']=sorted({s.attrib.get('message','') for s in data.iter('skipped')})
    assert values['failures']==values['errors']==0
    return values

def benchmark(path):
    pipeline=read(path/'pipeline.json'); q=read(path/'qualification/qualification.json')
    assert q['raw_coordinates_changed'] is False and q['estimator_selected_using_evaluation'] is False
    assert q['empirical_admission']['passed'] is False
    utility=q['bound_utility']
    assert utility['calibration_sha256']==q['selected_error_model']['calibration_sha256']
    return dict(enrollment=q['enrollment'], selected_error_model=q['selected_error_model'],
        utility_splits=utility['splits'], utility_categories=utility['evaluation_strata'],
        raw_localization_slices=q['slices'], empirical_admission=q['empirical_admission'],
        exact_retained_replay=pipeline['replay_verified'], native_environment=pipeline['environment'],
        grouping_limit=pipeline['grouping_limit'], source_bound_reference_agreement=True,
        human_accuracy_validated=False, evaluation_used_to_retune=False)

assert read(release/'validation.json')['completed']
assert read(run/'linux-validation.json')['returncode']==0
audit=read(run/'release-audit.result.json')
retain_files(run,evidence/'development')
shutil.copytree(run/'linux-native',evidence/'development/linux-native')
superseded=evidence/'development/superseded-intake';superseded.mkdir()
for name in ('recipe.json','conversions.json','study.spec.json','manifest.json'):
    shutil.copyfile(run/'intake/dataset'/name,superseded/name)
post=evidence/'postseal';retain_files(release,post)
retain_files(release/'wheel-check',post/'wheel-check',('.json','.xml'))
shutil.copytree(release/'wheel-check/native-pipeline',post/'installed-native')
shutil.copytree(release/'archive-native',post/'archive-native')
labs={}
for name,path in [('source',release/'source-lab'),('installed',release/'wheel-check/milestone'),('archive',release/'archive-lab')]:
    data=read(path/'laboratory.json')
    assert data['admitted_pair']['raw_arrays_unchanged'] and data['withheld_query']['observer_calls']==0
    labs[name]=dict(positive=data['positive_control']['admission']['status'],negative=data['negative_control']['admission']['status'],
        raw_arrays_checked=data['admitted_pair']['raw_arrays_checked'],raw_arrays_unchanged=True,withheld_observer_calls=0)
    shutil.copyfile(path/'laboratory.json',post/(name+'-laboratory.json'))
archive_source=release/'facemath-0.13.0.zip';archive=root/'packages/facemath-0.13.0-utility-20261002.zip'
for source,target in [(archive_source,archive),(archive_source.with_suffix('.release.json'),archive.with_suffix('.release.json')),
                      (release/'validation.json',archive.with_suffix('.validation.json'))]:
    assert not target.exists();shutil.copyfile(source,target)
result=dict(schema='facemath.bound_utility_validation.v1',version='0.13.0',created_utc=datetime.now(timezone.utc).isoformat(),
    engine_code_sha256=read(release/'environment.log')['engine_code_sha256'],
    parent=read(run/'parent-baseline.json'),
    historical_review_sha256=sha256((root/'REPOSITORY_REVIEW.json').read_bytes()).hexdigest(),
    historical_review_scope='Unchanged historical 79-module review, not a refreshed source review',
    prospective_plan=read(run/'prospective-freeze-v2.json'),
    source_provenance=dict(archive=read(root/'examples/helen-utility-cohort.json')['archive'],
        public_source='https://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz',
        label_source='Original full-precision PTS in public dlib 300W repack; published semi-automatic annotations',
        candidate_images=330,candidate_filename_roots=298,selected_roots=60,groups_per_split=20,
        grouping='numeric filename-root rule is a conservative curator inference; not external identity validation',
        terms=read(root/'examples/helen-utility-cohort.json')['image_terms'],
        selection_outcome_blind=True),
    parent_model_and_array_invariance='development/parent-invariance.result.json',
    previous_cohort_source_overlap=read(run/'source-overlap.result.json'),
    suites=dict(windows_source_311=suite(release/'source.xml'),installed_wheel=suite(release/'wheel-check/pytest.xml'),
                extracted_archive=suite(release/'archive.xml'),linux_311=suite(run/'linux.xml')),
    milestone_controls=labs,
    native_runs={name:benchmark(path) for name,path in [('windows_source',release/'source-native'),
        ('windows_installed',release/'wheel-check/native-pipeline'),('windows_archive',release/'archive-native'),('linux_source',run/'linux-native')]},
    repeatability={p.name:read(p) for p in run.glob('compare-*.result.json')},
    fitted_array_repeatability={p.name:read(p) for p in run.glob('fit-*.result.json')},
    native_raw_invariance={p.name:read(p) for p in run.glob('audit-*.result.json')},
    preserved_failures=[
        dict(record='development/focused.json',reason='40 passed; one test constructor omitted required RoutingPolicy definition',resolution='Corrected fixture; focused and final full suites pass'),
        dict(record='development/parent-invariance.json',reason='Audit compared Python tuple-valued serialization directly to decoded JSON lists',resolution='Compared actual JSON-normalized representations and all original array bytes; production math unchanged'),
        dict(record='development/prospective-freeze.json',reason='Pre-inference candidate selection contained two HELEN faces with one filename root',resolution='Superseded before any native observations; retain original plan/intake metadata; select one member per root')],
    package=read(release/'validation.json')['archive'],archive_source_parity=audit,
    installed_bytes='postseal/wheel-check/installed-byte-check.json',
    completion_boundary=dict(automated_sourcing=True,new_human_testing_required=False,
        empirical_query_admission=False,person_independence_established=False,visibility_known=False,
        native_training_overlap_known=False,cross_os_exact_replay=False,
        hosted_execution='Prior scheduler refused jobs before steps due to billing/spending limit; current push checked separately',
        macos_native='Previous constructor abort unresolved'),
    next_actions=[
        dict(priority='P1',task='Prespecify useful width and acquisition-domain criteria before a new independent evaluation',
             references=['docs/BOUND_UTILITY_DESIGN.md','src/facemath/observation/utility.py'],
             acceptance='Freeze criteria, independent source grouping and missingness evidence before outcomes; retain all failures; no retuning this HELEN evaluation'),
        dict(priority='P1',task='Evaluate transfer of a previously frozen fitted model on new source-disjoint data',
             references=['src/facemath/observation/joint.py','docs/WAVE10_ROADMAP.md'],
             acceptance='Keep development center/scatter and calibrated radius fixed; explicit domain-shift limit and full reference denominator; do not silently refit on transfer'),
        dict(priority='P1',task='Resolve actual macOS native constructor failure and execute hosted matrix when scheduler starts',
             references=['.github/workflows/milestone.yml','scripts/check_native_runtime.py'],
             acceptance='Actual isolated native outputs and full regression; billing refusal is not execution')])
with (evidence/'VALIDATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(package=result['package'],suites=result['suites'],milestones=labs),indent=2))
