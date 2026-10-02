"""Retain completed gates and the actual superseded failures without rewriting them."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[2]
run = Path(__file__).resolve().parent
release = run/'release-final'
evidence = root/'evidence/source-cohort-20261002'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def retain_files(source,target,suffixes=('.json','.log','.xml','.py','.txt')):
    target.mkdir(parents=True,exist_ok=False)
    for path in sorted(source.iterdir()):
        if source == run and path.name.startswith('retain-release.'):
            continue
        if path.is_file() and path.suffix in suffixes:
            shutil.copyfile(path,target/path.name)

def suite(path, require_pass=True):
    data=ET.parse(path).getroot()
    nodes=[data] if data.tag=='testsuite' else list(data)
    result={k:sum(int(n.attrib.get(k,0)) for n in nodes) for k in ('tests','failures','errors','skipped')}
    result['passed']=result['tests']-result['failures']-result['errors']-result['skipped']
    result['skip_reasons']=sorted({s.attrib.get('message','') for s in data.iter('skipped')})
    if require_pass:assert result['failures']==result['errors']==0,result
    return result

def benchmark(path):
    pipeline=read(path/'pipeline.json');q=read(path/'qualification/qualification.json')
    spec=read(path/'dataset/study.spec.json')
    counts={split:sum(c['split']==split for c in spec['cases']) for split in ('development','calibration','evaluation')}
    assert counts==dict(development=20,calibration=20,evaluation=20)
    assert q['raw_coordinates_changed'] is False and q['estimator_selected_using_evaluation'] is False
    assert q['empirical_admission']['passed'] is False
    assert q['selected_error_model']['coverage']['finite_envelope'] is True
    result=dict(enrollment=q['enrollment'],groups=counts,selected_error_model=q['selected_error_model'],
        raw_baseline=q['raw_baseline'],slices=q['slices'],per_landmark=q['per_landmark'],
        empirical_admission=q['empirical_admission'],exact_retained_replay=pipeline['replay_verified'],
        reference_kind=pipeline['reference_kind'],native_environment=pipeline['environment'],
        measured_coordinates_changed=False,evaluation_used_to_retune=False,
        interpretation='Published-reference agreement in this frozen cohort only; unknown visibility, training overlap and identity independence')
    return result

assert read(release/'validation.json')['completed']
assert read(run/'linux-final-validation.json')['returncode']==0
assert read(run/'python314-cohort.json')['returncode']==0
retain_files(run,evidence/'development')
shutil.copytree(run/'linux-final-native',evidence/'development/linux-final-native')
shutil.copytree(run/'release-check/source-native',evidence/'development/superseded-source-native')
retain_files(run/'release-check',evidence/'development/superseded-validation',('.json','.log','.xml'))
post=evidence/'postseal';retain_files(release,post)
retain_files(release/'wheel-check',post/'wheel-check',('.json','.xml'))
shutil.copytree(release/'wheel-check/native-pipeline',post/'installed-native')
shutil.copytree(release/'archive-native',post/'archive-native')
labs={}
for name,path in [('source',release/'source-lab'),('installed',release/'wheel-check/milestone'),('archive',release/'archive-lab')]:
    data=read(path/'laboratory.json')
    assert data['admitted_pair']['raw_arrays_unchanged'] and data['withheld_query']['observer_calls']==0
    labs[name]=dict(positive=data['positive_control']['admission']['status'],negative=data['negative_control']['admission']['status'],raw_arrays_checked=data['admitted_pair']['raw_arrays_checked'],raw_arrays_unchanged=True,withheld_observer_calls=0)
    shutil.copyfile(path/'laboratory.json',post/(name+'-laboratory.json'))
archive_source=release/'facemath-0.12.0.zip';archive=root/'packages/facemath-0.12.0-cohort-20261002.zip'
for source,target in [(archive_source,archive),(archive_source.with_suffix('.release.json'),archive.with_suffix('.release.json')),(release/'validation.json',archive.with_suffix('.validation.json'))]:
    assert not target.exists();shutil.copyfile(source,target)
result=dict(schema='facemath.source_cohort_validation.v1',version='0.12.0',created_utc=datetime.now(timezone.utc).isoformat(),
    engine_code_sha256=read(release/'environment.log')['engine_code_sha256'],
    historical_review_sha256=sha256((root/'REPOSITORY_REVIEW.json').read_bytes()).hexdigest(),
    parent=read(run/'baseline.json'),source_provenance=read(run/'source-provenance.json'),
    source_conversion_check=read(run/'coordinate-source-check.json'),
    cross_host_observations=read(run/'cross-host-observations.json'),
    suites=dict(windows_source_311=suite(release/'source.xml'),installed_wheel=suite(release/'wheel-check/pytest.xml'),extracted_archive=suite(release/'archive.xml'),linux_311=suite(run/'linux-final.xml'),windows_314_cohort_only=suite(run/'python314.xml')),
    milestone_controls=labs,
    native_runs={name:benchmark(path) for name,path in [('windows_source',release/'source-native'),('windows_installed',release/'wheel-check/native-pipeline'),('windows_archive',release/'archive-native'),('linux_source',run/'linux-final-native')]},
    repeatability={p.name:read(p) for p in run.glob('compare-*.result.json')},
    native_raw_invariance={p.name:read(p) for p in run.glob('audit-*.result.json')},
    preserved_failures=[dict(record='development/real-intake.json',reason='New adapter initially collided with existing cohort analysis module; CLI import failed before intake',resolution='Original module restored byte-for-byte; new source_cohort module and CLI coexistence regression'),
        dict(record='development/linux-local-tests.json',suite=suite(run/'linux.xml',False),reason='Two unsafe extension-header tests exposed newer CPython bypassing frombuf; third code-fingerprint refusal occurred during concurrent source correction',resolution='Validate before _proc_member dispatch; new unchanged-source full suites and native runs'),
        dict(record='development/superseded-validation-stop.json',reason='Stopped old Windows validation after source suite/native cohort passed, before applying archive fix',resolution='Fresh complete source/wheel/archive validation under final fingerprint')],
    package=read(release/'validation.json')['archive'],installed_bytes='postseal/wheel-check/installed-byte-check.json',archive_source_parity='development/archive-source-parity.result.json',
    retained_fitted_model='preseal/native-pipeline/qualification/localization_calibration.json',
    completion_boundary=dict(automated_sourcing=True,new_human_testing_required=False,finite_research_calibration=True,empirical_query_admission=False,person_independence_established=False,training_overlap_known=False,cross_os_exact_replay=False,hosted_execution='scheduler refused parent jobs before any steps; inspect current run separately',macos_native='previous constructor abort unresolved'),
    next_actions=[dict(priority='P1',task='New prospectively frozen domain cohort with independent source lineage and explicit missingness/visibility evidence',references=['docs/SOURCED_COHORT_DESIGN.md','examples/lfpw-cohort.json','src/facemath/observation/source_cohort.py'],acceptance='Source bytes, labels and group inventory frozen before inference; report all outcomes, delivery and width; no retuning on this 20-group evaluation'),
        dict(priority='P1',task='Characterize bound utility using existing fixed error model and a new prespecified study',references=['src/facemath/observation/joint.py','src/facemath/observation/study_io.py','docs/WAVE10_ROADMAP.md'],acceptance='Widths and downstream identifiable geometry alongside coverage, complete denominators, no anatomical expansion without reference support'),
        dict(priority='P1',task='Execute hosted/macOS qualification after scheduler can start jobs',references=['.github/workflows/milestone.yml','scripts/check_native_runtime.py'],acceptance='Actual isolated native stderr/return code and complete tests/pipeline; billing refusal is not test execution')])
with (evidence/'VALIDATION.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(package=result['package'],suites=result['suites'],milestone=labs),indent=2))
