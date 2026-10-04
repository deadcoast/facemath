"""Retain completed evidence and a current root handoff without rewriting history."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

run = Path(__file__).resolve().parent
root = run.parents[1]
release = run/'release'
evidence = root/'evidence/source-window-20261003'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def retain_files(source, target, existing=False, suffixes=('.json','.log','.xml','.py')):
    target.mkdir(parents=True, exist_ok=existing)
    for path in sorted(source.iterdir()):
        if source == run and path.name.startswith('retain-release.'):
            continue
        if path.is_file() and path.suffix in suffixes:
            dest = target/path.name
            if dest.exists():
                assert dest.read_bytes() == path.read_bytes(), str(dest)
            else:
                shutil.copyfile(path, dest)

def suite(path):
    data = ET.parse(path).getroot()
    nodes = [data] if data.tag == 'testsuite' else list(data)
    result = {k:sum(int(n.attrib.get(k,0)) for n in nodes) for k in ('tests','failures','errors','skipped')}
    result['passed'] = result['tests'] - result['failures'] - result['errors'] - result['skipped']
    result['skip_reasons'] = sorted({s.attrib.get('message','') for s in data.iter('skipped')})
    assert result['failures'] == result['errors'] == 0
    return result

def benchmark(path):
    report = read(path/'evaluation/transfer.json')
    return dict(outcomes=report['outcomes'], enrolled=report['enrolled_cases'], eligible=report['eligible_cases'],
        prospectively_excluded=report['prospectively_excluded'],
        coverage={k:v for k,v in report['coverage'].items() if k != 'cases'},
        raw_localization=report['raw_localization'], bound_utility=report['bound_utility']['splits']['evaluation'],
        report_sha256=report['content_sha256'], fit_performed=report['fit_performed'],
        raw_coordinates_changed=report['raw_coordinates_changed'], empirical_admission=report['empirical_admission'],
        exact_retained_replay=read(path/'pipeline.json')['replay_verified'])

validation = read(release/'validation.json')
assert validation['completed'] and read(run/'linux-authorized.json')['returncode'] == 0
assert read(run/'cross-host.result.json')['expected_refusal']
audit = read(run/'release-audit.result.json')
retain_files(run, evidence/'development', existing=True)
shutil.copytree(run/'linux-native', evidence/'development/linux-native')
post = evidence/'postseal'
retain_files(release, post)
retain_files(release/'wheel-check', post/'wheel-check', suffixes=('.json','.xml'))
shutil.copytree(release/'wheel-check/native-pipeline', post/'installed-native')
shutil.copytree(release/'archive-native', post/'archive-native')
labs = {}
for name, path in [('source',release/'source-lab'),('installed',release/'wheel-check/milestone'),('archive',release/'archive-lab')]:
    data = read(path/'laboratory.json')
    assert data['admitted_pair']['raw_arrays_unchanged'] and data['withheld_query']['observer_calls'] == 0
    labs[name] = dict(positive=data['positive_control']['admission']['status'],
        negative=data['negative_control']['admission']['status'],
        raw_arrays_checked=data['admitted_pair']['raw_arrays_checked'], raw_arrays_unchanged=True,
        withheld_observer_calls=0)
    shutil.copyfile(path/'laboratory.json', post/(name+'-laboratory.json'))
archive_source = release/'facemath-0.15.0.zip'
archive = root/'packages/facemath-0.15.0-source-window-20261003.zip'
for source, target in [(archive_source,archive), (archive_source.with_suffix('.release.json'),archive.with_suffix('.release.json')),
                       (release/'validation.json',archive.with_suffix('.validation.json'))]:
    assert not target.exists()
    shutil.copyfile(source,target)
assert sha256(archive.read_bytes()).hexdigest() == audit['archive_sha256']
result = dict(schema='facemath.source_window_validation.v1', version='0.15.0',
    created_utc=datetime.now(timezone.utc).isoformat(), engine_code_sha256=audit['engine_code_sha256'],
    parent_commit='381e15af01427ae78cf01717d9f1c2a7fe8067ed', parent=read(run/'parent-baseline.json'),
    source_python_modules=len(list((root/'src/facemath').rglob('*.py'))),
    historical_review_sha256=sha256((root/'REPOSITORY_REVIEW.json').read_bytes()).hexdigest(),
    historical_review_scope='Unchanged historical 79-module review; the current continuation review is separately scoped',
    prospective_plan='prospective-freeze.json',
    selection_window=read(run/'prospective-freeze.json')['selection_window'],
    source_overlap_checks=read(run/'prospective-freeze.json')['source_overlap_checks'],
    frozen_model=dict(capsule_sha256=read(evidence/'helen-model.json')['content_sha256'],
        model_bytes_sha256=audit['model_bytes_sha256'], refit_performed=False, unchanged_in_all_four_runs=True),
    suites=dict(parent=suite(run/'parent.xml'), windows_source=suite(release/'source.xml'),
        installed_wheel=suite(release/'wheel-check/pytest.xml'), extracted_archive=suite(release/'archive.xml'),
        linux_source=suite(run/'linux.xml')),
    milestone_controls=labs,
    native_runs={name:benchmark(path) for name,path in [('windows_source',release/'source-native'),
        ('windows_installed',release/'wheel-check/native-pipeline'),('windows_archive',release/'archive-native'),
        ('linux_source',run/'linux-native')]},
    raw_invariance={name:dict(arrays_checked=p['raw_arrays_unchanged'], unchanged=True,
        model_bytes_unchanged=p['model_bytes_unchanged']) for name,p in audit['pipelines'].items()},
    repeatability={name:dict(report_exact=p['report_exact'], all_outcomes_equal=p['all_outcomes_equal'],
        nominal_and_sample_arrays_exact=all(c.get('same_outcome') and all(a['bytes_equal'] for a in c['arrays'].values()) for c in p['cases']),
        raw_validation_arrays_exact=all(a['bytes_equal'] for a in p['raw_arrays'].values()))
        for name,p in audit['comparisons'].items()},
    compatibility='development/compatibility.result.json',
    support_diagnostics='development/support-diagnostics.result.json',
    cross_host_negative_control='development/cross-host.result.json',
    cross_host_report_comparison='development/linux-comparison.result.json',
    package=dict(**validation['archive'], path=str(archive.relative_to(root)), bytes=archive.stat().st_size),
    source_archive_parity=dict(files_checked=len(audit['source_files_exact']), exact=True),
    installed_bytes='postseal/wheel-check/installed-byte-check.json',
    command_records=['development/parent-baseline.json','development/focused.json','postseal/commands.json',
        'postseal/wheel-check/command-*.json','development/linux-*.json'],
    preserved_failures=[dict(record='development/linux.json', reason='WSL service denied sandbox access before any Linux code ran',
        actual_error='Wsl/Service/E_ACCESSDENIED', resolution='Distinct approved isolated offline execution in linux-authorized.json')],
    completion_boundary=dict(empirical_query_admission=False, person_independence_established=False,
        visibility_known=False, anatomical_equivalence_validated=False, native_training_overlap_known=False,
        repeated_runs_are_independent_samples=False, adaptive_cross_study_selection_controlled=False,
        cross_os_exact_replay=False, general_prior_enrollment_registry=False, stratified_model_transfer=False,
        hosted_ci='Not invoked this turn; previous billing-blocked attempts are historical evidence only',
        macos_native='Not executed this turn; previous constructor abort remains unresolved'))
with (evidence/'VALIDATION.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,indent=2)
handoff = dict(schema='facemath.continuation_review.v1', version='0.15.0', created_utc=result['created_utc'],
    parent_commit=result['parent_commit'], engine_code_sha256=result['engine_code_sha256'],
    review_scope=dict(authority=['AGENTS.md','README.md','docs/MODEL_TRANSFER_DESIGN.md','docs/WAVE10_ROADMAP.md'],
        inspected=['src/facemath/observation/source_cohort.py','src/facemath/observation/dataset.py',
            'src/facemath/observation/transfer.py','src/facemath/observation/transfer_io.py',
            'tests/test_source_cohort.py','tests/test_model_transfer.py','scripts/validate_current.py','scripts/check_wheel.py'],
        scope='Continuation of source intake and frozen transfer; not a new audit of every historical research module',
        historical_review='REPOSITORY_REVIEW.json', historical_review_unchanged_sha256=result['historical_review_sha256']),
    completed=[dict(id='SW-01', change='Versioned deterministic photograph-root windows after grouping, before split allocation',
        evidence=['docs/SOURCE_WINDOW_DESIGN.md','tests/test_source_cohort.py'],
        done='Integer bounds, exhausted inventory, split/group separation, order invariance and resealed tampering tested; old reader refusal and legacy recipe parity executed'),
        dict(id='SW-02', change='Frozen 60-portrait HELEN holdout with unchanged donor model and no source/group aliases against retained cohorts',
        evidence=['examples/helen-transfer-window.json','evidence/source-window-20261003/prospective-freeze.json'],
        done='Distinct native source, wheel, archive and Linux acquisitions; full enrollment retained; no refit or empirical admission'),
        dict(id='SW-03', change='Full regression, milestone controls, replay and portable package validation',
        evidence=['evidence/source-window-20261003/VALIDATION.json'], done='Source, wheel and extracted archive passed; observed arrays/model bytes audited')],
    tests=result['suites'], native_holdout=result['native_runs']['windows_source'],
    artifacts=dict(validation='evidence/source-window-20261003/VALIDATION.json', archive=result['package'],
        wheel='runs/transfer-holdout-20261003/release/wheels/facemath-0.15.0-py3-none-any.whl'),
    completion_boundary=result['completion_boundary'],
    open_gates=[
        dict(id='NEXT-01', priority='P1', title='Independent reference semantics and visibility',
            evidence=['docs/WAVE10_ROADMAP.md','evidence/source-window-20261003/VALIDATION.json'],
            impact='Published PTS agreement cannot authorize anatomical accuracy or empirical queries',
            next_action='Freeze independently produced reference8 equivalence/visibility labels and acquisition categories before native outcomes',
            done_criteria='Retained reference provenance and all categories/group denominators evaluated under a frozen policy',
            dependencies=['Independent reference evidence; not a detector availability check']),
        dict(id='NEXT-02', priority='P1', title='Cross-study enrollment registry remains open',
            impact='Rank windows separate one pinned source universe; donor guards do not track arbitrary unrelated prior evaluations',
            next_action='Design a pinned registry of prior source/group/lineage enrollment across model and study versions',
            done_criteria='Prospective exclusions cover every registered split, failure and exclusion without result-dependent replacement',
            dependencies=['Explicit registry scope, immutable exports and same-source alias policy']),
        dict(id='NEXT-03', priority='P1', title='New model or domain changes require fresh development and calibration',
            impact='These transfer outcomes are evaluation evidence and cannot tune the existing HELEN fit',
            next_action='Prespecify useful-width/domain criteria and a distinct development/calibration study before a new evaluation',
            done_criteria='New frozen fit family/protocol with complete held-out coverage, missingness and utility evidence',
            dependencies=['New source enrollment; no reuse of this holdout as fresh acceptance']),
        dict(id='NEXT-04', priority='P1', title='Stratified transfer routing and denominators',
            impact='Only pooled full16 transfer is supported', next_action='Specify domain/category routing before implementing stratified capsules',
            done_criteria='Preserve each category fit/radius/protocol and all category denominators without pooled fallback',
            dependencies=['Prospective routing contract and category enrollment']),
        dict(id='NEXT-05', priority='P1', title='macOS and hosted execution remain open',
            impact='Local Windows/Linux evidence does not establish hosted or macOS native operation',
            next_action='Diagnose retained macOS native constructor abort and run the configured hosted matrix when infrastructure permits',
            done_criteria='Actual native process receipts and full platform regressions', dependencies=['macOS runtime and functioning hosted scheduler'])])
with (root/'CONTINUATION_REVIEW.json').open('x',encoding='utf-8') as f:
    json.dump(handoff,f,indent=2)
print(json.dumps(dict(package=result['package'],suites=result['suites'],milestones=labs,
    native={k:{'outcomes':v['outcomes'],'covered_groups':v['coverage']['covered_groups']} for k,v in result['native_runs'].items()}),indent=2))
