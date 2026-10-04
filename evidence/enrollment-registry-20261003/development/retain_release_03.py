"""Retain final executed evidence; preserve old records and the sealed archive."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

run = Path(__file__).resolve().parent
root = run.parents[1]
release = run/'release-03'
evidence = root/'evidence/enrollment-registry-20261003'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def retain_files(source, target, existing=False, suffixes=('.json','.log','.xml','.py')):
    target.mkdir(parents=True, exist_ok=existing)
    for path in sorted(source.iterdir()):
        if source == run and (path.name.startswith('retain-release.') or path.name == 'enrollment.draft.py'):
            continue
        if path.is_file() and path.suffix in suffixes:
            dest = target/path.name
            if dest.exists():
                assert dest.read_bytes() == path.read_bytes(), str(dest)
            else:
                shutil.copyfile(path, dest)

def suite(path):
    tree = ET.parse(path).getroot()
    nodes = [tree] if tree.tag == 'testsuite' else list(tree)
    result = {k:sum(int(n.attrib.get(k,0)) for n in nodes) for k in ('tests','failures','errors','skipped')}
    result['passed'] = result['tests'] - result['failures'] - result['errors'] - result['skipped']
    result['skip_reasons'] = sorted({s.attrib.get('message','') for s in tree.iter('skipped')})
    assert result['failures'] == result['errors'] == 0
    return result

def benchmark(path):
    report = read(path/'evaluation/transfer.json')
    return dict(outcomes=report['outcomes'], enrolled=report['enrolled_cases'], eligible=report['eligible_cases'],
        prospectively_excluded=report['prospectively_excluded'], coverage=report['coverage'],
        raw_localization=report['raw_localization'], bound_utility=report['bound_utility']['splits']['evaluation'],
        report_sha256=report['content_sha256'], fit_performed=report['fit_performed'],
        raw_coordinates_changed=report['raw_coordinates_changed'], empirical_admission=report['empirical_admission'],
        exact_retained_replay=read(path/'pipeline.json')['replay_verified'])

validation = read(release/'validation.json')
assert validation['completed'] and read(run/'wsl-launch-03.json')['returncode'] == 0
assert read(run/'cross-host-03.result.json')['expected_refusal']
audit = read(run/'release-audit-03.result.json')
registry_audit = read(run/'registry-audit-03.result.json')
freeze = read(evidence/'prospective-freeze-03.json')
assert audit['engine_code_sha256'] == freeze['engine_code_sha256']
assert registry_audit['all_checks_recomputed']
retain_files(run, evidence/'development', existing=True)
shutil.copytree(run/'linux-native-03', evidence/'postseal/linux-native')
post = evidence/'postseal'
retain_files(release, post, existing=True)
retain_files(release/'wheel-check', post/'wheel-check', suffixes=('.json','.xml'))
shutil.copytree(release/'wheel-check/native-pipeline', post/'installed-native')
shutil.copytree(release/'archive-native', post/'archive-native')
labs = {}
for name,path in [('source',release/'source-lab'), ('installed',release/'wheel-check/milestone'), ('archive',release/'archive-lab')]:
    data = read(path/'laboratory.json')
    assert data['admitted_pair']['raw_arrays_unchanged'] and data['withheld_query']['observer_calls'] == 0
    labs[name] = dict(positive=data['positive_control']['admission']['status'],
        negative=data['negative_control']['admission']['status'], raw_arrays_checked=data['admitted_pair']['raw_arrays_checked'],
        raw_arrays_unchanged=True, withheld_observer_calls=0)
    shutil.copyfile(path/'laboratory.json', post/(name+'-laboratory.json'))
archive_source = release/'facemath-0.16.0.zip'
archive = root/'packages/facemath-0.16.0-enrollment-registry-20261003.zip'
for source,target in [(archive_source,archive), (archive_source.with_suffix('.release.json'),archive.with_suffix('.release.json')),
                      (release/'validation.json',archive.with_suffix('.validation.json'))]:
    assert not target.exists()
    shutil.copyfile(source,target)
assert sha256(archive.read_bytes()).hexdigest() == audit['archive_sha256']
historical_digest = sha256((root/'REPOSITORY_REVIEW.json').read_bytes()).hexdigest()
assert historical_digest == 'ec9fb50eccadfc148f12981d1bcbd763735f96f6a19e125dd8c818a007c2790a'
boundary = dict(empirical_query_admission=False, person_independence_established=False, visibility_known=False,
    anatomical_equivalence_validated=False, native_training_overlap_known=False, repeated_runs_are_independent_samples=False,
    adaptive_cross_study_selection_controlled=False, cross_os_exact_replay=False,
    general_prior_enrollment_registry=True, registry_complete_for_all_research=False,
    registry_limit='explicitly registered full study locks and model inheritance; external scope completeness unproven',
    stratified_model_transfer=False, new_scientific_acceptance_threshold=False,
    hosted_ci='Not invoked this turn; previous billing-blocked attempts are historical evidence only',
    macos_native='Not executed this turn; previous constructor abort remains unresolved')
result = dict(schema='facemath.enrollment_registry_validation.v1', version='0.16.0',
    created_utc=datetime.now(timezone.utc).isoformat(), engine_code_sha256=audit['engine_code_sha256'],
    parent_commit='66a87e5d0fbf1f0aafcb77ad8a853c82229c0d02', parent=read(run/'parent-baseline.json'),
    source_python_modules=len(list((root/'src/facemath').rglob('*.py'))), historical_review_sha256=historical_digest,
    historical_review_scope='Unchanged historical 79-module audit; current continuation scope is separate',
    prospective_plan='prospective-freeze-03.json', selection_window=freeze['selection_window'],
    registry=dict(prospective_sha256=freeze['registry_sha256'], prospective_scope=freeze['registry_scope'],
        registered_studies=5, registered_cases=243, registered_sources=freeze['registered_sources'],
        target_check=freeze['check'], prior_overlap='prior-overlap.json',
        prior_pipeline_refused_before_observation=True, future_registry=read(evidence/'next-registry-construction.json'),
        future_registry_path='next-registry.json', guards=registry_audit),
    frozen_model=dict(capsule_sha256=read(evidence/'helen-model.json')['content_sha256'],
        model_bytes_sha256=audit['model_bytes_sha256'], refit_performed=False, unchanged_in_all_four_runs=True),
    suites=dict(parent=suite(run/'parent.xml'), windows_source=suite(release/'source.xml'),
        installed_wheel=suite(release/'wheel-check/pytest.xml'), extracted_archive=suite(release/'archive.xml'),
        linux_source=suite(run/'linux-03.xml')),
    milestone_controls=labs,
    native_runs={name:benchmark(path) for name,path in [('windows_source',release/'source-native'),
        ('windows_installed',release/'wheel-check/native-pipeline'), ('windows_archive',release/'archive-native'), ('linux_source',run/'linux-native-03')]},
    raw_invariance={name:dict(arrays_checked=p['raw_arrays_unchanged'], unchanged=True,
        model_bytes_unchanged=p['model_bytes_unchanged']) for name,p in audit['pipelines'].items()},
    repeatability={name:dict(report_exact=p['report_exact'], all_outcomes_equal=p['all_outcomes_equal'],
        nominal_and_sample_arrays_exact=all(c.get('same_outcome') and all(a['bytes_equal'] for a in c['arrays'].values()) for c in p['cases']),
        raw_validation_arrays_exact=all(a['bytes_equal'] for a in p['raw_arrays'].values())) for name,p in audit['comparisons'].items()},
    compatibility='development/compatibility-03.result.json', cross_host_negative_control='development/cross-host-03.result.json',
    cross_host_report_comparison='development/linux-comparison-03.result.json',
    package=dict(**validation['archive'], path=archive.relative_to(root).as_posix(), bytes=archive.stat().st_size),
    source_archive_parity=dict(files_checked=len(audit['source_files_exact']), exact=True),
    installed_bytes='postseal/wheel-check/installed-byte-check.json',
    command_records=['development/parent-baseline.json','development/focused-*.json','postseal/commands.json',
        'postseal/wheel-check/command-*.json','development/linux-*.json','development/registry-freeze.json'],
    expected_refusals=['development/reject-prior-check.json','development/reject-prior-pipeline.json',
        'development/old-study-reader.json','development/old-capsule-reader.json','development/cross-host-03.result.json'],
    repaired_defect=dict(description='Explicit null registry or empty registry/prior option could be treated as absent',
        reproduction='superseded-01/null-bypass.result.json',pre_fix_source='superseded-01/source/src/facemath',
        fix_receipts='development/null-fixed-03.result.json',boundary_regression='development/boundary-03.xml',
        all_supplied_file_options_validated=True,source_observations_unchanged='development/post-fix-invariance.result.json'),
    superseded_attempts=dict(first='superseded-01/STATUS.json',
        intermediate='development/intermediate-attempt-02.json',
        interpretation='Only release-03 and linux-native-03 qualify the final frozen fingerprint; earlier outputs retain their actual status'),
    completion_boundary=boundary,
    artifact_timing='Archive seals preseal evidence; completed archive/installed/Linux logs and final summaries are retained alongside it without changing sealed bytes')
with (evidence/'VALIDATION.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2)
previous = read(evidence/'development/parent-continuation.json')
gates = [g for g in previous['open_gates'] if g['id'] != 'NEXT-02']
gates.insert(1, dict(id='NEXT-06', priority='P1', title='Prospective registry stewardship and independent source identities',
    impact='The registry cannot infer omitted studies, undeclared aliases or native training overlap',
    next_action='Use next-registry.json for future freezing and append every new complete source plan before further evaluation; separately establish person-level identity and reference provenance',
    done_criteria='Explicit scope inventory covers prior research, stable source-qualified aliases are independently documented, and new enrollment is frozen before outcomes',
    dependencies=['Independent provenance and an explicit sampling/selection policy; hashes alone do not establish truth']))
handoff = dict(schema='facemath.continuation_review.v1', version='0.16.0', created_utc=result['created_utc'],
    parent_commit=result['parent_commit'], engine_code_sha256=result['engine_code_sha256'],
    review_scope=dict(authority=['AGENTS.md','README.md','docs/ENROLLMENT_REGISTRY_DESIGN.md','docs/WAVE10_ROADMAP.md'],
        inspected=['src/facemath/observation/enrollment.py','src/facemath/observation/study.py',
            'src/facemath/observation/transfer.py','src/facemath/observation/deployment.py',
            'src/facemath/observation/dataset.py','src/facemath/observation/transfer_io.py','src/facemath/cli.py',
            'tests/test_enrollment_registry.py','scripts/validate_current.py','scripts/check_wheel.py'],
        scope='Cross-study source reuse, frozen enrollment propagation and native integration; not a new audit of every historical mathematics module',
        historical_review='REPOSITORY_REVIEW.json', historical_review_unchanged_sha256=historical_digest,
        previous_continuation='evidence/enrollment-registry-20261003/development/parent-continuation.json'),
    completed=[dict(id='ER-01', change='NEXT-02 immutable full-enrollment receipts and scoped registries',
        evidence=['docs/ENROLLMENT_REGISTRY_DESIGN.md','tests/test_enrollment_registry.py'],
        done='Full splits, failures and exclusions; shared source/hash/group/lineage aliases; immutable append; no automatic replacement'),
        dict(id='ER-02', change='Versioned study/model binding and qualification query/context propagation',
        done='Recomputed guards on load/acquisition/replay; inherited registry carried by model capsules; old-reader refusal executed'),
        dict(id='ER-03', change='243-case prospective registry plus 12-source native integration',
        done='Prior 60-source cohort refused before inference; distinct source/wheel/archive/Linux runs; unchanged model; future registry retains all 255 cases'),
        dict(id='ER-04', change='Full source, installed wheel, extracted archive and Ubuntu validation',
        evidence=['evidence/enrollment-registry-20261003/VALIDATION.json'], done='Required controls, replay, source/package bytes and original observation arrays verified')],
    tests=result['suites'], native_integration=result['native_runs']['windows_source'], registry=result['registry'],
    artifacts=dict(validation='evidence/enrollment-registry-20261003/VALIDATION.json', archive=result['package'],
        next_registry='evidence/enrollment-registry-20261003/next-registry.json',
        wheel='runs/enrollment-registry-20261003/release-03/wheels/facemath-0.16.0-py3-none-any.whl'),
    completion_boundary=boundary, open_gates=gates)
assert (root/'CONTINUATION_REVIEW.json').read_bytes() == (evidence/'development/parent-continuation.json').read_bytes()
(root/'CONTINUATION_REVIEW.json').write_text(json.dumps(handoff,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(version=result['version'],suites=result['suites'],package=result['package'],
                     registry_cases=255,empirical_admission=False),indent=2))
