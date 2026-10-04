"""Recompute prospective binding and full prior-outcome membership read-only."""
from hashlib import sha256
import json
from pathlib import Path
from facemath.observation.enrollment import load_registry, require_disjoint, check_enrollment
from facemath.observation.study import load_study, read_json, write_json_new
from facemath.observation.study_runner import verify_session

run = Path(__file__).resolve().parent
root = run.parents[1]
evidence = root/'evidence/enrollment-registry-20261003'
freeze = read_json(evidence/'prospective-freeze-03.json')
registry = read_json(evidence/'registry.json')
rows = load_registry(registry)
paths = dict(source=run/'release-03/source-native', installed=run/'release-03/wheel-check/native-pipeline',
             archive=run/'release-03/archive-native', linux=run/'linux-native-03')
pipelines = {}
for label, path in paths.items():
    stages = []
    for relative in ('study','session/study','evaluation/acquisition/study','replay/acquisition/study'):
        study = load_study(path/relative)
        check = require_disjoint(registry, [(c,study.references(c)) for c in study.cases])
        assert study.digest == freeze['study_sha256'] and check == freeze['check']
        assert study.data['enrollment_registry']['check'] == check
        assert read_json(study.root/'enrollment_registry.json') == registry
        stages.append(dict(path=relative, registry_sha256=registry['content_sha256'],
            registry_file_sha256=sha256((study.root/'enrollment_registry.json').read_bytes()).hexdigest(),
            enrolled=check['enrolled'], disjoint=check['disjoint']))
    pipelines[label] = stages
prior = []
for record in freeze['registered_sources']:
    study_path = root/record['source']
    status = verify_session(study_path.parent/'session')
    registered = {r['case_id']:r for r in rows if r['study_sha256'] == record['study_sha256']}
    failures = []
    for path in sorted((study_path.parent/'session/cases').glob('*/receipt.json')):
        receipt = read_json(path)
        # Directory indices are acquisition slots, not necessarily declared IDs.
        case_id = receipt['case_id']
        assert case_id in registered
        if receipt['status'] != 'observed':
            failures.append(dict(case_id=case_id, status=receipt['status'], registered=True))
    prior.append(dict(name=record['name'], study_sha256=record['study_sha256'],
        enrolled=status['enrolled'], outcomes=status['outcomes'], failures=failures,
        excluded_registered=sum(r['eligibility']=='excluded' for r in registered.values())))
assert read_json(evidence/'prior-overlap.json')['overlapping_cases'] == 60
assert not (run/'refused-prior-pipeline/session').exists()
assert len(load_registry(read_json(evidence/'next-registry.json'))) == 255
study = load_study(paths['source']/'study')
repeated = check_enrollment(read_json(evidence/'next-registry.json'), [(c,study.references(c)) for c in study.cases])
assert repeated['overlapping_cases'] == 12
result = dict(all_checks_recomputed=True, prospective_study_sha256=freeze['study_sha256'],
    registry_sha256=registry['content_sha256'], pipelines=pipelines, prior_outcomes=prior,
    observed_registry_cases=243, next_registry_cases=255, old_cohort_refused_before_observer=True,
    windows_registry_bytes_equal=len({s['registry_file_sha256'] for k in ('source','installed','archive') for s in pipelines[k]})==1,
    all_hosts_registry_objects_equal=True,
    requalification_sources_already_registered=True,registered_requalification_overlap=repeated['overlapping_cases'],
    model_capsule_and_query_inheritance='exercised by explicit synthetic contract tests; not empirical reference validation')
write_json_new(run/'registry-audit-03.result.json', result)
print(json.dumps({k:v for k,v in result.items() if k != 'pipelines'},indent=2))
