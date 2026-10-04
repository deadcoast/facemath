"""Carry every executed source plan into an explicitly expanded future scope."""
from datetime import datetime, timezone
import json
from pathlib import Path
from facemath.observation.enrollment import (export_enrollment, build_registry, load_registry, check_enrollment)
from facemath.observation.study import load_study, read_json, write_json_new

run = Path(__file__).resolve().parent
root = run.parents[1]
evidence = root/'evidence/enrollment-registry-20261003'
study = load_study(run/'release/source-native/study')
target = evidence/'exports/helen-registry-integration.json'
export_enrollment(study.root, target)
previous = read_json(evidence/'registry.json')
old_rows = load_registry(previous)
receipts = [read_json(p) for p in sorted((evidence/'exports').glob('*.json'))]
scope = ('FACEMATH retained post-Wave10 native research source plans explicitly supplied as complete study exports, '
         'including published-PTS and earlier teacher-model pilot enrollment across splits, exclusions and all '
         'observation outcomes. Membership is the retained receipt inventory; other projects, undeclared person '
         'aliases and native training enrollment are outside this registry.')
# The prospective snapshot explicitly named five plans. Expanding that declared
# scope needs a NEW root, not a disguised change inside an immutable extension.
registry = build_registry(receipts, scope)
rows = load_registry(registry)
old_tokens = {(r['study_sha256'],r['case_id']) for r in old_rows}
new_tokens = {(r['study_sha256'],r['case_id']) for r in rows}
assert old_tokens <= new_tokens and len(rows) == 255 and len(new_tokens-old_tokens) == 12
assert all(r in rows for r in old_rows)
write_json_new(evidence/'next-registry.json', registry)
check = check_enrollment(registry, [(c,study.references(c)) for c in study.cases])
assert not check['disjoint'] and check['overlapping_cases'] == 12
write_json_new(evidence/'next-registry-check.json', check)
record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
    previous_registry_sha256=previous['content_sha256'], next_registry_sha256=registry['content_sha256'],
    scope_expansion='distinct root with all six complete exports; immutable five-study prospective snapshot preserved',
    every_prior_row_retained=True, previous_cases=len(old_rows), registered_cases=len(rows), added_cases=12,
    membership_changed_by_outcome=False, current_study_sha256=study.digest,
    next_action='Use next-registry.json as --prior when registering further studies, and as --registry before acquisition')
write_json_new(evidence/'next-registry-construction.json', record)
print(json.dumps(record,indent=2))
