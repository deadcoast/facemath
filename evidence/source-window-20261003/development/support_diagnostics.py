"""Retain exact support failures from the completed source run without changing it."""
from hashlib import sha256
import json
from pathlib import Path

run = Path(__file__).resolve().parent
base = run/'release/source-native'
report_path = base/'evaluation/raw_validation/validation.json'
report = json.loads(report_path.read_text())
study = json.loads((base/'session/study/study.lock.json').read_text())
cases = {c['case_id']: c for c in study['cases']}
rows = []
for row in report['cases']:
    if row['failure']:
        case = cases[row['case_id']]
        assert all(row['reference_visible'])
        assert row['observed'] == [True]*7 + [False]
        h, w = case['image_shape']
        predicted = row['predicted_xy'][-1]
        assert predicted[1] > h-1
        rows.append(dict(case_id=row['case_id'], group_id=case['group_id'],
            image_shape_hw=[h,w], landmark='chin', predicted_xy_px=predicted,
            reference_xy_px=row['reference_xy'][-1], stored_raster_max_y_px=h-1,
            below_image_by_px=predicted[1]-(h-1), failure=row['failure'],
            reference_annotation_present=True, native_observation_present=True))
assert len(rows) == 4
record = dict(raw_report_sha256=sha256(report_path.read_bytes()).hexdigest(),
    evidence_scope='Read-only diagnosis of this completed frozen source run; not a new fit or threshold',
    rows=rows, denominator=60, observed_landmarks=476, enrolled_landmarks=480,
    raw_coordinates_changed=False, replacement_or_retry_performed=False,
    inference='These four incomplete-case failures arise from out-of-image predicted chins, not absent reference labels')
with (run/'support-diagnostics.result.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
