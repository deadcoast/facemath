"""Audit actual native correspondence, full enrollment and raw-array preservation.

The old/new nose comparison uses the two prespecified vertices from the same
native observation. It fits nothing and never selects a replacement on outcomes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from facemath.observation.contracts import Observation, array_digest
from facemath.observation.correspondence import MP_REFERENCE8_V2
from facemath.observation.dataset import verify_pipeline
from facemath.observation.reference import REFERENCE_NAMES, mapped_prediction
from facemath.observation.study import read_json, write_json_new


def audit(root: Path) -> dict:
    verification = verify_pipeline(root)
    study = read_json(root/'study/study.lock.json')
    raw = read_json(root/'qualification/raw_validation/validation.json')
    qualification = read_json(root/'qualification/qualification.json')
    assert qualification['live_human_observer_validation'] is False
    assert qualification['reference_evidence']['reference_kinds'] == ['external_dataset']
    assert qualification['reference_evidence']['native_observation_executed'] is True
    rows = {r['case_id']: r for r in raw['cases']}
    assert study['observer']['correspondence'] == MP_REFERENCE8_V2
    assert len(rows) == sum(c['eligibility'] == 'included' for c in study['cases'])
    cases = []
    with np.load(root/'qualification/arrays.npz', allow_pickle=False) as arrays:
        order = qualification['array_order']['cases']
        for i, case in enumerate(study['cases']):
            folder = f'case_{i:05d}'
            result = {k: case[k] for k in ('case_id','group_id','split','eligibility')}
            if case['eligibility'] != 'included':
                cases.append(dict(result, status='prospectively_excluded', raw_arrays_checked=0,
                                  exclusion_reason=case['exclusion_reason']))
                continue
            receipt = read_json(root/'session/cases'/folder/'receipt.json')
            result.update(status=receipt['status'], raw_arrays_checked=0)
            if receipt['status'] == 'observed':
                paths = [root/base/folder/'observation.json' for base in
                         ('session/cases','qualification/acquisition/cases','replay/acquisition/cases')]
                data = [p.read_bytes() for p in paths]
                assert data[0] == data[1] == data[2]
                obs = Observation.from_dict(json.loads(data[0]))
                assert obs.correspondence == MP_REFERENCE8_V2 and obs.xy.shape == (478,2)
                row = rows[case['case_id']]
                predicted, covariance, support = mapped_prediction(obs)
                np.testing.assert_array_equal(predicted, row['predicted_xy'])
                np.testing.assert_array_equal(covariance, row['covariance'])
                np.testing.assert_array_equal(arrays['source_predicted_xy'][order.index(case['case_id'])], predicted)
                span = row['normalization_span_px']
                normalized = np.array(row['error_norm_px']) / span
                old_nose = float(np.linalg.norm(obs.xy[1] - np.array(row['reference_xy'])[4])/span)
                result.update(raw_arrays_checked=2, observation_sha256=sha256(data[0]).hexdigest(),
                    xy_sha256=array_digest(obs.xy), samples_sha256=array_digest(obs.samples),
                    observed_reference_points=row['observed'], error_eye_spans=normalized.tolist(),
                    legacy_nose_error_eye_spans=old_nose,
                    native_vertices=478, successful_samples=len(obs.samples))
            cases.append(result)
    splits = {}
    for split in ('development','calibration','evaluation'):
        enrolled = [c for c in cases if c['split'] == split]
        observed = [c for c in enrolled if c['status'] == 'observed']
        landmarks = []
        for i, name in enumerate(REFERENCE_NAMES):
            errors = [c['error_eye_spans'][i] for c in observed if c['observed_reference_points'][i]]
            landmarks.append(dict(name=name, enrolled=len(enrolled), observed=len(errors),
                within_005=sum(e <= .05 for e in errors),
                median_error_eye_spans=float(np.median(errors)) if errors else None))
        old = [c['legacy_nose_error_eye_spans'] for c in observed if c['observed_reference_points'][4]]
        splits[split] = dict(enrolled=len(enrolled), observed=len(observed), landmarks=landmarks,
            legacy_nose=dict(within_005=sum(e <= .05 for e in old),
                median_error_eye_spans=float(np.median(old)) if old else None))
    return dict(verified=verification['verified'], pipeline=str(root),
        study_sha256=qualification['study_sha256'], correspondence=MP_REFERENCE8_V2,
        raw_arrays_checked=sum(c['raw_arrays_checked'] for c in cases),
        raw_observation_bytes_unchanged=True, source_predictions_are_selected_native_vertices=True,
        cases=cases, splits=splits, selected_error_model=qualification['selected_error_model'],
        empirical_admission=False,
        comparison_scope='MP1 versus prespecified MP4 on identical native outputs; no coordinate correction or refit for comparison',
        limits=['published PTS agreement, not independent anatomical validation',
                'chin correspondence remains unresolved',
                'complete enrollment includes failures and exclusions; medians condition on observed points'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.pipeline.resolve())
    if args.out is not None:
        write_json_new(args.out, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}, indent=2))
