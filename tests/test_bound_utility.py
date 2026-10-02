from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from facemath.observation.contracts import ObservationError
from facemath.observation.utility import case_utility, joint_bound_utility
from test_wave7_joint_model import fitted, rows


def enrollment(data):
    return [dict(case_id=r.case_id, group_id=r.group_id, split=r.split,
                 stratum='all', eligibility='included') for r in data]


@pytest.fixture(scope='module')
def model():
    return fitted()[0]


def test_complete_widths_preserve_model_arrays_and_split_denominators(model):
    data = rows()
    before = deepcopy(model.to_dict())
    raw = [r.predicted_xy.tobytes() for r in data]
    report = joint_bound_utility(data, model, enrollment(data))
    assert report['calibration_sha256'] == model.digest
    assert report['splits']['evaluation']['landmark_diameter']['total'] == 20*8
    assert report['splits']['evaluation']['complete_landmark_groups'] == 20
    assert report['splits']['development']['enrolled_cases'] == 20
    assert report['splits']['calibration']['enrolled_cases'] == 32
    assert all(c['mouth_eye_ratio']['available'] for c in report['cases'])
    assert not report['acceptance_decision'] and not report['raw_coordinates_changed']
    assert [r.predicted_xy.tobytes() for r in data] == raw
    assert model.to_dict() == before


def test_exclusions_and_missing_points_keep_frozen_denominators(model):
    data = rows()
    missing = data[-1].observed.copy(); missing[5] = False
    data = data[:-1]+(replace(data[-1], observed=missing, failure='unobservable_predicted_reference_points'),)
    enrolled = enrollment(data)
    enrolled.append(dict(case_id='excluded', group_id=data[-1].group_id, split='evaluation',
                         stratum='all', eligibility='excluded', exclusion_reason='prespecified'))
    report = joint_bound_utility(data, model, enrolled)
    s = report['splits']['evaluation']
    assert s['enrolled_cases'] == 21 and s['eligible_cases'] == 20
    assert s['enrolled_groups'] == 20 and s['complete_landmark_groups'] == 19
    assert s['landmark_diameter']['total'] == 168 and s['landmark_diameter']['finite'] == 159
    assert s['mouth_eye_ratio']['unavailable'] == 2
    assert report['cases'][-2]['mouth_eye_ratio']['reason'] == 'unobserved_dependency'
    assert report['cases'][-1]['exclusion_reason'] == 'prespecified'


@pytest.mark.parametrize('kind', ['failed', 'missing_eye', 'collapsed_eye'])
def test_unavailable_gauges_are_not_zero_widths(model, kind):
    row = rows()[-1]
    if kind == 'failed':
        row = replace(row, observed=np.zeros(8, bool), failure='timeout', observer_recipe=None)
    elif kind == 'missing_eye':
        support = row.observed.copy(); support[0] = False
        row = replace(row, observed=support)
    else:
        xy = row.predicted_xy.copy(); xy[:4] = xy[0]
        row = replace(row, predicted_xy=xy)
    out = joint_bound_utility((row,), model, enrollment((row,)))
    assert out['splits']['evaluation']['landmark_diameter']['finite'] == 0
    assert out['cases'][0]['mouth_eye_ratio']['width'] is None


def test_uncertain_reference_normalizer_may_vanish(model):
    # Failed calibration attempts produce an unavailable radius in a distinct
    # fixture, without editing the retained model or any production run.
    changed = tuple(replace(r, failure='fixture', observed=np.zeros(8, bool), simultaneous_score=None)
                    if r.split == 'calibration' else r for r in rows())
    unbounded, _ = fitted(changed)
    out = joint_bound_utility(changed, unbounded, enrollment(changed))
    assert out['splits']['evaluation']['mouth_eye_ratio']['finite'] == 0
    assert out['cases'][-1]['reason'] == 'no_finite_radius'
    # A very short error-model eye span can collapse despite finite delivery.
    altered_beta = model.bias.beta.copy()
    predicted = rows()[-1].predicted_xy
    from facemath.observation.bias import ocular_basis
    chart = np.linalg.solve(ocular_basis(predicted), predicted.T).T
    altered_beta[:4] = chart[:4]-chart[:4].mean(axis=0)
    # Build a valid new fitted center with the same displacement of each group.
    vectors = model.bias.group_vectors + (altered_beta-model.bias.beta)
    bias = replace(model.bias, beta=vectors.mean(axis=0), group_vectors=vectors)
    scatter = replace(model.scatter, bias_sha256=bias.digest)
    shifted = replace(model, bias=bias, scatter=scatter)
    value = case_utility(rows()[-1], shifted, 'all')
    assert all(x is not None for x in value['landmark_diameters'])
    assert value['eye_span']['lower'] == 0
    assert not value['mouth_eye_ratio']['available']
    assert value['mouth_eye_ratio']['reason'] == 'denominator_may_vanish'


def test_units_follow_proper_rotation_scale_and_translation(model):
    row = rows()[-1]
    angle = .731; scale = 2.4
    matrix = scale*np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = replace(row, predicted_xy=row.predicted_xy@matrix.T+[123, -42])
    a, b = [case_utility(r, model, 'all') for r in (row, moved)]
    assert b['observed_eye_span_px'] == pytest.approx(a['observed_eye_span_px']*scale)
    assert b['landmark_diameters'] == a['landmark_diameters']
    for name in ('eye_span', 'mouth_width', 'mouth_eye_ratio'):
        assert b[name]['lower'] == pytest.approx(a[name]['lower'], abs=1e-12)
        assert b[name]['upper'] == pytest.approx(a[name]['upper'], abs=1e-12)


def test_shared_translation_does_not_inflate_geometry_like_independent_landmarks(model):
    result = case_utility(rows()[-1], model, 'all')
    # Fixture errors are dominated by common translation; geometry cancels that
    # dependence while marginal landmark projections correctly remain wide.
    assert result['mouth_width']['width'] < min(result['landmark_diameters'])


def test_inventory_or_provenance_mismatch_refused(model):
    data = rows(); enrolled = enrollment(data)
    with pytest.raises(ObservationError, match='enrollment'):
        joint_bound_utility(data[:-1], model, enrolled)
    enrolled[-1]['split'] = 'development'
    with pytest.raises(ObservationError, match='split or group'):
        joint_bound_utility(data, model, enrolled)
    with pytest.raises(ObservationError, match='recipe'):
        case_utility(replace(data[-1], observer_recipe='0'*64), model, 'all')


def test_stratum_without_deployable_radius_has_no_pooled_fallback(model):
    from facemath.observation.stratified import RoutingPolicy, stratify_joint
    from facemath.observation.joint import joint_scored_rows
    data = rows(); enrolled = enrollment(data)
    for case in enrolled:
        case['stratum'] = 'missing' if case['split'] == 'evaluation' else 'all'
    policy = RoutingPolicy(('all', 'missing'), definition='prespecified fixture categories', minimum_calibration_groups=10)
    scored = joint_scored_rows(tuple(r for r in data if r.split != 'development'), model.bias, model.scatter)
    routed = stratify_joint(model, scored, enrolled, policy)
    report = joint_bound_utility(data, routed, enrolled)
    assert report['evaluation_strata']['missing']['enrolled_cases'] == 20
    assert report['evaluation_strata']['missing']['mouth_eye_ratio']['finite'] == 0
    assert report['cases'][-1]['reason'] == 'no_finite_category_radius'
