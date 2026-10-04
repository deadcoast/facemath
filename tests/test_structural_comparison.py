from dataclasses import replace
import json

import numpy as np
import pytest

from facemath.observation.artifacts import write_bundle, verify_bundle
from facemath.observation.intrinsic_sets import EYE_VECTOR, EllipticSource, SourceRelations
from facemath.observation.readout import pair_readout, render_pair_readout, terminal_readout
from facemath.observation.structure import measure_structure, observed_value, reference_bound, structural_registry
from test_wave6_intrinsic import configuration, run_pair
from test_wave6_sets import draws, source


POINTS = np.array([[-3., 0.], [-1., 0.], [1., 0.], [3., 0.], [1., 2.], [-2., 4.], [2., 4.], [0., 8.]])
EXPECTED = np.array([.5, .5, .5, 1., .5, 1., 2., .5, 1., .25, 0., 0., 0., 0.])


def profile(a=POINTS, b=POINTS, *, supported=None, quantiles=(0., 0.)):
    support = np.ones((2, 8), bool) if supported is None else supported
    evaluators = tuple(SourceRelations(EllipticSource(p, np.zeros((8, 2, 2)), support[s], q or 0., 64))
                       for s, (p, q) in enumerate(zip((a, b), quantiles)))
    return measure_structure(np.stack((a, b)), support, evaluators, quantiles, 'a'*64, configuration())


def test_hand_computed_observed_measurements_and_signed_difference():
    b = POINTS.copy(); b[4] += [1, 1]; b[6, 1] += 2
    p = profile(b=b)
    np.testing.assert_allclose(p.nominal[0], EXPECTED, atol=1e-14)
    rows = {r['key']: r for r in p.to_dict()['rows']}
    assert rows['eye_to_nose_height']['observed_delta_b_minus_a'] == .25
    assert rows['nose_lateral_offset']['observed_delta_b_minus_a'] == .25
    assert rows['mouth_corner_height_difference']['observed_b'] == .5
    assert rows['mouth_width']['observed_b'] == pytest.approx(np.sqrt(20)/4)
    assert p.to_dict()['prespecified_measurements'] == 14
    assert sum(p.to_dict()['observed_counts'].values()) == 14
    for row in rows.values():
        lo, hi = row['reference_delta_b_minus_a']
        assert lo <= row['observed_delta_b_minus_a'] <= hi
        assert hi-lo < 1e-9


@pytest.mark.parametrize('angle,scale', [(.43, .7), (-1.2, 4.)])
def test_invariant_under_independent_proper_similarity_and_sensitive_to_reflection(angle, scale):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = POINTS @ rotation.T * scale + [31, -17]
    np.testing.assert_allclose(profile(b=transformed).nominal[1], EXPECTED, atol=1e-13)
    reflected = profile(b=POINTS*[-1, 1])
    for i, spec in enumerate(reflected.measurements):
        assert reflected.nominal[1, i] == pytest.approx(EXPECTED[i]*(-1 if spec.operation == 'height' else 1))


def test_swap_negates_signed_delta_and_preserves_gap():
    b = POINTS.copy(); b[5, 0] -= 1; b[7] += [.4, .2]
    ab, ba = profile(b=b), profile(a=b)
    np.testing.assert_allclose(ab.delta_lower, -ba.delta_upper)
    np.testing.assert_allclose(ab.delta_upper, -ba.delta_lower)
    np.testing.assert_array_equal(ab.gap_lower, ba.gap_lower)
    np.testing.assert_array_equal(ab.gap_upper, ba.gap_upper)


def test_missing_chin_retains_other_rows_and_frozen_denominator():
    support = np.ones((2, 8), bool); support[1, 7] = False
    p = profile(supported=support); data = p.to_dict()
    missing = [r['key'] for r in data['rows'] if r['observed_b'] is None]
    assert missing == ['eye_to_chin_height', 'mouth_to_chin_height', 'chin_lateral_offset']
    assert data['observed_counts']['unavailable'] == 3
    assert data['reference_counts']['unavailable'] == 3
    assert sum(data['reference_counts'].values()) == 14
    assert np.all(p.source_available[0])


def test_source_radius_missingness_is_not_pooled_or_given_zero_radius_bounds():
    p = profile(quantiles=(0., None))
    assert np.all(p.source_available[0]) and not np.any(p.source_available[1])
    assert np.all(p.nominal_valid) and not np.any(p.pair_available)
    assert set(p.source_reasons[1]) == {'unbounded_localization_calibration'}


def test_normalizer_collapse_and_algebraic_cancellation():
    e = SourceRelations(source(q=1000))
    assert not e.projected_ratio(EYE_VECTOR).available
    assert e.projected_ratio(EYE_VECTOR).reason == 'denominator_may_vanish'
    e = SourceRelations(source())
    assert e.projected_ratio(EYE_VECTOR).lower == e.projected_ratio(EYE_VECTOR).upper == 1.
    assert e.projected_ratio(-EYE_VECTOR).lower == e.projected_ratio(-EYE_VECTOR).upper == -1.
    assert e.projected_ratio(EYE_VECTOR, perpendicular=True).upper == 0.
    assert e.projected_ratio(np.zeros(8)).upper == 0.


def test_common_translation_covariance_cancels_in_shared_graph():
    translation = np.tile(np.eye(2), (8, 1))*.4
    joint = translation @ translation.T
    e = SourceRelations(EllipticSource(POINTS, np.tile(np.eye(2)*.16, (8, 1, 1)), np.ones(8, bool), 2., 128, joint))
    for expected, spec in zip(EXPECTED, structural_registry(configuration())):
        bound = reference_bound(e, spec)
        assert bound.lower <= expected <= bound.upper
        assert bound.upper-bound.lower < 1e-6


@pytest.mark.parametrize('seed', [12, 19])
def test_analytic_outer_bounds_survive_boundary_sample_falsification(seed):
    s = source(seed, magnitude=.035)
    e = SourceRelations(s)
    points = draws(s, count=200, seed=seed)
    for spec in structural_registry(configuration()):
        bound = reference_bound(e, spec)
        values = [observed_value(p, np.ones(8, bool), spec)[0] for p in points]
        assert bound.available and min(values) >= bound.lower-1e-12 and max(values) <= bound.upper+1e-12


def test_instrument_export_readout_and_raw_state_preservation(tmp_path):
    bare, _ = run_pair(delta=3, q=None)
    calibrated, (a, b, oa, ob) = run_pair(delta=3)
    pb, pc = bare.intrinsic.structural_comparison, calibrated.intrinsic.structural_comparison
    np.testing.assert_array_equal(pb.nominal, pc.nominal)
    np.testing.assert_array_equal(pb.nominal_valid, pc.nominal_valid)
    np.testing.assert_array_equal(bare.state.observation_a.xy, oa.xy)
    for key, value in bare.raw_geometry_arrays().items():
        np.testing.assert_array_equal(value, calibrated.raw_geometry_arrays()[key], err_msg=key)
    assert set(k for k in bare.raw_geometry_arrays() if k.startswith('structure_')) == {
        'structure_nominal', 'structure_nominal_valid', 'structure_expression_a', 'structure_expression_b'}
    assert not np.any(pb.source_available) and np.all(pc.source_available)
    bundle = write_bundle(calibrated, tmp_path/'pair', a, b)
    assert verify_bundle(bundle)['verified']
    data = pair_readout(calibrated.evidence())
    assert json.loads((bundle/'summary.json').read_text())['structural_comparison'] == data['structural_comparison']
    exported = np.load(bundle/'arrays.npz')
    for key, value in pc.arrays().items():
        np.testing.assert_array_equal(exported[key], value)
        assert not value.flags.writeable
    html = render_pair_readout(calibrated.evidence())
    assert html.count('<th scope="row">') == 14
    assert 'B − A' in html and 'Viewer-left eye width' in html
    text = terminal_readout({**data, 'report': str(bundle/'report.html')})
    assert 'Mouth to chin height: A=' in text and 'reference delta' in text


def test_uncalibrated_and_partial_html_does_not_print_zero_for_missing_values():
    support = np.ones((2, 8), bool); support[1, 7] = False
    p = profile(supported=support, quantiles=(None, None))
    from facemath.observation.readout import render_structure
    html = render_structure(p.to_dict())
    assert html.count('<th scope="row">') == 14 and '3 unavailable' in html
    assert 'unbounded localization calibration' in html and '>Unavailable<' in html


def test_cli_returns_structural_values_from_the_production_pair_path(tmp_path):
    from facemath.observation.study import save_rgb
    _, (a, b, oa, ob) = run_pair(delta=2, q=None)
    for key, image, observation in [('a', a, oa), ('b', b, ob)]:
        save_rgb(tmp_path/f'{key}.png', image)
        (tmp_path/f'{key}.json').write_text(json.dumps(observation.to_dict()))
    import subprocess, sys
    result = subprocess.run([sys.executable, '-m', 'facemath', 'analyze', str(tmp_path/'a.png'), str(tmp_path/'b.png'),
        '--observation-a', str(tmp_path/'a.json'), '--observation-b', str(tmp_path/'b.json'),
        '--out', str(tmp_path/'cli'), '--no-flow', '--no-edges', '--draws', '4'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert len(data['comparison']['structural_comparison']['rows']) == 14
