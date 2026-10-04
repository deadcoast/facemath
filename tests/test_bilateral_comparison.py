import json

import numpy as np
import pytest

from facemath.observation.bilateral import measure_bilateral, nominal_components, expressions, norm_box
from facemath.observation.intrinsic_sets import EllipticSource, SourceRelations
from facemath.observation.readout import pair_readout, render_bilateral, terminal_readout
from facemath.observation.artifacts import write_bundle, verify_bundle
from test_wave6_intrinsic import configuration, run_pair
from test_wave6_sets import draws, source


POINTS = np.array([[-3., 0.], [-1., 0.], [1., 0.], [3., 0.], [1., 2.], [-2., 4.], [2., 4.], [0., 8.]])


def profile(a=POINTS, b=POINTS, *, support=None, quantiles=(0., 0.), calibration='a'*64):
    support = np.ones((2, 8), bool) if support is None else support
    evaluators = tuple(SourceRelations(EllipticSource(p, np.zeros((8, 2, 2)), support[s], quantiles[s] or 0., 64))
                       for s, p in enumerate((a, b)))
    return measure_bilateral(np.stack((a, b)), support, evaluators, quantiles, calibration,
                             configuration(), np.tile(np.eye(3), (2, 1, 1)))


def test_exact_components_and_direction_reversal_not_hidden_by_equal_magnitude():
    b = POINTS.copy(); b[4, 0] = -1; b[6] += [2, 1]
    data = profile(b=b).to_dict()
    mouth, nose = data['rows'][2:4]
    assert mouth['observed_a'] == [0, 0]
    assert mouth['observed_b'] == [.5, .25]
    assert mouth['observed_configuration_change'] == pytest.approx(np.sqrt(5)/4)
    assert nose['observed_magnitude_delta_b_minus_a'] == 0
    assert nose['observed_delta_b_minus_a'] == [-.5, 0]
    assert nose['observed_configuration_change'] == .5
    assert nose['observed_status'] == 'outside_tolerance'
    np.testing.assert_allclose(data['rows'][2]['display_segments'][1], [[2, 4], [4, 5]])
    np.testing.assert_allclose(nose['display_segments'][1], [[0, 2], [-1, 2]])
    for row in data['rows']:
        for component, bounds in zip(row['observed_delta_b_minus_a'], row['reference_delta_b_minus_a']):
            assert bounds[0] <= component <= bounds[1]
            assert bounds[1] - bounds[0] < 1e-9
        lo, hi = row['reference_configuration_change']
        assert lo <= row['observed_configuration_change'] <= hi


def test_unchanged_asymmetric_pair_has_zero_change_and_nonzero_source_magnitude():
    nose = profile().to_dict()['rows'][3]
    assert nose['observed_magnitude_a'] == nose['observed_magnitude_b'] == .25
    assert nose['observed_configuration_change'] == 0


@pytest.mark.parametrize('angle,scale', [(.6, .3), (-.9, 3.)])
def test_proper_similarity_invariance_and_reflection_sign(angle, scale):
    points = POINTS.copy(); points[6, 1] += 1
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = points @ rotation.T * scale + [12, -7]
    pair = profile(a=points, b=transformed)
    np.testing.assert_allclose(pair.nominal[0], pair.nominal[1], atol=1e-14)
    reflected = profile(a=points, b=points * [-1, 1])
    np.testing.assert_allclose(reflected.nominal[1], reflected.nominal[0] * [1, -1], atol=1e-14)


def test_swap_negates_components_but_preserves_change():
    b = POINTS.copy(); b[6] += [-.5, .25]
    for ab, ba in zip(profile(b=b).to_dict()['rows'], profile(a=b).to_dict()['rows']):
        np.testing.assert_allclose(ab['observed_delta_b_minus_a'], -np.array(ba['observed_delta_b_minus_a']))
        assert ab['observed_configuration_change'] == ba['observed_configuration_change']
        np.testing.assert_allclose(ab['reference_configuration_change'], ba['reference_configuration_change'])


def test_missing_point_keeps_five_rows_and_only_withholds_affected_feature():
    support = np.ones((2, 8), bool); support[1, 6] = False
    pair = profile(support=support); data = pair.to_dict()
    assert data['prespecified_rows'] == sum(data['observed_counts'].values()) == 5
    assert data['observed_counts']['unavailable'] == 1
    assert data['reference_counts']['unavailable'] == 1
    assert data['rows'][2]['observed_b'] is None
    assert data['rows'][2]['display_segments'][1] is None
    assert data['rows'][2]['source_intervals'][1]['reason'] == 'unobserved_dependency'
    np.testing.assert_array_equal(pair.nominal[1, 2], [0, 0])
    assert 'B: measurement unavailable' in render_bilateral(data)


def test_per_source_radius_and_uncalibrated_observations_are_separate():
    pair = profile(quantiles=(0., None))
    assert pair.source_available[0].all() and not pair.source_available[1].any()
    assert pair.nominal_valid.all()
    assert all(r['reference_configuration_change'] is None for r in pair.to_dict()['rows'])
    bare = profile(calibration=None)
    assert not bare.source_available.any()
    np.testing.assert_array_equal(pair.nominal, bare.nominal)


def test_shared_translation_error_cancels_without_independent_landmark_noise():
    translations = np.tile(np.eye(2), (8, 1)) * .4
    graph = SourceRelations(EllipticSource(POINTS, np.tile(np.eye(2)*.16, (8, 1, 1)),
                                          np.ones(8, bool), 2., 128, translations @ translations.T))
    pair = measure_bilateral(np.stack((POINTS, POINTS)), np.ones((2, 8), bool), (graph, graph),
                             (2., 2.), 'a'*64, configuration(), np.tile(np.eye(3), (2, 1, 1)))
    assert np.max(pair.source_upper - pair.source_lower) < 1e-6
    assert np.all(pair.source_lower <= pair.nominal) and np.all(pair.nominal <= pair.source_upper)


def test_collapsed_eye_denominator_withholds_bound_not_observed_value():
    graph = SourceRelations(source(q=1000))
    pair = measure_bilateral(np.stack((POINTS, POINTS)), np.ones((2, 8), bool), (graph, graph),
                             (1000., 1000.), 'a'*64, configuration(), np.tile(np.eye(3), (2, 1, 1)))
    assert not pair.source_available.any() and pair.nominal_valid.all()
    assert set(pair.source_reasons[0]) == {'denominator_may_vanish'}


def test_outer_rectangles_survive_boundary_samples_without_defining_bounds():
    error_source = source(23, magnitude=.035)
    evaluator = SourceRelations(error_source)
    samples = np.array([nominal_components(p, np.ones(8, bool))[0] for p in draws(error_source, count=200, seed=23)])
    for i, coefficients in enumerate(expressions()):
        bounds = [evaluator.projected_ratio(v, perpendicular=c == 1) for c, v in enumerate(coefficients)]
        lo, hi = np.array([b.lower for b in bounds]), np.array([b.upper for b in bounds])
        assert np.all(samples[:, i] >= lo) and np.all(samples[:, i] <= hi)
        norm = norm_box(lo, hi)
        magnitudes = np.linalg.norm(samples[:, i], axis=1)
        assert min(magnitudes) >= norm.lower and max(magnitudes) <= norm.upper


def test_instrument_path_bundle_arrays_and_display_coordinates(tmp_path):
    bare, _ = run_pair(delta=3, q=None)
    calibrated, (a, b, oa, ob) = run_pair(delta=3)
    for key, value in bare.raw_geometry_arrays().items():
        np.testing.assert_array_equal(value, calibrated.raw_geometry_arrays()[key], err_msg=key)
    pair = calibrated.intrinsic.bilateral_comparison
    # Actual right endpoints must coincide with the existing canonical point,
    # not the fitted error-set center or a separately registered point.
    from facemath.observation.reference import mapped_prediction
    from facemath.observation.provider import apply_points
    for s, (observation, transform) in enumerate(((oa, calibrated.state.source_to_canonical_a),
                                                 (ob, calibrated.state.source_to_canonical_b))):
        expected = apply_points(mapped_prediction(observation)[0], transform)[[3, 2, 6, 4, 7]]
        np.testing.assert_allclose(pair.display_segments[s, :, 1], expected, atol=1e-12)
    bundle = write_bundle(calibrated, tmp_path/'pair', a, b)
    assert verify_bundle(bundle)['verified']
    arrays = np.load(bundle/'arrays.npz')
    for key, value in pair.arrays().items():
        np.testing.assert_array_equal(arrays[key], value)
        assert not value.flags.writeable
    data = pair_readout(calibrated.evidence())
    assert data['bilateral_comparison'] == json.loads((bundle/'summary.json').read_text())['bilateral_comparison']
    text = terminal_readout({**data, 'report': str(bundle/'report.html')})
    assert 'Nose midline offset: A=' in text
    html = render_bilateral(data['bilateral_comparison'], image_prefix='runs/example/result/')
    assert html.count('<details data-bilateral=') == 5
    assert html.count('href="runs/example/result/canonical_') == 10
