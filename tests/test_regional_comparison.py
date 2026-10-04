from dataclasses import replace
import json

import numpy as np
import pytest

from facemath.observation import FacialInstrument, InstrumentConfig, ObservationError
from facemath.observation.contracts import Observation, array_digest
from facemath.observation.lab import phantom_points, phantom_observation, render_phantom
from facemath.observation.regional import (
    REGIONAL_NAMES, displacement_components, polar_components, deformation_components,
)


CONFIG = InstrumentConfig(canvas_size=96, eye_span_px=32, uncertainty_draws=4,
                          enable_flow=False, enable_edges=False)


def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def test_displacement_separates_movement_and_shape_without_refitting():
    a = np.array([[0., 0.], [2, 0], [0, 2], [2, 2]])
    deformation = np.array([[-1., 0], [1, 0], [-1, 0], [1, 0]])
    b = a + [3, -4] + deformation
    result = displacement_components(a, b, np.ones(4, bool), 2, 2)
    assert result.shift_b_minus_a == (1.5, -2.)
    assert result.translation_magnitude == 2.5
    assert result.centered_rms == .5
    assert result.total_rms == pytest.approx(np.sqrt(6.5))
    assert result.total_rms**2 == pytest.approx(result.translation_magnitude**2 + result.centered_rms**2)
    assert result.within_tolerance == 0
    assert result.within_tolerance_fraction_bounds == (0., 0.)
    reverse = displacement_components(b, a, np.ones(4, bool), 2, 2)
    assert reverse.shift_b_minus_a == (-1.5, 2.)
    assert reverse.centered_rms == result.centered_rms
    np.testing.assert_array_equal(a, [[0, 0], [2, 0], [0, 2], [2, 2]])


def test_missing_landmarks_cannot_turn_into_perfect_agreement():
    a = np.zeros((4, 2)); b = a.copy(); b[2:] = 1000
    result = displacement_components(a, b, np.array([True, True, False, False]), 1, .05)
    assert result.total == 4 and result.supported == result.within_tolerance == 2
    assert result.within_tolerance_fraction_bounds == (.5, 1.)
    assert result.total_rms == 0
    unavailable = displacement_components(a, b, np.zeros(4, bool), 1, .05)
    assert unavailable.within_tolerance_fraction_bounds == (0., 1.)
    assert unavailable.total_rms is unavailable.shift_b_minus_a is None
    assert unavailable.unavailable_reason


@pytest.mark.parametrize('theta', [-2.7, -.4, 0., .8, 2.9])
def test_finite_rotation_has_no_stretch(theta):
    r = rotation(theta)
    result = polar_components(r[None], np.ones(1, bool))
    np.testing.assert_allclose(result.rotation[0], r, atol=1e-14)
    np.testing.assert_allclose(result.stretch[0], np.eye(2), atol=1e-14)
    np.testing.assert_allclose(result.log_stretch, 0, atol=1e-14)
    assert result.rotation_rad[0] == pytest.approx(theta)
    assert result.supported[0]


@pytest.mark.parametrize('x,y,theta', [(1.2, 1.2, .4), (.7, .7, -1.), (1.3, .8, .2), (1., 1., 0.)])
def test_stretch_area_and_rotation_match_analytic_maps(x, y, theta):
    j = rotation(theta) @ np.diag([x, y])
    result = polar_components(j[None], np.ones(1, bool))
    np.testing.assert_allclose(result.rotation[0], rotation(theta), atol=1e-14)
    np.testing.assert_allclose(result.stretch[0], np.diag([x, y]), atol=1e-14)
    np.testing.assert_allclose(result.log_stretch[0], np.diag(np.log([x, y])), atol=1e-14)
    assert result.log_area[0] == pytest.approx(np.log(x*y), abs=1e-14)
    np.testing.assert_allclose(result.rotation @ result.stretch, j[None], atol=1e-14)
    summary = deformation_components(result, np.zeros(16, int), 20)
    assert summary.support_fraction == .8
    assert summary.area_ratio_mean == pytest.approx(x*y)
    assert summary.rotation_rms_rad == pytest.approx(abs(theta), abs=1e-14)
    assert summary.log_stretch_rms**2 == pytest.approx(
        summary.log_area_rms**2/2 + summary.anisotropic_log_strain_rms**2, abs=1e-14)
    if x == y:
        assert summary.anisotropic_log_strain_rms < 1e-14


def test_simple_shear_is_area_preserving_but_not_rigid():
    gamma = .6
    result = polar_components(np.array([[[1., gamma], [0., 1.]]]), np.ones(1, bool))
    assert result.rotation_rad[0] == pytest.approx(-np.arctan(gamma/2))
    np.testing.assert_allclose(np.linalg.eigvalsh(result.log_stretch[0]), [-np.arcsinh(gamma/2), np.arcsinh(gamma/2)])
    summary = deformation_components(result, np.zeros(8, int), 8)
    assert summary.area_ratio_mean == pytest.approx(1.)
    assert summary.log_area_rms < 1e-14
    assert summary.anisotropic_log_strain_rms == pytest.approx(np.sqrt(2)*np.arcsinh(gamma/2))
    assert summary.chart_shear_rms > 0


def test_polar_components_rotate_with_chart_and_invert_without_changing_shape_energy():
    j = np.array([[1.3, .4], [-.2, .8]])
    q = rotation(.9)
    result = polar_components(np.array([j, q @ j @ q.T, np.linalg.inv(j)]), np.ones(3, bool))
    np.testing.assert_allclose(result.log_stretch[1], q @ result.log_stretch[0] @ q.T, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(result.deviatoric_log_stretch, axis=(1,2)),
                               np.repeat(np.linalg.norm(result.deviatoric_log_stretch[0]), 3), atol=1e-14)
    assert result.log_area[2] == pytest.approx(-result.log_area[0])
    assert result.rotation_rad[2] == pytest.approx(-result.rotation_rad[0])


def test_folds_collapse_and_unsupported_maps_keep_slots_with_masks():
    maps = np.array([np.diag([-1., 1.]), np.diag([0., 1.]), np.eye(2), np.eye(2)])
    result = polar_components(maps, np.array([True, True, False, True]))
    np.testing.assert_array_equal(result.supported, [False, False, False, True])
    np.testing.assert_array_equal(result.rotation[:3], 0)
    assert not result.rotation.flags.writeable
    with pytest.raises(ValueError):
        result.rotation.flags.writeable = True
    with pytest.raises(ObservationError, match='unsupported triangle'):
        deformation_components(result, np.array([0]*8), 8)
    unavailable = deformation_components(result, np.array([], dtype=int), 24)
    assert unavailable.total_pixels == 24 and unavailable.supported_pixels == 0
    assert unavailable.log_area_mean is None and unavailable.unavailable_reason


def test_circular_rotation_cancellation_is_not_reported_as_zero_rotation():
    result = polar_components(np.array([rotation(np.pi/2), rotation(-np.pi/2)]), np.ones(2, bool))
    summary = deformation_components(result, np.array([0, 1]*4), 8)
    assert summary.rotation_mean_rad is None
    assert summary.rotation_mean_unavailable_reason == 'circular_mean_cancels'
    assert summary.rotation_rms_rad == pytest.approx(np.pi/2)
    assert summary.log_stretch_rms < 1e-14


@pytest.mark.parametrize('bad', ['shape', 'nonfinite', 'mask', 'normalizer', 'tolerance'])
def test_displacement_rejects_invalid_inputs(bad):
    a = np.zeros((4, 2)); b = a.copy(); support = np.ones(4, bool); norm = 1.; tolerance = .05
    if bad == 'shape': b = np.zeros((3, 2))
    if bad == 'nonfinite': b[0,0] = np.inf
    if bad == 'mask': support = np.ones(4, int)
    if bad == 'normalizer': norm = 0
    if bad == 'tolerance': tolerance = True
    with pytest.raises(ObservationError):
        displacement_components(a, b, support, norm, tolerance)


def native_schema_control():
    """Synthetic control, with enough unrelated points to retain clipped anchors."""
    from facemath.observation.correspondence import SEMANTIC_21, mediapipe_anchors
    points = np.random.default_rng(12).uniform(35, 220, (478, 2))
    phantom = dict(zip(SEMANTIC_21, phantom_points()))
    for name, index in mediapipe_anchors('mediapipe478.reference8.v2').items():
        points[index] = phantom[name]
    image = np.zeros((256, 256, 3), np.uint8)
    observation = Observation(array_digest(image), image.shape[:2], 'mediapipe478.v1',
        tuple(f'mp_{i:03}' for i in range(478)), points, np.stack([points]*3),
        'synthetic-test-control', '0'*64, .1, 3, ('a','b','c'),
        correspondence='mediapipe478.reference8.v2')
    return image, observation


def test_unavailable_holdout_preserves_primary_native_schema_comparison():
    from facemath.observation.anatomy import anatomy_for
    image, observation = native_schema_control()
    anatomy = anatomy_for(observation)
    removed = sorted({i for name in anatomy.fit_anchor_names if name not in ('left_eye','right_eye','nose_tip')
                      for i in anatomy.anchors[name]})
    xy = observation.xy.copy(); xy[removed] = [-100, -100]
    samples = observation.samples.copy(); samples[:,removed] = [-100,-100]
    partial = replace(observation, xy=xy, samples=samples)
    result = FacialInstrument(CONFIG).run(image, image, partial, partial)
    assert result.state.fit_names == ('left_eye', 'right_eye', 'nose_tip')
    np.testing.assert_array_equal(result.profile.heldout_fit_supported, [False, False, True, True])
    np.testing.assert_array_equal(result.profile.heldout_residuals[:2], 0)
    ledger = {m.name:m for m in result.profile.measurements}
    assert ledger['geometry.all.rms'].value < 1e-12
    withheld = ledger['registration.heldout_eyes.rms']
    assert withheld.value is None and withheld.support > 0
    assert withheld.total == len(np.unique(np.r_[anatomy.region_indices('left_eye'), anatomy.region_indices('right_eye')]))
    assert 'holdout_registration_unavailable' in withheld.exclusion
    assert len(result.evidence()['regional_comparison']['rows']) == 7
    assert 'fewer than three' in result.evidence()['registration_diagnostics']['heldout'][0]['failure']


def test_failed_sensitivity_refit_keeps_requested_slots_and_withholds_intervals(monkeypatch):
    import facemath.observation.measure as module
    original = module._residual
    calls = 0
    def unavailable_once(a, b, state, excluded=None):
        nonlocal calls
        if excluded is None:
            calls += 1
            if calls == 2:
                raise ObservationError('controlled unobservable perturbation')
        return original(a, b, state, excluded)
    monkeypatch.setattr(module, '_residual', unavailable_once)
    p = phantom_points(); image = render_phantom(p); obs = phantom_observation(image, p)
    result = FacialInstrument(CONFIG).run(image, image, obs, obs)
    np.testing.assert_array_equal(result.profile.sensitivity_draw_supported, [True, False, True, True])
    assert result.profile.sensitivity_draws.shape == (4,7)
    np.testing.assert_array_equal(result.profile.sensitivity_draws[1], 0)
    values = [m for m in result.profile.measurements if m.name.startswith('geometry.') and m.name.endswith('.rms')]
    assert all(m.value is not None and m.interval is None for m in values)
    diagnostic = result.evidence()['registration_diagnostics']['sensitivity']
    assert diagnostic['requested_draws'] == 4 and diagnostic['completed_draws'] == 3
    assert not diagnostic['intervals_available'] and diagnostic['failures'][0]['draw'] == 1


def test_production_profile_exports_components_and_exact_existing_rms(tmp_path):
    from facemath import cli
    from facemath.observation.artifacts import write_bundle
    from facemath.observation.readout import inspect_readout, terminal_readout
    p = phantom_points(); q = p.copy(); q[19,0] -= 5; q[20,0] += 5
    a, b = render_phantom(p), render_phantom(q)
    result = FacialInstrument(CONFIG).run(a, b, phantom_observation(a,p), phantom_observation(b,q))
    ledger = {m.name:m for m in result.profile.measurements}
    assert tuple(row.region for row in result.profile.regional.rows) == REGIONAL_NAMES
    for row in result.profile.regional.rows:
        assert row.displacement.total_rms == pytest.approx(ledger[f'geometry.{row.region}.rms'].value)
        assert row.displacement.total_rms**2 == pytest.approx(
            row.displacement.translation_magnitude**2 + row.displacement.centered_rms**2, abs=1e-14)
        if row.region != 'all':
            assert row.deformation.log_stretch_rms == pytest.approx(ledger[f'deformation.{row.region}.log_stretch_rms'].value)
    regional = result.profile.regional.arrays()
    assert len(regional) == 10
    for name, value in regional.items():
        np.testing.assert_array_equal(value, result.raw_geometry_arrays()[name])
    write_bundle(result, tmp_path/'pair', a, b)
    readout = inspect_readout(tmp_path/'pair')
    assert readout['regional_comparison'] == json.loads(json.dumps(result.profile.regional.to_dict()))
    assert 'Regional movement and deformation' in terminal_readout(readout)
    assert 'regional.jaw.anisotropic_log_strain_rms' in (tmp_path/'pair/report.html').read_text()
    cli.main(['replay', str(tmp_path/'pair'), '--out', str(tmp_path/'replayed')])
    with np.load(tmp_path/'replayed/arrays.npz') as arrays:
        for name, expected in regional.items():
            np.testing.assert_array_equal(arrays[name], expected)
