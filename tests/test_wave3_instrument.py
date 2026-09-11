from dataclasses import replace

import numpy as np
import pytest

from facemath.observation import FacialInstrument, InstrumentConfig, ObservationError
from facemath.observation.lab import phantom_observation, phantom_points, render_phantom
from facemath.observation.provider import apply_points

CONFIG = InstrumentConfig(
    canvas_size=128, eye_span_px=40, uncertainty_draws=8, enable_flow=False
)


def run(p, q, *, image_a=None, image_b=None, config=CONFIG, sigma=0.1):
    a = render_phantom(p) if image_a is None else image_a
    b = render_phantom(q) if image_b is None else image_b
    return FacialInstrument(config).run(
        a, b, phantom_observation(a, p, sigma), phantom_observation(b, q, sigma)
    )


def metrics(result):
    return {m.name: m for m in result.profile.measurements}


@pytest.mark.parametrize(
    "theta,scale,shift",
    [(-0.1, 0.9, [5, 3]), (0.05, 1.1, [-2, -3]), (0, 1, [7, -5]), (0, 0.8, [10, 20])],
)
def test_entire_instrument_is_similarity_invariant(theta, scale, shift):
    p = phantom_points()
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    matrix[:2, 2] = shift
    result = run(p, apply_points(p, matrix))
    assert metrics(result)["geometry.all.rms"].value < 1e-12
    assert metrics(result)["geometry.all.within_tolerance"].value == 1


@pytest.mark.parametrize("seed", range(6))
def test_landmark_metrics_are_input_exchange_symmetric(seed):
    rng = np.random.default_rng(seed)
    p = phantom_points()
    q = p + rng.normal(0, 2, p.shape)
    ab, ba = metrics(run(p, q)), metrics(run(q, p))
    names = [n for n in ab if n.startswith(("geometry.", "contour.", "registration."))]
    for name in names:
        np.testing.assert_allclose(
            ab[name].value, ba[name].value, atol=1e-12, err_msg=name
        )


def test_jaw_localization_and_monotone_response():
    p = phantom_points()
    jaw = []
    for amount in (0, 2, 4, 8):
        q = p.copy()
        q[19, 0] -= amount
        q[20, 0] += amount
        m = metrics(run(p, q))
        jaw.append(m["geometry.jaw.rms"].value)
        if amount:
            assert m["geometry.jaw.rms"].value > 5 * m["geometry.central.rms"].value
            assert (
                m["registration.heldout_jaw.rms"].value >= m["geometry.jaw.rms"].value
            )
    assert np.all(np.diff(jaw) > 0)


def test_texture_cannot_change_landmark_measurements():
    p = phantom_points()
    q = p.copy()
    q[11, 1] += 4
    rng = np.random.default_rng(42)
    random_image = rng.integers(0, 256, render_phantom(p).shape, dtype=np.uint8)
    ordinary = metrics(run(p, q))
    changed = metrics(run(p, q, image_b=random_image))
    assert ordinary["geometry.all.rms"].value == changed["geometry.all.rms"].value


def test_enabling_flow_does_not_change_anatomical_score():
    p = phantom_points()
    q = p.copy()
    q[19, 0] -= 8
    noflow = run(p, q)
    flow = run(p, q, config=replace(CONFIG, enable_flow=True))
    for name, m in metrics(noflow).items():
        assert m.value == metrics(flow)[name].value
    assert flow.flow is not None


def test_flat_image_flow_has_no_evidential_support():
    p = phantom_points()
    q = p.copy()
    q[11, 1] += 8
    blank = np.zeros(render_phantom(p).shape, np.uint8)
    result = run(
        p, q, image_a=blank, image_b=blank, config=replace(CONFIG, enable_flow=True)
    )
    assert not result.flow.admissible.any()
    assert metrics(result)["geometry.nose.rms"].value > 0
    assert metrics(result)["flow.nose.anatomical_disagreement"].value is None


def test_noise_floor_changes_sensitivity_not_point_estimate():
    p = phantom_points()
    low = metrics(run(p, p, sigma=0.1))["geometry.all.rms"]
    high = metrics(run(p, p, sigma=2.0))["geometry.all.rms"]
    assert low.value == high.value == 0
    assert high.interval[1] > 5 * low.interval[1]
    assert "not_accuracy_CI" in high.interval_kind


def test_seed_reproduces_raw_sensitivity_draws():
    p = phantom_points()
    q = p.copy()
    q[18, 1] += 4
    first, second = run(p, q), run(p, q)
    np.testing.assert_array_equal(
        first.profile.sensitivity_draws, second.profile.sensitivity_draws
    )


def test_geometry_state_transforms_roundtrip():
    p = phantom_points()
    q = p.copy()
    q[19, 0] -= 4
    state = run(p, q).state
    for m, points in (
        (state.source_to_canonical_a, p),
        (state.source_to_canonical_b, q),
    ):
        np.testing.assert_allclose(
            apply_points(apply_points(points, m), np.linalg.inv(m)), points, atol=1e-12
        )


def test_no_synthetic_fallback_without_observation():
    image = render_phantom(phantom_points())
    with pytest.raises(ObservationError, match="supply two"):
        FacialInstrument(CONFIG).run(image, image)


def test_mismatched_order_fails_explicitly_not_silently_mispairs():
    p = phantom_points()
    image = render_phantom(p)
    obs = phantom_observation(image, p)
    reversed_obs = replace(
        obs, names=obs.names[::-1], xy=obs.xy[::-1], samples=obs.samples[:, ::-1]
    )
    with pytest.raises(ObservationError, match="identical semantic ordering"):
        FacialInstrument(CONFIG).run(image, image, obs, reversed_obs)


def test_no_population_or_identity_percentage_in_evidence():
    p = phantom_points()
    ev = run(p, p).evidence()
    assert ev["consensus_similarity_percent"] is None
    assert ev["population_percentile"] is None
    assert all(
        m["value"] is None or np.isfinite(m["value"]) for m in ev["measurement_profile"]
    )


def test_off_image_anatomy_is_excluded_from_fit_field_and_report():
    points = phantom_points()
    target_points = points.copy()
    for i in (10, 11, 12, 13):
        target_points[i] = [500 + i, 500]
    image = render_phantom(points)
    a = phantom_observation(image, points)
    b = phantom_observation(image, target_points)
    config = InstrumentConfig(
        canvas_size=96, eye_span_px=32, uncertainty_draws=4, enable_flow=True
    )
    result = FacialInstrument(config).run(image, image, a, b)
    metrics = {m.name: m for m in result.profile.measurements}
    assert "nose_bridge" not in result.state.fit_names
    assert "nose_tip" not in result.state.fit_names
    assert not {10, 11, 12, 13} & set(result.field.triangles.ravel())
    assert metrics["geometry.nose.rms"].value is None
    assert metrics["geometry.nose.rms"].support == 0
    assert metrics["deformation.nose.log_stretch_rms"].value is None
    assert metrics["geometry.central.rms"].value < 1e-12


def test_clipped_eye_anchor_cannot_silently_define_the_canonical_chart():
    points = phantom_points()
    image = render_phantom(points)
    a = phantom_observation(image, points)
    target = points.copy()
    target[2] = [-1, 40]
    b = phantom_observation(image, target)
    with pytest.raises(ObservationError, match="eye anchors"):
        FacialInstrument().run(image, image, a, b)
