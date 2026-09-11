import numpy as np
import pytest

from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import ObservationError
from facemath.observation.edge_support import (
    normal_rays,
    sample_edge_profile,
)
from facemath.observation.instrument import FacialInstrument
from facemath.observation.lab import phantom_observation, phantom_points, render_phantom


def vertical_step(x=64):
    image = np.zeros((128, 128, 3), np.uint8)
    image[:, x:] = 255
    points = np.c_[np.full(10, 60.0), np.linspace(25, 95, 10)]
    normals = np.tile([1.0, 0.0], (10, 1))
    return image, points, normals


def test_subpixel_step_location_recovered_without_moving_landmarks():
    image, p, n = vertical_step()
    original = p.copy()
    profile = sample_edge_profile(image, np.ones((128, 128), bool), p, n, radius_px=6)
    assert profile.accepted.all()
    np.testing.assert_allclose(profile.best_offset_px, 3.5, atol=0.51)
    np.testing.assert_array_equal(p, original)
    assert profile.response.shape == (10, 25)


def test_two_equally_strong_edges_are_ambiguous():
    image, p, n = vertical_step(61)
    image[:, 67:] = 0
    profile = sample_edge_profile(image, np.ones((128, 128), bool), p, n, radius_px=10)
    assert not profile.accepted.any()
    assert np.all(profile.ambiguity_ratio > 0.9)


def test_flat_image_does_not_become_perfect_edge_agreement():
    image, p, n = vertical_step()
    image[:] = 120
    profile = sample_edge_profile(image, np.ones((128, 128), bool), p, n, radius_px=6)
    assert not profile.accepted.any() and np.all(profile.peak_strength == 0)


def test_source_padding_is_not_an_edge_observation():
    image, p, n = vertical_step()
    support = np.ones((128, 128), bool)
    support[:, 65:] = False
    profile = sample_edge_profile(image, support, p, n, radius_px=6)
    assert not profile.ray_supported.any() and not profile.accepted.any()


def test_edge_at_search_boundary_not_called_localized():
    image, p, n = vertical_step(70)
    profile = sample_edge_profile(image, np.ones((128, 128), bool), p, n, radius_px=4)
    assert not profile.accepted.any()


def test_closed_polyline_normals_are_unit_and_ordered():
    p, n = normal_rays(
        np.array([[10, 10], [20, 10], [20, 20], [10, 20]]), 16, closed=True
    )
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1)
    assert len(p) == 16 and np.all(p >= 10) and np.all(p <= 20)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((2, 2)),
        np.zeros((1, 2)),
        np.ones((3, 3)),
        np.array([[0, 0], [np.nan, 1]]),
    ],
)
def test_invalid_contour_has_no_invented_normal(points):
    with pytest.raises(ObservationError):
        normal_rays(points, 16, closed=False)


def test_edge_channel_uses_canonical_state_without_altering_profile():
    p = phantom_points()
    a = render_phantom(p)
    b = render_phantom(p, seed=8)
    oa, ob = phantom_observation(a, p), phantom_observation(b, p)
    result = FacialInstrument(
        InstrumentConfig(uncertainty_draws=4, enable_flow=False)
    ).run(a, b, oa, ob)
    assert len(result.edges.profiles) == 8
    assert "edge_a_left_eye_response" in result.arrays()
    assert result.evidence()["edge_diagnostics"]["interpretation"].startswith(
        "image edge"
    )
    assert all(
        m.value < 1e-12
        for m in result.profile.measurements
        if m.name.endswith(".rms") and m.name.startswith("geometry.")
    )


@pytest.mark.parametrize(
    "options",
    [
        {"edge_samples": True},
        {"edge_samples": 2},
        {"edge_radius_ipd": float("nan")},
        {"edge_maximum_ambiguity": 1},
        {"edge_gradient_floor": 0},
        {"enable_edges": 1},
    ],
)
def test_edge_configuration_is_strict(options):
    with pytest.raises(ObservationError):
        InstrumentConfig(**options)
