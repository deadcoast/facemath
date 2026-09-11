import numpy as np
import pytest

from facemath.observation.anatomy import anatomy_for
from facemath.observation.canonical import (
    InstrumentConfig,
    balanced_similarity,
    fit_anatomy,
    proper_similarity,
)
from facemath.observation.contracts import ObservationError
from facemath.observation.lab import phantom_observation, phantom_points, render_phantom
from facemath.observation.provider import apply_points


@pytest.mark.parametrize("seed", range(12))
def test_proper_similarity_recovers_known_transform(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(0, 10, (30, 2))
    theta, scale = rng.uniform(-2, 2), rng.uniform(0.3, 3)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    matrix[:2, 2] = rng.normal(0, 10, 2)
    target = apply_points(points, matrix)
    fit, error = proper_similarity(points, target, rng.uniform(0.1, 1, len(points)))
    np.testing.assert_allclose(fit, matrix, atol=1e-12)
    assert error < 1e-12


def test_reflection_correction_uses_signed_singular_values():
    points = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float)
    reflected = points * [-1, 1]
    matrix, _ = proper_similarity(points, reflected)
    assert np.linalg.det(matrix[:2, :2]) > 0
    assert np.isclose(np.sqrt(np.linalg.det(matrix[:2, :2])), 0.6)
    # Independent closed-form rectangle solution: residual mean square is .8.
    assert np.isclose(
        np.mean(np.sum((apply_points(points, matrix) - reflected) ** 2, axis=1)), 0.8
    )


@pytest.mark.parametrize(
    "weights", [[0, 0, 0], [-1, 1, 1], [1, np.nan, 1], [1, 1], [0, 0, 1]]
)
def test_invalid_fit_weights(weights):
    p = np.array([[0, 0], [1, 0], [0, 1]], float)
    with pytest.raises(ObservationError):
        proper_similarity(p, p, weights)


def test_degenerate_source_fit_rejected():
    with pytest.raises(ObservationError):
        proper_similarity(np.ones((5, 2)), np.ones((5, 2)))


@pytest.mark.parametrize("seed", range(8))
def test_balanced_similarity_swapped_pair_is_exact_inverse(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(20, 2))
    b = a + rng.normal(0, 0.2, a.shape)
    weights = rng.uniform(0.1, 1, len(a))
    ab = balanced_similarity(a, b, weights)
    ba = balanced_similarity(b, a, weights)
    np.testing.assert_allclose(ba @ ab, np.eye(3), atol=1e-12)


def test_region_exclusion_really_removes_region_fit_anchors():
    p = phantom_points()
    image = render_phantom(p)
    anatomy = anatomy_for(phantom_observation(image, p))
    q = p.copy()
    q[19, 0] -= 8
    q[20, 0] += 8
    matrix, _, names = fit_anatomy(p, q, anatomy, InstrumentConfig(), "jaw")
    assert not {"jaw_left", "jaw_right", "chin"} & set(names)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)
