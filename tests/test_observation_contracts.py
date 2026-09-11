import json
from dataclasses import replace

import numpy as np
import pytest

from facemath.observation.anatomy import anatomy_for, eye_span
from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import (
    Observation,
    ObservationError,
    array_digest,
    frozen_array,
    rgb_image,
)
from facemath.observation.lab import phantom_observation, phantom_points, render_phantom


def sample():
    points = phantom_points()
    image = render_phantom(points)
    return image, phantom_observation(image, points)


def test_observation_roundtrip_preserves_all_data_and_digest():
    _, obs = sample()
    clone = Observation.from_dict(
        json.loads(json.dumps(obs.to_dict(), allow_nan=False))
    )
    assert clone.digest == obs.digest
    np.testing.assert_array_equal(clone.samples, obs.samples)


@pytest.mark.parametrize("field", ["xy", "samples", "covariance", "in_bounds"])
def test_arrays_are_actually_immutable(field):
    _, obs = sample()
    arr = getattr(obs, field)
    with pytest.raises(ValueError):
        arr.setflags(write=True)
    with pytest.raises(ValueError):
        arr.flat[0] = 3


def test_constructor_does_not_alias_mutable_input():
    _, obs = sample()
    x = np.array(obs.xy)
    copy = replace(obs, xy=x)
    x[:] = 0
    assert copy.xy.sum() > 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_arrays_rejected(bad):
    with pytest.raises(ObservationError):
        frozen_array([1, bad])


@pytest.mark.parametrize("dtype", [float, np.float32, np.int16, np.uint16, bool])
def test_rgb_range_is_never_guessed(dtype):
    with pytest.raises(ObservationError):
        rgb_image(np.zeros((20, 20, 3), dtype=dtype))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise_floor_px": 0},
        {"noise_floor_px": -1},
        {"noise_floor_px": np.nan},
        {"attempted_samples": 0},
        {"sample_labels": ()},
        {"image_sha256": "123"},
        {"metadata_json": '{"x":NaN}'},
        {"metadata_json": "[]"},
        {"schema": ""},
    ],
)
def test_invalid_observations_are_rejected(kwargs):
    _, obs = sample()
    with pytest.raises((ObservationError, ValueError)):
        replace(obs, **kwargs)


def test_duplicate_semantic_names_rejected():
    _, obs = sample()
    with pytest.raises(ObservationError):
        replace(obs, names=("same",) * len(obs.names))


def test_observation_bound_to_exact_raster():
    image, obs = sample()
    image = image.copy()
    image[0, 0, 0] ^= 1
    with pytest.raises(ObservationError, match="different image"):
        obs.require_image(image)


def test_covariance_preserves_empirical_repeatability_and_declared_floor():
    _, obs = sample()
    delta = np.zeros_like(obs.xy)
    delta[:, 0] = 2
    obs = replace(
        obs,
        samples=np.stack((obs.xy - delta, obs.xy + delta)),
        attempted_samples=2,
        sample_labels=("minus", "plus"),
        noise_floor_px=1,
    )
    np.testing.assert_allclose(obs.covariance[:, 0, 0], 9)
    np.testing.assert_allclose(obs.covariance[:, 1, 1], 1)
    np.testing.assert_allclose(obs.covariance[:, 0, 1], 0)


def test_same_bytes_different_shape_have_different_fingerprints():
    a = np.arange(12).reshape(3, 4)
    assert array_digest(a) != array_digest(a.reshape(4, 3))


def test_grammar_all_means_all_observed_landmarks():
    _, obs = sample()
    anatomy = anatomy_for(obs)
    assert np.array_equal(anatomy.region_indices("all"), np.arange(21))
    assert eye_span(obs.xy, anatomy) == 48
    with pytest.raises(TypeError):
        anatomy.anchors["nose_tip"] = (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"canvas_size": 32},
        {"canvas_size": 512.5},
        {"eye_span_px": 0},
        {"eye_span_px": 999},
        {"uncertainty_draws": 0},
        {"seed": -1},
        {"texture_floor": np.inf},
        {"huber_delta_ipd": 0},
        {"fit_iterations": True},
    ],
)
def test_invalid_instrument_config(kwargs):
    with pytest.raises(ObservationError):
        InstrumentConfig(**kwargs)
