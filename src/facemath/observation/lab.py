"""Known-geometry controls. These are schematic phantoms, never human baselines."""

from __future__ import annotations

import cv2
import numpy as np

from .anatomy import SEM_CONTOURS, SEMANTIC_21
from .canonical import InstrumentConfig
from .contracts import Observation, array_digest, canonical_json, object_digest
from .instrument import FacialInstrument
from .provider import apply_points
from .sequence_guard import crossfade, sequence_diagnostics


def phantom_points() -> np.ndarray:
    return np.array(
        [
            [28, 24],
            [72, 24],
            [33, 40],
            [43, 40],
            [38, 37],
            [38, 43],
            [57, 40],
            [67, 40],
            [62, 37],
            [62, 43],
            [50, 44],
            [50, 57],
            [45, 59],
            [55, 59],
            [42, 69],
            [58, 69],
            [50, 67],
            [50, 72],
            [50, 84],
            [32, 76],
            [68, 76],
        ],
        float,
    ) * 2 + [28, 0]


def render_phantom(
    points: np.ndarray, seed: int = 2, shape: tuple[int, int] = (224, 256)
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = cv2.GaussianBlur(rng.normal(0, 25, shape).astype(np.float32), (3, 3), 0)
    gray = np.clip(70 + texture, 0, 255).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    for region, ids in SEM_CONTOURS.items():
        pts = np.rint(points[list(ids)]).astype(np.int32)
        cv2.polylines(
            image, [pts], region not in ("jaw", "nose"), (190, 190, 190), 2, cv2.LINE_AA
        )
    for point in points:
        p = tuple(np.rint(point).astype(int))
        cv2.circle(image, p, 3, (230, 230, 230), -1)
        cv2.circle(image, p, 1, (10, 10, 10), -1)
    return image


def phantom_observation(
    image: np.ndarray, points: np.ndarray, sigma: float = 0.1
) -> Observation:
    return Observation(
        array_digest(image),
        image.shape[:2],
        "semantic21.v1",
        SEMANTIC_21,
        points,
        points[None],
        "synthetic_known_geometry",
        object_digest({"provider": "facemath_phantom", "version": 1}),
        sigma,
        1,
        ("ground_truth",),
        ("synthetic schematic, not a human face or population sample",),
        canonical_json(
            {"truth": "explicit_render_coordinates", "sigma": "assumed_sensor_noise"}
        ),
    )


def laboratory() -> dict:
    points = phantom_points()
    image = render_phantom(points)
    obs = phantom_observation(image, points)
    config = InstrumentConfig(
        uncertainty_draws=24, canvas_size=192, eye_span_px=64, enable_flow=False, seed=7
    )
    instrument = FacialInstrument(config)
    controls = []
    for amount in (0.0, 2.0, 4.0, 8.0, 12.0):
        q = points.copy()
        q[19, 0] -= amount
        q[20, 0] += amount
        target = render_phantom(q)
        result = instrument.run(image, target, obs, phantom_observation(target, q))
        metrics = {m.name: m.value for m in result.profile.measurements}
        controls.append(
            {
                "jaw_expansion_per_side_px": amount,
                "jaw_rms": metrics["geometry.jaw.rms"],
                "central_rms": metrics["geometry.central.rms"],
                "heldout_jaw_rms": metrics["registration.heldout_jaw.rms"],
            }
        )
    theta = 0.12
    transform = np.eye(3)
    transform[:2, :2] = 1.08 * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    transform[:2, 2] = [5, -8]
    q = apply_points(points, transform)
    b = render_phantom(q)
    invariance = instrument.run(image, b, obs, phantom_observation(b, q))
    maximum_residual = max(
        m.value
        for m in invariance.profile.measurements
        if m.name.startswith("geometry.")
        and m.name.endswith(".rms")
        and m.value is not None
    )
    # Entirely unrelated random textures still form a rank-one centered crossfade.
    rng = np.random.default_rng(420)
    a, b = rng.random((32, 32)), rng.random((32, 32))
    alpha = np.linspace(0, 1, 17)
    null = sequence_diagnostics(crossfade(a, b, alpha), alpha)
    return {
        "kind": "synthetic_falsification_controls_not_face_population",
        "jaw_controls": controls,
        "similarity_transform_max_residual": maximum_residual,
        "unrelated_crossfade_null": null,
        "checks": {
            "jaw_response_monotone": all(
                x["jaw_rms"] < y["jaw_rms"] for x, y in zip(controls, controls[1:])
            ),
            "similarity_invariance": maximum_residual < 1e-10,
            "unrelated_crossfade_rank1": abs(null["centered_rank1_energy_fraction"] - 1)
            < 1e-12,
            "null_not_similarity_evidence": null[
                "admissible_as_facial_similarity_evidence"
            ]
            is False,
        },
    }
