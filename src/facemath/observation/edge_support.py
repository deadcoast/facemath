"""Image-derived contour checks in the shared canonical gauge.

Landmarks define search locations; they are NEVER moved to selected edges. This
is not statistically independent of landmarks or proof of anatomical identity.
Full normal-ray responses, observability, ambiguity and offsets are retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import cv2
import numpy as np

from .contracts import ObservationError, frozen_array


@dataclass(frozen=True)
class EdgeProfile:
    points: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    response: np.ndarray
    orientation: np.ndarray
    ray_supported: np.ndarray
    accepted: np.ndarray
    best_offset_px: np.ndarray
    peak_strength: np.ndarray
    ambiguity_ratio: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            dtype = bool if name in ("ray_supported", "accepted") else float
            object.__setattr__(self, name, frozen_array(getattr(self, name), dtype))


def normal_rays(
    points: np.ndarray, samples: int, *, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or not np.isfinite(points).all()
    ):
        raise ObservationError("polyline must have at least two finite xy points")
    if type(samples) is not int or not 4 <= samples <= 4096:
        raise ObservationError("contour samples must be in [4,4096]")
    path = np.vstack((points, points[0])) if closed else points
    delta = np.diff(path, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-8):
        raise ObservationError("zero-length contour segments cannot define normals")
    cumulative = np.r_[0, np.cumsum(lengths)]
    # Mid-bin samples avoid vertex singularities; never fit a second contour.
    distances = (np.arange(samples) + 0.5) * cumulative[-1] / samples
    ids = np.minimum(
        np.searchsorted(cumulative, distances, side="right") - 1, len(lengths) - 1
    )
    tangent = delta[ids] / lengths[ids, None]
    pts = path[ids] + tangent * (distances - cumulative[ids])[:, None]
    normals = np.c_[-tangent[:, 1], tangent[:, 0]]
    return pts, normals


def sample_edge_profile(
    image: np.ndarray,
    valid_source: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    radius_px: float,
    step_px: float = 0.5,
    minimum_gradient: float = 0.025,
    maximum_ambiguity: float = 0.85,
) -> EdgeProfile:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ObservationError("edge input must be RGB uint8")
    h, w = image.shape[:2]
    if valid_source.shape != (h, w) or valid_source.dtype != bool:
        raise ObservationError("edge source support must match the image")
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or normals.shape != points.shape
        or not np.isfinite(points).all()
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    ):
        raise ObservationError("edge rays need finite points and unit normals")
    if (
        not 0 < step_px <= radius_px <= 32
        or minimum_gradient <= 0
        or not 0 < maximum_ambiguity < 1
    ):
        raise ObservationError("invalid edge profile thresholds")
    count = int(np.ceil(radius_px / step_px))
    offsets = np.linspace(-radius_px, radius_px, 2 * count + 1)
    locations = points[:, None] + normals[:, None] * offsets[None, :, None]
    x, y = locations[..., 0].astype(np.float32), locations[..., 1].astype(np.float32)
    gray = cv2.GaussianBlur(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, (3, 3), 0.7
    )
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    # Erode by gradient support: a plausible edge near missing/black border is not observation.
    support_eroded = cv2.erode(
        valid_source.astype(np.uint8),
        np.ones((5, 5), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    def sample(field: np.ndarray) -> np.ndarray:
        return cv2.remap(field, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    sx, sy = sample(gx), sample(gy)
    response = np.abs(sx * normals[:, 0, None] + sy * normals[:, 1, None])
    orientation = response / np.maximum(np.hypot(sx, sy), 1e-12)
    supported = (sample(support_eroded.astype(np.float32)) >= 1 - 1e-6).all(axis=1)
    best = np.argmax(response, axis=1)
    strength = response[np.arange(len(points)), best]
    # A broad edge is not a second peak. Competing LOCAL maxima farther than
    # 1.5px are ambiguity; exclude the main lobe and monotone shoulders.
    candidates = (response[:, 1:-1] >= response[:, :-2]) & (
        response[:, 1:-1] >= response[:, 2:]
    )
    candidates &= np.abs(offsets[None, 1:-1] - offsets[best, None]) >= 1.5
    alternate = np.max(np.where(candidates, response[:, 1:-1], 0), axis=1)
    ratio = alternate / np.maximum(strength, 1e-12)
    accepted = supported & (strength >= minimum_gradient) & (ratio < maximum_ambiguity)
    accepted &= orientation[np.arange(len(points)), best] >= 0.65
    accepted &= (best > 0) & (
        best < len(offsets) - 1
    )  # truncated peak is not localized
    return EdgeProfile(
        points,
        normals,
        offsets,
        response,
        orientation,
        supported,
        accepted,
        offsets[best],
        strength,
        ratio,
    )


@dataclass(frozen=True)
class EdgeDiagnostics:
    profiles: Mapping[str, EdgeProfile]
    exclusions: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "exclusions", MappingProxyType(dict(self.exclusions)))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            f"edge_{key}_{name}": getattr(profile, name)
            for key, profile in self.profiles.items()
            for name in profile.__dataclass_fields__
        }

    def summary(self) -> dict:
        return {
            "method": "landmark_guided_normal_gradient_profiles.v1",
            "interpretation": "image edge support/disagreement, not ground truth; never moves anatomy",
            "profiles": {
                k: {
                    "rays": len(p.points),
                    "source_supported": int(p.ray_supported.sum()),
                    "accepted": int(p.accepted.sum()),
                    "median_absolute_offset_px": float(
                        np.median(np.abs(p.best_offset_px[p.accepted]))
                    )
                    if p.accepted.any()
                    else None,
                }
                for k, p in self.profiles.items()
            },
            "exclusions": dict(self.exclusions),
        }


def diagnose_edges(state: object) -> EdgeDiagnostics:
    profiles, exclusions = {}, {}
    for side, obs, xy, image, support in (
        ("a", state.observation_a, state.xy_a, state.image_a, state.valid_source_a),
        ("b", state.observation_b, state.xy_b, state.image_b, state.valid_source_b),
    ):
        for region in ("left_eye", "right_eye", "mouth", "jaw"):
            key = f"{side}_{region}"
            ids = list(state.anatomy.contours[region])
            if not np.all(obs.in_bounds[ids]):
                exclusions[key] = "contour_has_unobservable_source_vertices"
                continue
            try:
                points, normals = normal_rays(
                    xy[ids], state.config.edge_samples, closed=region != "jaw"
                )
            except ObservationError as exc:
                exclusions[key] = str(exc)
                continue
            profiles[key] = sample_edge_profile(
                image,
                support,
                points,
                normals,
                radius_px=min(
                    16.0, state.config.edge_radius_ipd * state.normalization_span_px
                ),
                minimum_gradient=state.config.edge_gradient_floor,
                maximum_ambiguity=state.config.edge_maximum_ambiguity,
            )
    return EdgeDiagnostics(profiles, exclusions)
