"""One declared coordinate gauge, with held-out-region registration diagnostics.

The fit is orientation-preserving. High residuals are measurements, not reasons
to reject an otherwise observable dissimilar pair. Huber fit weights never
remove those landmarks from the downstream comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import cv2
import numpy as np

from .anatomy import Anatomy, anatomy_for, anatomical_masks, eye_span
from .contracts import Observation, ObservationError, frozen_array, object_digest
from .provider import apply_points


@dataclass(frozen=True)
class InstrumentConfig:
    canvas_size: int = 256
    eye_span_px: float = 80.0
    minimum_source_eye_span_px: float = 8.0
    huber_delta_ipd: float = .04
    fit_iterations: int = 8
    uncertainty_draws: int = 96
    seed: int = 0
    tolerance_ipd: float = .05
    max_triangle_condition: float = 100.0
    fb_threshold_px: float = 1.5
    texture_floor: float = 2e-5
    sparse_dense_tolerance_ipd: float = .04
    enable_flow: bool = True
    enable_edges: bool = True
    edge_samples: int = 32
    edge_radius_ipd: float = .05
    edge_gradient_floor: float = .025
    edge_maximum_ambiguity: float = .85
    enable_intrinsic: bool = True
    intrinsic_length_tolerance: float = .05
    intrinsic_angle_tolerance_rad: float = .08726646259971647
    intrinsic_area_tolerance: float = .05
    intrinsic_asymmetry_tolerance: float = .05
    intrinsic_directions: int = 128

    def __post_init__(self) -> None:
        integers = {"canvas_size": self.canvas_size, "fit_iterations": self.fit_iterations,
                    "uncertainty_draws": self.uncertainty_draws, "seed": self.seed, "edge_samples": self.edge_samples}
        for key, value in integers.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ObservationError(f"{key} must be an integer")
        if not 64 <= self.canvas_size <= 1024:
            raise ObservationError("canvas_size must be in [64,1024]")
        if not 10 <= self.eye_span_px <= self.canvas_size / 2:
            raise ObservationError("eye_span_px must be in [10, canvas_size/2]")
        if not 4 <= self.uncertainty_draws <= 4096 or not 1 <= self.fit_iterations <= 100:
            raise ObservationError("invalid sensitivity or iteration budget")
        for name in ("minimum_source_eye_span_px", "huber_delta_ipd", "tolerance_ipd",
                     "max_triangle_condition", "fb_threshold_px", "texture_floor",
                     "sparse_dense_tolerance_ipd"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ObservationError(f"{name} must be finite and positive")
        if not isinstance(self.enable_flow, bool) or not isinstance(self.enable_edges, bool):
            raise ObservationError("diagnostic switches must be boolean")
        if not 4 <= self.edge_samples <= 256:
            raise ObservationError("edge_samples must be in [4,256]")
        if not np.isfinite(self.edge_radius_ipd) or not .01 <= self.edge_radius_ipd <= .2:
            raise ObservationError("edge_radius_ipd must be in [.01,.2]")
        if not np.isfinite(self.edge_gradient_floor) or not 0 < self.edge_gradient_floor < 1:
            raise ObservationError("invalid edge gradient floor")
        if not np.isfinite(self.edge_maximum_ambiguity) or not 0 < self.edge_maximum_ambiguity < 1:
            raise ObservationError("invalid edge ambiguity threshold")
        if type(self.enable_intrinsic) is not bool:
            raise ObservationError("enable_intrinsic must be boolean")
        if (type(self.intrinsic_directions) is not int or not 32 <= self.intrinsic_directions <= 2048
                or self.intrinsic_directions % 2):
            raise ObservationError("intrinsic_directions must be even and in [32,2048]")
        for name in ("intrinsic_length_tolerance", "intrinsic_angle_tolerance_rad",
                     "intrinsic_area_tolerance", "intrinsic_asymmetry_tolerance"):
            if isinstance(getattr(self, name), bool) or not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ObservationError(f"{name} must be finite and positive")
        if self.intrinsic_angle_tolerance_rad > np.pi or self.intrinsic_asymmetry_tolerance > 2:
            raise ObservationError("intrinsic angle/asymmetry tolerance exceeds its range")
        if self.seed < 0:
            raise ObservationError("seed must be nonnegative")

    def recipe(self) -> dict:
        from dataclasses import asdict
        return {"engine": "facemath-observation-7.0", **asdict(self)}


def proper_similarity(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Weighted proper Umeyama least squares, including signed singular values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape or len(a) < 2:
        raise ObservationError("similarity fit requires corresponding [N,2], N>=2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ObservationError("similarity points must be finite")
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(a),) or not np.isfinite(w).all() or np.any(w < 0) or np.count_nonzero(w) < 2:
        raise ObservationError("weights must have at least two positive finite entries")
    w = w / w.sum()
    ca, cb = w @ a, w @ b
    xa, xb = a - ca, b - cb
    denom = float(np.sum(w[:, None] * xa**2))
    if denom <= 1e-12:
        raise ObservationError("source configuration has zero weighted spread")
    u, singular, vt = np.linalg.svd((xa * w[:, None]).T @ xb)
    sign = np.ones(2)
    sign[-1] = 1 if np.linalg.det(vt.T @ u.T) >= 0 else -1
    rotation = vt.T @ np.diag(sign) @ u.T
    scale = float(singular @ sign / denom)
    if scale <= 1e-10:
        raise ObservationError("proper similarity collapses; configuration may be reflected/degenerate")
    matrix = np.eye(3)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = cb - matrix[:2, :2] @ ca
    error = apply_points(a, matrix) - b
    return matrix, float(np.sqrt(np.sum(w * np.sum(error**2, axis=1))))


def balanced_similarity(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Symmetric scale Procrustes, NOT predictive least-squares Umeyama.

    With weighted centered variances va,vb, s=sqrt(vb/va). Exchanging A/B
    gives the inverse map. The geometry residual normalizer below transforms
    by the same factor, so scalar landmark distances are exchange-symmetric.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.asarray(weights, float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape or len(a) < 2:
        raise ObservationError("balanced fit requires corresponding [N,2], N>=2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ObservationError("balanced fit points must be finite")
    if (w.shape != (len(a),) or not np.isfinite(w).all()
            or np.any(w < 0) or np.count_nonzero(w) < 2):
        raise ObservationError("balanced fit needs at least two positive finite weights")
    w = w / w.sum()
    ca, cb = w @ a, w @ b
    xa, xb = a - ca, b - cb
    va = float(np.sum(w[:, None] * xa**2))
    vb = float(np.sum(w[:, None] * xb**2))
    if min(va, vb) <= 1e-12:
        raise ObservationError("balanced fit has zero spread")
    u, _, vt = np.linalg.svd((xa * w[:, None]).T @ xb)
    sign = 1 if np.linalg.det(vt.T @ u.T) >= 0 else -1
    rotation = vt.T @ np.diag([1, sign]) @ u.T
    matrix = np.eye(3)
    matrix[:2, :2] = np.sqrt(vb / va) * rotation
    matrix[:2, 2] = cb - matrix[:2, :2] @ ca
    return matrix


def symmetric_normalizer(a: np.ndarray, b: np.ndarray, anatomy: Anatomy, matrix: np.ndarray) -> float:
    return float(np.sqrt(eye_span(apply_points(a, matrix), anatomy) * eye_span(b, anatomy)))


def _anchor_array(xy: np.ndarray, anatomy: Anatomy, names: tuple[str, ...]) -> np.ndarray:
    pts = anatomy.points(xy)
    return np.array([pts[name] for name in names])


def fit_anatomy(a: np.ndarray, b: np.ndarray, anatomy: Anatomy, config: InstrumentConfig,
                excluded: str | None = None, supported: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    exclusion = {
        "eyes": {"left_eye", "right_eye"},
        "left_eye": {"left_eye"}, "right_eye": {"right_eye"},
        "nose": {"nose_bridge", "nose_tip"}, "mouth": {"mouth_left", "mouth_right"},
        "jaw": {"jaw_left", "jaw_right", "chin"},
    }.get(excluded, set())
    if supported is None:
        supported = np.ones(len(a), dtype=bool)
    else:
        supported = np.asarray(supported, dtype=bool)
        if supported.shape != (len(a),):
            raise ObservationError("fit support must match landmark count")
    names = tuple(n for n in anatomy.fit_anchor_names
                  if n not in exclusion and np.all(supported[list(anatomy.anchors[n])]))
    if len(names) < 3:
        raise ObservationError("fewer than three observable registration anchors")
    aa, bb = _anchor_array(a, anatomy, names), _anchor_array(b, anatomy, names)
    if min(np.linalg.matrix_rank(aa - aa.mean(0), tol=1e-8), np.linalg.matrix_rank(bb - bb.mean(0), tol=1e-8)) < 2:
        raise ObservationError("registration anchors are collinear")
    weights = np.ones(len(aa))
    for _ in range(config.fit_iterations):
        matrix = balanced_similarity(aa, bb, weights)
        delta = config.huber_delta_ipd * symmetric_normalizer(a, b, anatomy, matrix)
        residual = np.linalg.norm(apply_points(aa, matrix) - bb, axis=1)
        weights = np.minimum(1.0, delta / np.maximum(residual, 1e-12))
    matrix = balanced_similarity(aa, bb, weights)
    return matrix, weights, names


def eye_chart(xy: np.ndarray, anatomy: Anatomy, config: InstrumentConfig,
              supported: np.ndarray | None = None) -> np.ndarray:
    anchors = anatomy.points(xy)
    left, right = anchors["left_eye"], anchors["right_eye"]
    d = eye_span(xy, anatomy)
    x = (right - left) / d
    rotation = np.array([[x[0], x[1]], [-x[1], x[0]]])
    matrix = np.eye(3)
    matrix[:2, :2] = (config.eye_span_px / d) * rotation
    origin = np.array([config.canvas_size / 2, .34 * config.canvas_size])
    matrix[:2, 2] = origin - matrix[:2, :2] @ ((left + right) / 2)
    # An off-image chin must not decide orientation for the observed face.
    # Use the first supported lower-face anchor; never turn the image over to fit.
    support = np.ones(len(xy), bool) if supported is None else np.asarray(supported, bool)
    if support.shape != (len(xy),):
        raise ObservationError("orientation support must match the landmark inventory")
    lower_anchors = [name for name in ("chin", "mouth_left", "mouth_right", "nose_tip")
                     if np.all(support[list(anatomy.anchors[name])])]
    if not lower_anchors:
        raise ObservationError("no observable lower-face anchor for orientation")
    if apply_points(np.array([anchors[lower_anchors[0]]]), matrix)[0, 1] <= origin[1]:
        raise ObservationError("face side/orientation convention inconsistent; do not silently reflect")
    return matrix


def warp_with_support(image: np.ndarray, matrix: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    # Constant borders, and fractional support excluded: no reflected pixels as evidence.
    rgb = cv2.warpAffine(image, matrix[:2], (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = cv2.warpAffine(np.ones(image.shape[:2], np.float32), matrix[:2], (size, size),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) >= 1 - 1e-6
    return rgb, support


@dataclass(frozen=True)
class CanonicalState:
    observation_a: Observation
    observation_b: Observation
    anatomy: Anatomy
    config: InstrumentConfig
    image_a: np.ndarray
    image_b: np.ndarray
    xy_a: np.ndarray
    xy_b: np.ndarray
    source_to_canonical_a: np.ndarray
    source_to_canonical_b: np.ndarray
    valid_source_a: np.ndarray
    valid_source_b: np.ndarray
    masks_a: Mapping[str, np.ndarray]
    masks_b: Mapping[str, np.ndarray]
    fit_weights: np.ndarray
    fit_names: tuple[str, ...]
    warnings: tuple[str, ...]
    recipe_sha256: str

    def __post_init__(self) -> None:
        for name in ("image_a", "image_b"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), np.uint8))
        for name in ("xy_a", "xy_b", "source_to_canonical_a", "source_to_canonical_b", "fit_weights"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        for name in ("valid_source_a", "valid_source_b"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), bool))
        for name in ("masks_a", "masks_b"):
            object.__setattr__(self, name, MappingProxyType({k: frozen_array(v, bool) for k, v in getattr(self, name).items()}))

    @property
    def normalization_span_px(self) -> float:
        return float(np.sqrt(eye_span(self.xy_a, self.anatomy) * eye_span(self.xy_b, self.anatomy)))

    @property
    def photometric_support(self) -> np.ndarray:
        return self.masks_a["face"] & self.masks_b["face"] & self.valid_source_a & self.valid_source_b


def canonicalize(image_a: np.ndarray, image_b: np.ndarray, a: Observation, b: Observation,
                 anatomy: Anatomy, config: InstrumentConfig) -> CanonicalState:
    a.require_image(image_a)
    b.require_image(image_b)
    if a.schema != b.schema or a.names != b.names:
        raise ObservationError("both observations must use identical semantic ordering")
    if anatomy.digest != anatomy_for(a).digest or anatomy.digest != anatomy_for(b).digest:
        raise ObservationError("both observations and canonical anatomy must use identical correspondence")
    warnings = []
    for label, obs in (("a", a), ("b", b)):
        span = eye_span(obs.xy, anatomy)
        if span < config.minimum_source_eye_span_px:
            raise ObservationError(f"{label}: too few source pixels between eyes for this recipe")
        if span < 40:
            warnings.append(f"{label}:low_source_eye_resolution")
        if np.mean(obs.in_bounds) < .8:
            raise ObservationError(f"{label}: more than 20% of landmarks outside source image")
        if len(obs.samples) < 3:
            warnings.append(f"{label}:repeatability_not_empirically_estimated")
        warnings.extend(f"{label}:{note}" for note in obs.source_notes)
    joint_observed = a.in_bounds & b.in_bounds
    # A canonical eye gauge requires observed eye anchors, not extrapolated ones.
    eye_ids = anatomy.anchors["left_eye"] + anatomy.anchors["right_eye"]
    if not np.all(joint_observed[list(eye_ids)]):
        raise ObservationError("canonical eye anchors must be inside both source images")
    m, weights, names = fit_anatomy(a.xy, b.xy, anatomy, config, supported=joint_observed)
    cb = eye_chart(b.xy, anatomy, config, supported=b.in_bounds)
    ca = cb @ m
    aa, va = warp_with_support(image_a, ca, config.canvas_size)
    bb, vb = warp_with_support(image_b, cb, config.canvas_size)
    pa, pb = apply_points(a.xy, ca), apply_points(b.xy, cb)
    masks_a = anatomical_masks(pa, anatomy, aa.shape[:2])
    masks_b = anatomical_masks(pb, anatomy, bb.shape[:2])
    from .journal import code_fingerprint, numeric_environment
    resolution_classes = sorted(int(np.digitize(eye_span(o.xy, anatomy), [40, 80])) for o in (a, b))
    recipe = object_digest({"config": config.recipe(), "anatomy": anatomy.digest,
        "engine_code_sha256": code_fingerprint(), "numeric_environment": numeric_environment(),
        "source_resolution_classes": resolution_classes,
        "providers": sorted((a.provider_fingerprint, b.provider_fingerprint)),
        "observation_sigma": sorted((a.noise_floor_px, b.noise_floor_px)),
        "sample_counts": sorted((len(a.samples), len(b.samples))),
        "sampling_protocols": sorted((o.attempted_samples, tuple(o.sample_labels)) for o in (a, b)),
        "geometry_units": "registered_eye_span_geometric_mean", "metric_revision": 2,
        "observable_landmark_indices": np.flatnonzero(joint_observed).tolist(), "fit_anchor_names": names})
    return CanonicalState(a, b, anatomy, config, aa, bb, pa, pb, ca, cb, va, vb,
                          masks_a, masks_b, weights, names, tuple(warnings), recipe)
