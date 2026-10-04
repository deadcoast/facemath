"""Image-derived contour checks in the shared canonical gauge.

Landmarks define search locations; they are NEVER moved to selected edges. This
is not statistically independent of landmarks or proof of anatomical identity.
Full normal-ray responses, observability, ambiguity and offsets are retained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
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


def normal_rays(points: np.ndarray, samples: int, *, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
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
    distances = (np.arange(samples)+.5)*cumulative[-1]/samples
    ids = np.minimum(np.searchsorted(cumulative, distances, side="right")-1, len(lengths)-1)
    tangent = delta[ids]/lengths[ids, None]
    pts = path[ids]+tangent*(distances-cumulative[ids])[:, None]
    normals = np.c_[-tangent[:, 1], tangent[:, 0]]
    return pts, normals


def sample_edge_profile(image: np.ndarray, valid_source: np.ndarray, points: np.ndarray, normals: np.ndarray,
                        *, radius_px: float, step_px: float = .5, minimum_gradient: float = .025,
                        maximum_ambiguity: float = .85) -> EdgeProfile:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ObservationError("edge input must be RGB uint8")
    h, w = image.shape[:2]
    if valid_source.shape != (h, w) or valid_source.dtype != bool:
        raise ObservationError("edge source support must match the image")
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if (points.ndim != 2 or points.shape[1] != 2 or normals.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)):
        raise ObservationError("edge rays need finite points and unit normals")
    if not 0 < step_px <= radius_px <= 32 or minimum_gradient <= 0 or not 0 < maximum_ambiguity < 1:
        raise ObservationError("invalid edge profile thresholds")
    count = int(np.ceil(radius_px/step_px))
    offsets = np.linspace(-radius_px, radius_px, 2*count+1)
    locations = points[:, None] + normals[:, None]*offsets[None, :, None]
    x, y = locations[..., 0].astype(np.float32), locations[..., 1].astype(np.float32)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)/255, (3, 3), .7)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    # Erode by gradient support: a plausible edge near missing/black border is not observation.
    support_eroded = cv2.erode(valid_source.astype(np.uint8), np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    def sample(field: np.ndarray) -> np.ndarray:
        return cv2.remap(field, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sx, sy = sample(gx), sample(gy)
    response = np.abs(sx*normals[:, 0, None] + sy*normals[:, 1, None])
    orientation = response/np.maximum(np.hypot(sx, sy), 1e-12)
    supported = (sample(support_eroded.astype(np.float32)) >= 1-1e-6).all(axis=1)
    best = np.argmax(response, axis=1)
    strength = response[np.arange(len(points)), best]
    # A broad edge is not a second peak. Competing LOCAL maxima farther than
    # 1.5px are ambiguity; exclude the main lobe and monotone shoulders.
    candidates = (response[:, 1:-1] >= response[:, :-2]) & (response[:, 1:-1] >= response[:, 2:])
    candidates &= np.abs(offsets[None, 1:-1]-offsets[best, None]) >= 1.5
    alternate = np.max(np.where(candidates, response[:, 1:-1], 0), axis=1)
    ratio = alternate/np.maximum(strength, 1e-12)
    accepted = supported & (strength >= minimum_gradient) & (ratio < maximum_ambiguity)
    accepted &= orientation[np.arange(len(points)), best] >= .65
    accepted &= (best > 0) & (best < len(offsets)-1)  # truncated peak is not localized
    return EdgeProfile(points, normals, offsets, response, orientation, supported, accepted,
                       offsets[best], strength, ratio)


REJECTION_REASONS = ("accepted", "source_unsupported", "weak_gradient", "ambiguous_peak",
                     "oblique_gradient", "search_boundary")
EDGE_REGIONS = ("left_eye", "right_eye", "mouth", "jaw")


def edge_rejection_codes(profile: EdgeProfile, minimum_gradient: float,
                         maximum_ambiguity: float) -> np.ndarray:
    """One first-failure reason per ray, using the sampler's existing rules."""
    best = np.argmax(profile.response, axis=1)
    tests = (~profile.ray_supported, profile.peak_strength < minimum_gradient,
             profile.ambiguity_ratio >= maximum_ambiguity,
             profile.orientation[np.arange(len(best)), best] < .65,
             (best == 0) | (best == len(profile.offsets)-1))
    codes = np.zeros(len(best), np.uint8)
    for code, rejected in enumerate(tests, 1):
        codes[(codes == 0) & rejected] = code
    if not np.array_equal(codes == 0, profile.accepted):
        raise ObservationError("edge rejection thresholds differ from sampled acceptance")
    return frozen_array(codes, np.uint8)


@dataclass(frozen=True)
class PairedEdgeProfile:
    """Two image responses on identical rays; no correspondence or warp is fit."""
    a: EdgeProfile
    b: EdgeProfile
    normalization_span_px: float
    tolerance_ipd: float
    minimum_gradient: float
    maximum_ambiguity: float
    accepted: np.ndarray = field(init=False)
    rejection_a: np.ndarray = field(init=False)
    rejection_b: np.ndarray = field(init=False)
    delta_b_minus_a_px: np.ndarray = field(init=False)
    displacement_b_minus_a_px: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        for name in ("points", "normals", "offsets"):
            if not np.array_equal(getattr(self.a, name), getattr(self.b, name)):
                raise ObservationError("paired edges require identical ray points, normals and offsets")
        if not len(self.a.points):
            raise ObservationError("paired edge rays cannot be empty")
        for value in (self.normalization_span_px, self.tolerance_ipd, self.minimum_gradient):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ObservationError("paired edge scales and thresholds must be finite and positive")
        if not np.isfinite(self.maximum_ambiguity) or not 0 < self.maximum_ambiguity < 1:
            raise ObservationError("invalid paired edge ambiguity threshold")
        for side in ("a", "b"):
            object.__setattr__(self, f"rejection_{side}", edge_rejection_codes(
                getattr(self, side), self.minimum_gradient, self.maximum_ambiguity))
        accepted = self.a.accepted & self.b.accepted
        delta = np.where(accepted, self.b.best_offset_px-self.a.best_offset_px, 0.)
        object.__setattr__(self, "accepted", frozen_array(accepted, bool))
        object.__setattr__(self, "delta_b_minus_a_px", frozen_array(delta))
        object.__setattr__(self, "displacement_b_minus_a_px", frozen_array(delta[:, None]*self.a.normals))

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {f"{side}_{name}": getattr(getattr(self, side), name)
                  for side in ("a", "b") for name in EdgeProfile.__dataclass_fields__}
        arrays.update({name: getattr(self, name) for name in (
            "accepted", "rejection_a", "rejection_b", "delta_b_minus_a_px", "displacement_b_minus_a_px")})
        return arrays

    def summary(self) -> dict:
        total, supported = len(self.accepted), int(self.accepted.sum())
        delta = self.delta_b_minus_a_px[self.accepted]
        within = int(np.sum(np.abs(delta) <= self.tolerance_ipd*self.normalization_span_px))
        stats = {"mean_signed_offset": float(np.mean(delta)) if supported else None,
                 "rms_offset": float(np.sqrt(np.mean(delta**2))) if supported else None,
                 "median_absolute_offset": float(np.median(np.abs(delta))) if supported else None,
                 "p90_absolute_offset": float(np.quantile(np.abs(delta), .9)) if supported else None}
        return {"total": total, "supported": supported, "unresolved": total-supported,
            "available": supported > 0, "reason": None if supported else "no_paired_localized_edges",
            "source_supported_both": int(np.sum(self.a.ray_supported & self.b.ray_supported)),
            "rejections": {side: {reason: int(np.sum(getattr(self, f"rejection_{side}") == code))
                for code, reason in enumerate(REJECTION_REASONS)} for side in ("a", "b")},
            "canonical_pixels": stats,
            "eye_spans": {k: None if v is None else v/self.normalization_span_px for k, v in stats.items()},
            "mean_displacement_b_minus_a_px": np.mean(self.displacement_b_minus_a_px[self.accepted], axis=0).tolist() if supported else None,
            "within_tolerance": within, "tolerance_eye_spans": self.tolerance_ipd,
            "within_tolerance_fraction_range": [within/total, (within+total-supported)/total],
            "normalization_span_px": self.normalization_span_px,
            "search_radius_px": float(self.a.offsets[-1]), "sample_step_px": float(self.a.offsets[1]-self.a.offsets[0])}


@dataclass(frozen=True)
class EdgeDiagnostics:
    profiles: Mapping[str, EdgeProfile]
    exclusions: Mapping[str, str]
    pairs: Mapping[str, PairedEdgeProfile] = field(default_factory=dict)
    requested_samples: int = 0
    tolerance_ipd: float = .05

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "exclusions", MappingProxyType(dict(self.exclusions)))
        object.__setattr__(self, "pairs", MappingProxyType(dict(self.pairs)))

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {f"edge_{key}_{name}": getattr(profile, name)
                  for key, profile in self.profiles.items() for name in profile.__dataclass_fields__}
        arrays.update({f"edge_pair_{key}_{name}": value for key, pair in self.pairs.items()
                       for name, value in pair.arrays().items()})
        return arrays

    def paired_summary(self) -> dict:
        rows = []
        for anchor in ("a", "b"):
            for region in EDGE_REGIONS:
                key = f"{anchor}_{region}"
                if key in self.pairs:
                    row = self.pairs[key].summary()
                else:
                    row = {"total": self.requested_samples, "supported": 0,
                        "unresolved": self.requested_samples, "available": False,
                        "reason": self.exclusions.get(key, "anchor_profile_unavailable"),
                        "source_supported_both": 0, "rejections": None,
                        "canonical_pixels": None, "eye_spans": None,
                        "mean_displacement_b_minus_a_px": None, "within_tolerance": 0,
                        "within_tolerance_fraction_range": [0., 1.]}
                rows.append({"anchor": anchor, "region": region, **row,
                    "array_prefix": f"edge_pair_{key}_" if key in self.pairs else None})
        return {"method": "facemath.shared_ray_edge_comparison.v21", "rows": rows,
            "tolerance_eye_spans": self.tolerance_ipd, "requested_rays_per_anchor": self.requested_samples,
            "sign": "B minus A along the anchor contour normal (-tangent_y, tangent_x)",
            "rejection_codes": {str(code): reason for code, reason in enumerate(REJECTION_REASONS)},
            "rejection_order": "first failing rule per side; counts partition the ray inventory",
            "range_meaning": "fixed-ray missingness range, not a confidence interval or resemblance probability",
            "limits": ["Landmark-guided gradients are image evidence, not independently labeled anatomical contours",
                "A-anchored and B-anchored rays are separate dependent inventories; do not pool them",
                "Normal displacement omits tangential motion; competing texture edges may differ in anatomical identity",
                "A missing edge or a peak outside the finite search window remains unresolved",
                "No landmark movement, extra alignment, fitted correspondence or reference confidence is introduced"]}

    def summary(self) -> dict:
        return {"method": "landmark_guided_normal_gradient_profiles.v1",
                "interpretation": "image edge support/disagreement, not ground truth; never moves anatomy",
                "profiles": {k: {"rays": len(p.points), "source_supported": int(p.ray_supported.sum()),
                                  "accepted": int(p.accepted.sum()),
                                  "median_absolute_offset_px": float(np.median(np.abs(p.best_offset_px[p.accepted]))) if p.accepted.any() else None}
                             for k, p in self.profiles.items()}, "exclusions": dict(self.exclusions),
                "paired_comparison": self.paired_summary()}


def diagnose_edges(state: object) -> EdgeDiagnostics:
    profiles, exclusions = {}, {}
    for side, obs, xy, image, support in (
        ("a", state.observation_a, state.xy_a, state.image_a, state.valid_source_a),
        ("b", state.observation_b, state.xy_b, state.image_b, state.valid_source_b),
    ):
        for region in EDGE_REGIONS:
            key = f"{side}_{region}"
            ids = list(state.anatomy.contours[region])
            if not np.all(obs.in_bounds[ids]):
                exclusions[key] = "contour_has_unobservable_source_vertices"
                continue
            try:
                points, normals = normal_rays(xy[ids], state.config.edge_samples, closed=region != "jaw")
            except ObservationError as exc:
                exclusions[key] = str(exc)
                continue
            profiles[key] = sample_edge_profile(image, support, points, normals,
                radius_px=min(16., state.config.edge_radius_ipd*state.normalization_span_px),
                minimum_gradient=state.config.edge_gradient_floor,
                maximum_ambiguity=state.config.edge_maximum_ambiguity)
    pairs = {}
    for key, anchor_profile in profiles.items():
        anchor = key.split("_", 1)[0]
        other = sample_edge_profile(state.image_b if anchor == "a" else state.image_a,
            state.valid_source_b if anchor == "a" else state.valid_source_a,
            anchor_profile.points, anchor_profile.normals,
            radius_px=min(16., state.config.edge_radius_ipd*state.normalization_span_px),
            minimum_gradient=state.config.edge_gradient_floor,
            maximum_ambiguity=state.config.edge_maximum_ambiguity)
        a, b = (anchor_profile, other) if anchor == "a" else (other, anchor_profile)
        pairs[key] = PairedEdgeProfile(a, b, state.normalization_span_px, state.config.tolerance_ipd,
            state.config.edge_gradient_floor, state.config.edge_maximum_ambiguity)
    return EdgeDiagnostics(profiles, exclusions, pairs, state.config.edge_samples, state.config.tolerance_ipd)
