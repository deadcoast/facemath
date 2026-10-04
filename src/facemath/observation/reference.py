"""Independent localization references, not a second detector or a face identity.

A deliberately small cross-schema grammar avoids pretending all 68 iBUG points
are anatomically identical to arbitrary MediaPipe vertices. References remain
in SOURCE PIXELS. Never align predictions to these labels when evaluating error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import json
import re

import numpy as np

from .anatomy import anatomy_for
from .contracts import Observation, ObservationError, canonical_json, frozen_array, object_digest
from .correspondence import correspondence_recipe

REFERENCE_NAMES = (
    "left_eye_outer", "left_eye_inner", "right_eye_inner", "right_eye_outer",
    "nose_tip", "mouth_left", "mouth_right", "chin",
)
REFERENCE_DEFINITIONS = (
    "Outer eye corner on the viewer's left; not pupil, eyelash, or brow.",
    "Inner eye corner on the viewer's left, toward the nose.",
    "Inner eye corner on the viewer's right, toward the nose.",
    "Outer eye corner on the viewer's right; not pupil, eyelash, or brow.",
    "Visible center of the nose tip; not the subnasale or nostril midpoint.",
    "Mouth corner on the viewer's left, at the lip junction.",
    "Mouth corner on the viewer's right, at the lip junction.",
    "Lowest visible central chin contour; not beard hair. Mark unobservable when hidden.",
)
IBUG68_INDICES = (36, 39, 42, 45, 30, 48, 54, 8)
REFERENCE_PROTOCOL = {
    "id": "facemath.reference8.v1", "names": REFERENCE_NAMES,
    "definitions": REFERENCE_DEFINITIONS, "sides": "viewer_in_source_image_no_mirroring",
    "coordinates": "zero_based_source_pixel_centers_xy",
    "normalization": "reference_eye_corner_midpoint_distance",
    "mapping_limit": "nominal cross-dataset semantic correspondence, not validated anatomical equivalence",
}
REFERENCE_PROTOCOL_SHA256 = object_digest(REFERENCE_PROTOCOL)
REFERENCE_KINDS = ("human_blind", "external_dataset", "manual_unblinded", "synthetic_fixture", "model_generated")
CALIBRATION_REFERENCE_KINDS = ("human_blind", "external_dataset", "synthetic_fixture", "model_generated")


def check_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ObservationError(f"{name} must be a full lowercase SHA-256")


@dataclass(frozen=True)
class ReferenceAnnotation:
    image_sha256: str
    image_shape: tuple[int, int]
    xy: np.ndarray
    visible: np.ndarray
    rater_id: str
    kind: str
    source: str
    prediction_exposure: str = "none_declared"
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        check_digest(self.image_sha256, "reference image digest")
        if (len(self.image_shape) != 2 or any(type(v) is not int or v < 8 for v in self.image_shape)):
            raise ObservationError("reference shape must contain two integer image dimensions >=8")
        xy = frozen_array(self.xy)
        visible_input = np.asarray(self.visible)
        if visible_input.dtype != bool or visible_input.shape != (len(REFERENCE_NAMES),):
            raise ObservationError("reference visible must be an explicit boolean [8] mask")
        visible = frozen_array(visible_input, bool)
        if xy.shape != (len(REFERENCE_NAMES), 2):
            raise ObservationError("reference requires [8,2] source coordinates")
        h, w = self.image_shape
        if np.any(visible & ((xy[:, 0] < 0) | (xy[:, 0] > w - 1) | (xy[:, 1] < 0) | (xy[:, 1] > h - 1))):
            raise ObservationError("visible reference point is outside its source image")
        if np.any(xy[~visible] != 0):
            raise ObservationError("unobservable reference coordinates must be zero sentinels")
        if self.kind not in REFERENCE_KINDS:
            raise ObservationError("unknown reference annotation kind")
        if self.prediction_exposure not in ("none_declared", "unknown", "shown"):
            raise ObservationError("prediction exposure must be explicit")
        if self.kind == "human_blind" and self.prediction_exposure != "none_declared":
            raise ObservationError("blind reference cannot declare exposed predictions")
        if any(not isinstance(v, str) or not v.strip() for v in (self.rater_id, self.source)):
            raise ObservationError("reference needs an explicit rater and source")
        metadata = json.loads(self.metadata_json)
        if not isinstance(metadata, dict):
            raise ObservationError("reference metadata must be an object")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "image_shape", tuple(self.image_shape))
        object.__setattr__(self, "metadata_json", canonical_json(metadata))

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "facemath.reference.v4", "protocol_sha256": REFERENCE_PROTOCOL_SHA256,
            "names": list(REFERENCE_NAMES), "image_sha256": self.image_sha256,
            "image_shape": list(self.image_shape), "xy": self.xy.tolist(),
            "visible": self.visible.tolist(), "rater_id": self.rater_id,
            "kind": self.kind, "source": self.source, "prediction_exposure": self.prediction_exposure,
            "metadata": json.loads(self.metadata_json),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> ReferenceAnnotation:
        if obj.get("format") != "facemath.reference.v4" or obj.get("protocol_sha256") != REFERENCE_PROTOCOL_SHA256:
            raise ObservationError("unknown reference format or protocol")
        if tuple(obj["names"]) != REFERENCE_NAMES:
            raise ObservationError("reference point ordering differs from the protocol")
        return cls(obj["image_sha256"], tuple(obj["image_shape"]), obj["xy"], obj["visible"],
                   obj["rater_id"], obj["kind"], obj["source"], obj["prediction_exposure"],
                   canonical_json(obj["metadata"]))


@dataclass(frozen=True)
class ReferenceConsensus:
    xy: np.ndarray
    visible: np.ndarray
    rater_count: np.ndarray
    disagreement_rms_px: np.ndarray
    annotation_digests: tuple[str, ...]
    rater_ids: tuple[str, ...]
    kind: str

    def __post_init__(self) -> None:
        for name, dtype in (("xy", float), ("visible", bool), ("rater_count", np.int64), ("disagreement_rms_px", float)):
            object.__setattr__(self, name, frozen_array(getattr(self, name), dtype))

    @property
    def eye_span_px(self) -> float:
        if not np.all(self.visible[:4]):
            raise ObservationError("all four eye corners must be reference-observable for normalization")
        d = float(np.linalg.norm(self.xy[2:4].mean(0) - self.xy[:2].mean(0)))
        if d < 8:
            raise ObservationError("reference interocular span below eight source pixels")
        return d


def consensus(annotations: tuple[ReferenceAnnotation, ...], minimum_raters: int = 1) -> ReferenceConsensus:
    if type(minimum_raters) is not int or minimum_raters < 1:
        raise ObservationError("minimum_raters must be positive")
    if not annotations or len({r.rater_id for r in annotations}) != len(annotations):
        raise ObservationError("references need nonempty distinct rater IDs")
    if len({(r.image_sha256, r.image_shape) for r in annotations}) != 1:
        raise ObservationError("reference annotations belong to different source images")
    kinds = {r.kind for r in annotations}
    if len(kinds) != 1:
        raise ObservationError("do not pool synthetic, exposed, and independent reference protocols")
    masks = np.stack([a.visible for a in annotations])
    coordinates = np.stack([a.xy for a in annotations])
    count = masks.sum(0)
    visible = count >= minimum_raters
    means = np.sum(coordinates * masks[..., None], axis=0) / np.maximum(count[:, None], 1)
    deviations = np.sum((coordinates - means[None])**2, axis=2)
    rms = np.sqrt(np.sum(deviations * masks, axis=0) / np.maximum(count, 1))
    means[~visible], rms[~visible] = 0, 0
    return ReferenceConsensus(means, visible, count, rms,
                              tuple(a.digest for a in annotations), tuple(a.rater_id for a in annotations), next(iter(kinds)))


def mapped_prediction(observation: Observation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anatomy = anatomy_for(observation)
    indices = []
    for name in REFERENCE_NAMES:
        point_ids = anatomy.anchors[name]
        if len(point_ids) != 1:
            raise ObservationError("reference grammar requires individual vertices, not averaged detector points")
        indices.append(point_ids[0])
    ids = np.asarray(indices)
    return observation.xy[ids], observation.covariance[ids], observation.in_bounds[ids]


def observer_recipe(observation: Observation) -> str:
    """Attempted protocol, not realized successes: do not select only stable detections."""
    return object_digest({
        "provider": observation.provider, "provider_fingerprint": observation.provider_fingerprint,
        "schema": observation.schema, "noise_floor_px": observation.noise_floor_px,
        "attempted_samples": observation.attempted_samples,
        "reference_protocol_sha256": REFERENCE_PROTOCOL_SHA256,
        **correspondence_recipe(observation.schema, observation.correspondence),
        "score": "max_reference8_mahalanobis_to_independent_consensus.v1",
    })


def parse_ibug68(text: str) -> np.ndarray:
    """All 68 original PTS coordinates, with no origin change or rounding."""
    match = re.fullmatch(r"\s*version:\s*1\s+n_points:\s*68\s*\{\s*(.*?)\s*\}\s*", text, re.S)
    if not match:
        raise ObservationError("expected version:1 n_points:68 PTS block")
    try:
        rows = [[float(x) for x in line.split()] for line in match[1].splitlines() if line.strip()]
        points = np.asarray(rows, float)
    except ValueError as exc:
        raise ObservationError("invalid PTS coordinate rows") from exc
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ObservationError("PTS must contain exactly 68 finite xy rows")
    return points


def import_ibug68(text: str, *, image_sha256: str, image_shape: tuple[int, int],
                  coordinate_origin: int, source: str) -> ReferenceAnnotation:
    """Strict .pts reader. Caller declares 0/1 origin; no guessing or clipping."""
    if coordinate_origin not in (0, 1) or type(coordinate_origin) is not int:
        raise ObservationError("coordinate_origin must explicitly be 0 or 1")
    points = parse_ibug68(text)
    points = points[list(IBUG68_INDICES)] - coordinate_origin
    h, w = image_shape
    visible = (points[:, 0] >= 0) & (points[:, 0] <= w-1) & (points[:, 1] >= 0) & (points[:, 1] <= h-1)
    # The file has no occlusion labels: in-bounds is not visibility ground truth.
    if not np.all(visible):
        raise ObservationError("PTS selected labels are out of frame; supply explicit visibility rather than clipping")
    return ReferenceAnnotation(image_sha256, image_shape, points, visible, "dataset_annotation", "external_dataset", source,
        "unknown", canonical_json({"source_schema": "ibug68", "source_origin": coordinate_origin,
        "source_indices_zero_based": IBUG68_INDICES, "visibility": "unavailable_in_pts_assumed_visible",
        "mapping_requires_review": True, "license": "use_requires_source_dataset_terms"}))
