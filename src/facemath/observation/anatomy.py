"""One landmark grammar used by registration, masks, contours, and statistics.

left/right names refer to the VIEWER in an unmirrored frontal rendering, not
subject identity. MediaPipe indices are stable IDs; never reorder by x position.
Contours are model-inferred polylines, not independent image-edge observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import cv2
import numpy as np

from .contracts import Observation, ObservationError, object_digest

# Names shared with Wave 2, now with an explicit image-side convention.
SEMANTIC_21 = (
    "temple_left", "temple_right", "left_eye_outer", "left_eye_inner",
    "left_eye_top", "left_eye_bottom", "right_eye_inner", "right_eye_outer",
    "right_eye_top", "right_eye_bottom", "nose_bridge", "nose_tip",
    "left_nostril", "right_nostril", "mouth_left", "mouth_right",
    "upper_lip", "lower_lip", "chin", "jaw_left", "jaw_right",
)
MP_ANCHORS = dict(zip(SEMANTIC_21, (
    127, 356, 33, 133, 159, 145, 362, 263, 386, 374, 168, 1,
    98, 327, 61, 291, 0, 17, 152, 172, 397,
)))
MP_CONTOURS = {
    "left_eye": (33, 160, 158, 133, 153, 144),
    "right_eye": (263, 387, 385, 362, 380, 373),
    "nose": (168, 197, 5, 1, 2, 98, 97, 2, 326, 327),
    "mouth": (61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91),
    "jaw": (234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
            377, 400, 378, 379, 365, 397, 288, 361, 323, 454),
    "face": (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109),
}
SEM_CONTOURS = {
    "left_eye": (2, 4, 3, 5), "right_eye": (7, 8, 6, 9),
    "nose": (10, 12, 11, 13), "mouth": (14, 16, 15, 17),
    "jaw": (19, 18, 20), "face": (0, 1, 20, 18, 19),
}


@dataclass(frozen=True)
class Anatomy:
    schema: str
    anchors: Mapping[str, tuple[int, ...]]
    contours: Mapping[str, tuple[int, ...]]
    fit_anchor_names: tuple[str, ...]
    point_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", MappingProxyType({k: tuple(v) for k, v in self.anchors.items()}))
        object.__setattr__(self, "contours", MappingProxyType({k: tuple(v) for k, v in self.contours.items()}))

    def points(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        return {name: xy[list(indices)].mean(axis=0) for name, indices in self.anchors.items()}

    def region_indices(self, name: str) -> np.ndarray:
        if name == "central":
            return np.unique(np.concatenate([self.region_indices(r) for r in ("left_eye", "right_eye", "nose", "mouth")]))
        if name == "all":
            return np.arange(self.point_count)
        return np.unique(self.contours[name])

    @property
    def digest(self) -> str:
        return object_digest({"schema": self.schema, "anchors": dict(self.anchors),
                              "contours": dict(self.contours), "fit": self.fit_anchor_names,
                              "sides": "viewer", "revision": 3})


def anatomy_for(obs: Observation) -> Anatomy:
    if obs.schema == "mediapipe478.v1":
        if obs.names != tuple(f"mp_{i:03}" for i in range(478)):
            raise ObservationError("MediaPipe schema requires exactly the ordered 478 stable IDs")
        anchors = {k: (v,) for k, v in MP_ANCHORS.items()}
        contours = MP_CONTOURS
    elif obs.schema == "semantic21.v1":
        if set(obs.names) != set(SEMANTIC_21):
            raise ObservationError("semantic21 schema needs all 21 semantic names")
        lookup = {n: i for i, n in enumerate(obs.names)}
        anchors = {n: (lookup[n],) for n in SEMANTIC_21}
        contours = {k: tuple(lookup[SEMANTIC_21[i]] for i in ids) for k, ids in SEM_CONTOURS.items()}
    else:
        raise ObservationError(f"unknown anatomical grammar: {obs.schema}")
    anchors["left_eye"] = anchors["left_eye_outer"] + anchors["left_eye_inner"]
    anchors["right_eye"] = anchors["right_eye_inner"] + anchors["right_eye_outer"]
    fit = ("left_eye", "right_eye", "nose_bridge", "nose_tip", "mouth_left", "mouth_right",
           "temple_left", "temple_right", "jaw_left", "jaw_right", "chin")
    return Anatomy(obs.schema, anchors, contours, fit, len(obs.names))


def eye_span(xy: np.ndarray, anatomy: Anatomy) -> float:
    p = anatomy.points(xy)
    d = float(np.linalg.norm(p["right_eye"] - p["left_eye"]))
    if not np.isfinite(d) or d <= 1e-8:
        raise ObservationError("eye anchors coincide")
    return d


def polygon(shape: tuple[int, int], points: np.ndarray, hull: bool = False) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    pts = np.rint(points).astype(np.int32)
    if len(pts) < 3:
        return mask.astype(bool)
    if hull:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    else:
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def anatomical_masks(xy: np.ndarray, anatomy: Anatomy, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    face = polygon(shape, xy[list(anatomy.contours["face"])])
    result = {"face": face}
    for name in ("left_eye", "right_eye", "nose", "mouth", "jaw"):
        result[name] = polygon(shape, xy[list(anatomy.contours[name])], hull=True) & face
    result["central"] = np.logical_or.reduce([result[k] for k in ("left_eye", "right_eye", "nose", "mouth")])
    return result
