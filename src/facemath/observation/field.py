"""A single piecewise-affine anatomical field and independent flow diagnostics.

The anatomical field is an interpolation MODEL between sparse observations.
Farneback flow is kept separate: cycle-consistent image motion is not proof of
cross-person anatomical correspondence, and neither is a physical tissue strain.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, QhullError

from .canonical import CanonicalState
from .contracts import ObservationError, frozen_array


@dataclass(frozen=True)
class AnatomicalField:
    triangles: np.ndarray
    simplex_index: np.ndarray
    uv: np.ndarray
    jacobians: np.ndarray
    determinant: np.ndarray
    green_strain: np.ndarray
    principal_log_stretch: np.ndarray
    condition: np.ndarray
    valid_triangles: np.ndarray
    valid: np.ndarray
    folded_fraction: float

    def __post_init__(self) -> None:
        for name in ("triangles", "simplex_index"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), np.int32))
        for name in ("valid", "valid_triangles"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), bool))
        for name in ("uv", "jacobians", "determinant", "green_strain", "principal_log_stretch", "condition"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))


def affine_triangle_field(a: np.ndarray, b: np.ndarray, shape: tuple[int, int],
                          source_support: np.ndarray, target_support: np.ndarray,
                          max_condition: float = 100.0) -> AnatomicalField:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) < 3:
        raise ObservationError("field requires matching [N,2] point arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ObservationError("field coordinates must be finite")
    if len(np.unique(a, axis=0)) != len(a):
        raise ObservationError("duplicate source landmarks make correspondence ambiguous")
    if source_support.shape != shape or target_support.shape != shape:
        raise ObservationError("source/target supports must match field shape")
    try:
        triangulation = Delaunay(a)
    except QhullError as exc:
        raise ObservationError("source landmark triangulation is degenerate") from exc
    triangles = triangulation.simplices
    aa, bb = a[triangles], b[triangles]
    basis_a = np.stack((aa[:, 1] - aa[:, 0], aa[:, 2] - aa[:, 0]), axis=-1)
    basis_b = np.stack((bb[:, 1] - bb[:, 0], bb[:, 2] - bb[:, 0]), axis=-1)
    condition = np.linalg.cond(basis_a)
    if not np.isfinite(condition).all():
        raise ObservationError("triangulation contains singular source triangles")
    jac = basis_b @ np.linalg.inv(basis_a)
    determinant = np.linalg.det(jac)
    green = .5 * (np.swapaxes(jac, -1, -2) @ jac - np.eye(2))
    singular = np.linalg.svd(jac, compute_uv=False)
    valid_tri = (condition <= max_condition) & (determinant > 1e-8) & (singular[:, -1] > 1e-8)
    logs = np.log(np.maximum(singular, 1e-12))  # invalid triangles are retained but excluded
    yy, xx = np.indices(shape)
    pixels = np.stack((xx.ravel(), yy.ravel()), axis=1)
    simplex = triangulation.find_simplex(pixels)
    inside = simplex >= 0
    uv = np.zeros((len(pixels), 2))
    selected = simplex[inside]
    bary2 = np.einsum("nij,nj->ni", triangulation.transform[selected, :2],
                      pixels[inside] - triangulation.transform[selected, 2])
    bary = np.c_[bary2, 1 - bary2.sum(axis=1)]
    target = np.einsum("ni,nij->nj", bary, b[triangles[selected]])
    uv[inside] = target - pixels[inside]
    uv = uv.reshape(*shape, 2)
    simplex = simplex.reshape(shape)
    qx = (xx + uv[..., 0]).astype(np.float32)
    qy = (yy + uv[..., 1]).astype(np.float32)
    mapped_support = cv2.remap(target_support.astype(np.float32), qx, qy,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) >= 1 - 1e-6
    valid = (simplex >= 0) & source_support & mapped_support
    valid &= (qx >= 0) & (qx <= shape[1] - 1) & (qy >= 0) & (qy <= shape[0] - 1)
    valid &= valid_tri[np.maximum(simplex, 0)]
    return AnatomicalField(triangles, simplex, uv, jac, determinant, green, logs, condition,
                           valid_tri, valid, float(np.mean(determinant <= 0)))


@dataclass(frozen=True)
class FlowDiagnostics:
    uv: np.ndarray
    cycle_error_px: np.ndarray
    texture_observability: np.ndarray
    photometric_residual: np.ndarray
    anatomical_disagreement_px: np.ndarray
    admissible: np.ndarray
    concordant: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            dtype = bool if name in ("admissible", "concordant") else float
            object.__setattr__(self, name, frozen_array(getattr(self, name), dtype))


def _texture(gray: np.ndarray, floor: float) -> np.ndarray:
    f = gray.astype(float) / 255
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    a, b, c = gaussian_filter(gx * gx, 1.5), gaussian_filter(gx * gy, 1.5), gaussian_filter(gy * gy, 1.5)
    minimum = np.maximum(.5 * (a + c - np.sqrt((a - c)**2 + 4 * b**2)), 0)
    return minimum / (minimum + floor)


def flow_diagnostics(state: CanonicalState, field: AnatomicalField) -> FlowDiagnostics:
    a = cv2.cvtColor(state.image_a, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(state.image_b, cv2.COLOR_RGB2GRAY)
    args = (.5, 4, 25, 4, 7, 1.5, 0)
    forward = cv2.calcOpticalFlowFarneback(a, b, None, *args)
    backward = cv2.calcOpticalFlowFarneback(b, a, None, *args)
    yy, xx = np.indices(a.shape, dtype=np.float32)
    qx, qy = xx + forward[..., 0], yy + forward[..., 1]
    def sample(array: np.ndarray) -> np.ndarray:
        return cv2.remap(array.astype(np.float32), qx, qy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    reverse = sample(backward)
    cycle = np.linalg.norm(forward + reverse, axis=-1)
    texture = np.minimum(_texture(a, state.config.texture_floor), sample(_texture(b, state.config.texture_floor)))
    target_valid = sample(state.masks_b["face"] & state.valid_source_b) >= 1 - 1e-6
    source_valid = state.masks_a["face"] & state.valid_source_a
    inside = (qx >= 0) & (qx <= a.shape[1] - 1) & (qy >= 0) & (qy <= a.shape[0] - 1)
    admissible = source_valid & target_valid & inside & (texture >= .1) & (cycle <= state.config.fb_threshold_px)
    disagreement = np.linalg.norm(forward - field.uv, axis=-1)
    concordant = admissible & field.valid & (disagreement <= state.config.sparse_dense_tolerance_ipd * state.normalization_span_px)
    photo = np.abs(a.astype(float) / 255 - sample(b.astype(float) / 255))
    return FlowDiagnostics(forward, cycle, texture, photo, disagreement, admissible, concordant)


def build_field(state: CanonicalState) -> AnatomicalField:
    from dataclasses import replace
    # Extrapolated/off-image predictions must not deform otherwise observed anatomy.
    ids = np.flatnonzero(state.observation_a.in_bounds & state.observation_b.in_bounds)
    field = affine_triangle_field(state.xy_a[ids], state.xy_b[ids], state.image_a.shape[:2],
                                  state.masks_a["face"] & state.valid_source_a,
                                  state.masks_b["face"] & state.valid_source_b,
                                  state.config.max_triangle_condition)
    # Export triangle vertices in the original, shared landmark index system.
    return replace(field, triangles=ids[field.triangles])
