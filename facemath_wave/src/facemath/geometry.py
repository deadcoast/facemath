from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist

from .core import LandmarkSet


@dataclass(frozen=True)
class ProcrustesResult:
    aligned_a: np.ndarray
    aligned_b: np.ndarray
    rotation: np.ndarray
    scale_a: float
    scale_b: float
    translation_a: np.ndarray
    translation_b: np.ndarray
    distance: float


def center_scale(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    p = np.asarray(points, dtype=float)
    center = p.mean(axis=0)
    centered = p - center
    scale = np.linalg.norm(centered)
    if scale <= 1e-12:
        raise ValueError("Degenerate point configuration")
    return centered / scale, center, scale


def generalized_procrustes(a: np.ndarray, b: np.ndarray) -> ProcrustesResult:
    an, ac, ascale = center_scale(a)
    bn, bc, bscale = center_scale(b)
    r, _ = orthogonal_procrustes(an, bn)
    aa = an @ r
    d = float(np.linalg.norm(aa - bn))
    return ProcrustesResult(aa, bn, r, ascale, bscale, ac, bc, d)


def similarity_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return 3x3 homogeneous similarity transform mapping 2D points a -> b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("a and b must both be [N,2]")
    ca, cb = a.mean(0), b.mean(0)
    xa, xb = a - ca, b - cb
    h = xa.T @ xb
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    denom = np.sum(xa * xa)
    s = float(np.trace((xa @ r.T).T @ xb) / denom)
    t = cb - s * (r @ ca)
    m = np.eye(3)
    m[:2, :2] = s * r
    m[:2, 2] = t
    pred = apply_homography(a, m)
    rmse = float(np.sqrt(np.mean(np.sum((pred - b) ** 2, axis=1))))
    return m, s, rmse


def apply_homography(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    h = np.c_[p, np.ones(len(p))]
    q = h @ np.asarray(m, dtype=float).T
    return q[:, :2] / q[:, 2:3]


def intrinsic_coordinates(points: np.ndarray, left_eye: int, right_eye: int) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    l, r = p[left_eye], p[right_eye]
    origin = (l + r) / 2
    x_axis = r - l
    scale = np.linalg.norm(x_axis)
    if scale <= 1e-12:
        raise ValueError("Eye anchors coincide")
    x_axis /= scale
    y_axis = np.array([-x_axis[1], x_axis[0]])
    basis = np.stack([x_axis, y_axis], axis=1)
    return ((p - origin) @ basis) / scale


def pairwise_distance_matrix(points: np.ndarray, scale: float | None = None) -> np.ndarray:
    d = cdist(points, points)
    return d if scale is None else d / float(scale)


def distance_matrix_error(a: np.ndarray, b: np.ndarray, scale_a: float | None = None, scale_b: float | None = None) -> float:
    da = pairwise_distance_matrix(a, scale_a)
    db = pairwise_distance_matrix(b, scale_b)
    return float(np.linalg.norm(da - db, ord="fro") / np.sqrt(da.size))


def angle(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    u, v = p - q, r - q
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-12:
        return np.nan
    return float(np.arccos(np.clip((u @ v) / denom, -1, 1)))


def angle_signature(points: np.ndarray, triples: Sequence[tuple[int, int, int]]) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return np.array([angle(p[i], p[j], p[k]) for i, j, k in triples])


def angle_error(a: np.ndarray, b: np.ndarray, triples: Sequence[tuple[int, int, int]]) -> float:
    aa = angle_signature(a, triples)
    bb = angle_signature(b, triples)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((aa[mask] - bb[mask]) ** 2)))


def epsilon_isometry_distortion(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    da = pairwise_distance_matrix(a)
    db = pairwise_distance_matrix(b)
    delta = da - db
    return float(np.max(np.abs(delta))), delta


def rigid_residuals(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    m, _, rmse = similarity_transform(a, b)
    pred = apply_homography(a, m)
    return pred, b - pred, rmse


def landmark_scale(points: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(points[i] - points[j]))


def relational_graph(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Weighted k-NN adjacency over landmarks; weights are inverse normalized distance."""
    d = cdist(points, points)
    n = len(points)
    a = np.zeros((n, n), float)
    for i in range(n):
        idx = np.argsort(d[i])[1 : min(k + 1, n)]
        for j in idx:
            w = 1.0 / max(d[i, j], 1e-12)
            a[i, j] = a[j, i] = max(a[i, j], w)
    if a.max() > 0:
        a /= a.max()
    return a


def graph_laplacian(adjacency: np.ndarray, normalized: bool = True) -> np.ndarray:
    a = np.asarray(adjacency, dtype=float)
    degree = a.sum(1)
    if not normalized:
        return np.diag(degree) - a
    inv = np.zeros_like(degree)
    mask = degree > 1e-12
    inv[mask] = degree[mask] ** -0.5
    return np.eye(len(a)) - np.diag(inv) @ a @ np.diag(inv)


def graph_spectrum(adjacency: np.ndarray, k: int | None = None) -> np.ndarray:
    vals = np.linalg.eigvalsh(graph_laplacian(adjacency))
    vals = np.sort(np.real(vals))
    return vals if k is None else vals[:k]


def spectral_graph_distance(a: np.ndarray, b: np.ndarray, k: int | None = None) -> float:
    sa, sb = graph_spectrum(a, k), graph_spectrum(b, k)
    m = min(len(sa), len(sb))
    return float(np.linalg.norm(sa[:m] - sb[:m]) / np.sqrt(m))
