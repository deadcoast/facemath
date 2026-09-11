from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline


def reflect_points(points: np.ndarray, midline_x: float) -> np.ndarray:
    p = np.asarray(points, dtype=float).copy()
    p[:, 0] = 2 * midline_x - p[:, 0]
    return p


def image_reflection_error(gray: np.ndarray, midline_x: float | None = None) -> float:
    img = np.asarray(gray, dtype=float)
    w = img.shape[1]
    m = (w - 1) / 2 if midline_x is None else float(midline_x)
    xs = np.arange(w)
    xr = np.rint(2 * m - xs).astype(int)
    mask = (xr >= 0) & (xr < w)
    diff = img[:, xs[mask]] - img[:, xr[mask]]
    return float(np.sqrt(np.mean(diff * diff)))


def paired_symmetry_signature(
    points: np.ndarray, pairs: list[tuple[int, int]], midline_x: float | None = None
) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    m = float(np.median(p[:, 0])) if midline_x is None else float(midline_x)
    rows = []
    for l, r in pairs:
        pl, pr = p[l], p[r]
        mirrored_l = np.array([2 * m - pl[0], pl[1]])
        delta = pr - mirrored_l
        rows.append([delta[0], delta[1], np.linalg.norm(delta)])
    return np.asarray(rows)


def symmetry_topology_distance(
    a: np.ndarray, b: np.ndarray, pairs: list[tuple[int, int]]
) -> float:
    sa = paired_symmetry_signature(a, pairs)
    sb = paired_symmetry_signature(b, pairs)
    return float(np.sqrt(np.mean((sa - sb) ** 2)))


def midline_deviation_signature(
    points: np.ndarray, indices: list[int], samples: int = 64
) -> np.ndarray:
    p = np.asarray(points, dtype=float)[indices]
    order = np.argsort(p[:, 1])
    p = p[order]
    ys, xs = p[:, 1], p[:, 0]
    if len(np.unique(ys)) < 4:
        x_interp = np.interp(np.linspace(ys.min(), ys.max(), samples), ys, xs)
    else:
        spline = UnivariateSpline(ys, xs, s=0, k=min(3, len(ys) - 1))
        yq = np.linspace(ys.min(), ys.max(), samples)
        x_interp = spline(yq)
    return x_interp - np.mean(x_interp)


def midline_signature_distance(
    a: np.ndarray, b: np.ndarray, indices: list[int]
) -> float:
    sa = midline_deviation_signature(a, indices)
    sb = midline_deviation_signature(b, indices)
    return float(np.sqrt(np.mean((sa - sb) ** 2)))
