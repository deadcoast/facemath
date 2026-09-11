from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass(frozen=True)
class FlowField:
    uv: np.ndarray  # [H,W,2]

    @property
    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.uv, axis=-1)


def optical_flow_farneback(gray_a: np.ndarray, gray_b: np.ndarray) -> FlowField:
    a = np.asarray(gray_a, dtype=np.float32)
    b = np.asarray(gray_b, dtype=np.float32)
    if a.max() <= 1.0:
        a = a * 255.0
    if b.max() <= 1.0:
        b = b * 255.0
    flow = cv2.calcOpticalFlowFarneback(
        a.astype(np.uint8), b.astype(np.uint8), None, 0.5, 5, 21, 5, 7, 1.5, 0
    )
    return FlowField(flow.astype(np.float64))


def flow_jacobian(flow: FlowField) -> np.ndarray:
    u, v = flow.uv[..., 0], flow.uv[..., 1]
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    j = np.empty(flow.uv.shape[:2] + (2, 2), float)
    j[..., 0, 0] = du_dx
    j[..., 0, 1] = du_dy
    j[..., 1, 0] = dv_dx
    j[..., 1, 1] = dv_dy
    return j


def strain_tensor(flow: FlowField) -> np.ndarray:
    j = flow_jacobian(flow)
    return 0.5 * (j + np.swapaxes(j, -1, -2))


def strain_energy(flow: FlowField, mask: np.ndarray | None = None) -> float:
    e = strain_tensor(flow)
    sq = np.sum(e * e, axis=(-1, -2))
    if mask is not None:
        sq = sq[np.asarray(mask, bool)]
    return float(np.mean(sq))


def smooth_deformation_energy(
    flow: FlowField,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    u = flow.uv
    mag = np.sum(u * u, axis=-1)
    first = sum(
        np.sum(g * g, axis=-1) for g in (np.gradient(u, axis=0), np.gradient(u, axis=1))
    )
    dyy = np.gradient(np.gradient(u, axis=0), axis=0)
    dxx = np.gradient(np.gradient(u, axis=1), axis=1)
    second = np.sum(dyy * dyy, axis=-1) + np.sum(dxx * dxx, axis=-1)
    total = alpha * mag + beta * first + gamma * second
    if mask is not None:
        total = total[np.asarray(mask, bool)]
    return float(np.mean(total))


def fit_thin_plate_spline(src: np.ndarray, dst: np.ndarray, smoothing: float = 0.0):
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    return RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=smoothing)


def tps_grid(
    src: np.ndarray,
    dst: np.ndarray,
    width: int,
    height: int,
    step: int = 8,
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    fn = fit_thin_plate_spline(src, dst, smoothing)
    ys, xs = np.mgrid[0:height:step, 0:width:step]
    q = np.c_[xs.ravel(), ys.ravel()]
    mapped = fn(q)
    flow = (mapped - q).reshape(
        len(range(0, height, step)), len(range(0, width, step)), 2
    )
    return q.reshape(flow.shape[:2] + (2,)), flow


def tps_bending_proxy(flow: np.ndarray) -> float:
    dxx = np.gradient(np.gradient(flow, axis=1), axis=1)
    dyy = np.gradient(np.gradient(flow, axis=0), axis=0)
    dxy = np.gradient(np.gradient(flow, axis=0), axis=1)
    e = np.sum(dxx * dxx + 2 * dxy * dxy + dyy * dyy, axis=-1)
    return float(np.mean(e))


def local_affine_strain(
    src: np.ndarray, dst: np.ndarray, neighborhoods: list[list[int]]
) -> np.ndarray:
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    out = []
    for ids in neighborhoods:
        x, y = src[ids], dst[ids]
        xc, yc = x.mean(0), y.mean(0)
        xx = x - xc
        yy = y - yc
        a, *_ = np.linalg.lstsq(xx, yy, rcond=None)
        # row-vector convention: yy ≈ xx @ a
        _, svals, _ = np.linalg.svd(a)
        out.append([*svals, np.linalg.norm(a - np.eye(2))])
    return np.asarray(out)
