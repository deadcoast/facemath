from __future__ import annotations
import numpy as np
from scipy.interpolate import splprep, splev


def resample_closed_contour(points: np.ndarray, n: int = 128) -> np.ndarray:
    p = np.asarray(points, float)
    closed = np.vstack([p, p[0]])
    tck, _ = splprep([closed[:, 0], closed[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, n, endpoint=False)
    x, y = splev(u, tck)
    return np.c_[x, y]


def contour_curvature(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    x, y = p[:, 0], p[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom = np.maximum(denom, 1e-12)
    return (dx * ddy - dy * ddx) / denom


def curvature_distance(a: np.ndarray, b: np.ndarray, samples: int = 128) -> float:
    aa = resample_closed_contour(a, samples)
    bb = resample_closed_contour(b, samples)
    ka, kb = contour_curvature(aa), contour_curvature(bb)
    return float(np.sqrt(np.mean((ka - kb) ** 2)))


def fourier_descriptors(points: np.ndarray, harmonics: int = 16) -> np.ndarray:
    p = resample_closed_contour(points, max(64, 4 * harmonics))
    z = p[:, 0] + 1j * p[:, 1]
    z = z - z.mean()
    f = np.fft.fft(z)
    # normalize translation, scale, approximate rotation/phase
    scale = np.abs(f[1]) if np.abs(f[1]) > 1e-12 else np.linalg.norm(f)
    f /= scale
    phase = np.angle(f[1])
    f *= np.exp(-1j * phase)
    idx = list(range(1, harmonics + 1)) + list(range(-harmonics, 0))
    return f[idx]


def fourier_descriptor_distance(a: np.ndarray, b: np.ndarray, harmonics: int = 16) -> float:
    fa, fb = fourier_descriptors(a, harmonics), fourier_descriptors(b, harmonics)
    return float(np.linalg.norm(fa - fb) / np.sqrt(len(fa)))
