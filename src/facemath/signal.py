from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel
from skimage.transform import pyramid_laplacian
from skimage.filters import gabor


def crossfade(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * np.asarray(a, float) + (1.0 - alpha) * np.asarray(b, float)


def difference_field(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, float) - np.asarray(b, float)


def gradient_field(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(gray, float)
    gx = sobel(g, axis=1)
    gy = sobel(g, axis=0)
    mag = np.hypot(gx, gy)
    ori = np.arctan2(gy, gx)
    return gx, gy, mag, ori


def structure_tensor(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    gx, gy, _, _ = gradient_field(gray)
    jxx = gaussian_filter(gx * gx, sigma)
    jxy = gaussian_filter(gx * gy, sigma)
    jyy = gaussian_filter(gy * gy, sigma)
    return np.stack([jxx, jxy, jxy, jyy], axis=-1).reshape(gray.shape + (2, 2))


def structure_tensor_distance(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    ja, jb = structure_tensor(a, sigma), structure_tensor(b, sigma)
    return float(np.sqrt(np.mean((ja - jb) ** 2)))


def phase_correlation(a: np.ndarray, b: np.ndarray) -> tuple[tuple[float, float], float]:
    af = np.asarray(a, np.float32)
    bf = np.asarray(b, np.float32)
    shift, response = cv2.phaseCorrelate(af, bf)
    return (float(shift[0]), float(shift[1])), float(response)


def fourier_phase_distance(a: np.ndarray, b: np.ndarray) -> float:
    fa = np.fft.fft2(np.asarray(a, float))
    fb = np.fft.fft2(np.asarray(b, float))
    pa = np.angle(fa)
    pb = np.angle(fb)
    d = np.angle(np.exp(1j * (pa - pb)))
    return float(np.sqrt(np.mean(d * d)))


def laplacian_pyramid_distance(a: np.ndarray, b: np.ndarray, max_layer: int = 4) -> np.ndarray:
    pa = list(pyramid_laplacian(a, max_layer=max_layer, downscale=2, channel_axis=None))
    pb = list(pyramid_laplacian(b, max_layer=max_layer, downscale=2, channel_axis=None))
    m = min(len(pa), len(pb))
    return np.array([np.sqrt(np.mean((pa[i] - pb[i]) ** 2)) for i in range(m)])


def gabor_scattering_signature(gray: np.ndarray, frequencies: tuple[float, ...] = (0.08, 0.16, 0.32), orientations: int = 6) -> np.ndarray:
    """Deterministic first/second-order scattering-like signature built from Gabor modulus cascades."""
    img = np.asarray(gray, float)
    # Bound computational cost while preserving multiscale structure.
    h, w = img.shape
    scale = min(1.0, 256.0 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(8, int(round(w*scale))), max(8, int(round(h*scale)))), interpolation=cv2.INTER_AREA)
    feats = []
    first_maps = []
    for f in frequencies:
        for k in range(orientations):
            theta = np.pi * k / orientations
            real, imag = gabor(img, frequency=f, theta=theta)
            mod = np.hypot(real, imag)
            first_maps.append(mod)
            feats.extend([mod.mean(), mod.std()])
    # second-order: filter modulus fields at coarser frequencies, summarized globally
    for mod in first_maps[::max(1, len(first_maps)//8)]:
        real, imag = gabor(mod, frequency=frequencies[0], theta=0)
        m2 = np.hypot(real, imag)
        feats.extend([m2.mean(), m2.std()])
    return np.asarray(feats)


def scattering_distance(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = gabor_scattering_signature(a), gabor_scattering_signature(b)
    return float(np.linalg.norm(sa - sb) / np.sqrt(len(sa)))


def residual_entropy(residual: np.ndarray, bins: int = 128) -> float:
    r = np.asarray(residual, float).ravel()
    hist, _ = np.histogram(r, bins=bins, density=False)
    p = hist[hist > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))
