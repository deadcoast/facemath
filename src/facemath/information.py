from __future__ import annotations
import zlib
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def shannon_entropy(values: np.ndarray, bins: int = 128) -> float:
    x = np.asarray(values, float).ravel()
    h, _ = np.histogram(x, bins=bins)
    p = h[h > 0].astype(float); p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def compressed_length(data: bytes, level: int = 9) -> int:
    return len(zlib.compress(data, level))


def normalized_compression_distance(a: bytes, b: bytes) -> float:
    ca, cb = compressed_length(a), compressed_length(b)
    cab = compressed_length(a + b)
    return float((cab - min(ca, cb)) / max(ca, cb, 1))


def algorithmic_mutual_information_proxy(a: bytes, b: bytes) -> float:
    return float(compressed_length(a) + compressed_length(b) - compressed_length(a + b))


def mutual_information_matrix(x: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    out = np.zeros((x.shape[1], y.shape[1]))
    for j in range(y.shape[1]):
        out[:, j] = mutual_info_regression(x, y[:, j], random_state=random_state)
    return out


def mdl_score(model_bytes: int, residual_bytes: int) -> int:
    return int(model_bytes + residual_bytes)
