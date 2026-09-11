from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import percentileofscore
from sklearn.decomposition import PCA


@dataclass
class ShapeModel:
    mean_: np.ndarray
    components_: np.ndarray
    variances_: np.ndarray
    original_shape: tuple[int, int]

    def project(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, float).ravel() - self.mean_
        return x @ self.components_.T

    def reconstruct(self, coeffs: np.ndarray) -> np.ndarray:
        x = self.mean_ + np.asarray(coeffs) @ self.components_
        return x.reshape(self.original_shape)


def fit_shape_model(shapes: np.ndarray, n_components: int | None = None) -> ShapeModel:
    x = np.asarray(shapes, float)
    orig = x.shape[1:]
    flat = x.reshape(len(x), -1)
    pca = PCA(n_components=n_components).fit(flat)
    return ShapeModel(pca.mean_, pca.components_, pca.explained_variance_, orig)


def mahalanobis_distance(a: np.ndarray, b: np.ndarray, covariance: np.ndarray, ridge: float = 1e-8) -> float:
    d = np.asarray(a, float).ravel() - np.asarray(b, float).ravel()
    cov = np.asarray(covariance, float) + ridge * np.eye(len(d))
    inv = np.linalg.pinv(cov)
    return float(np.sqrt(d @ inv @ d))


def empirical_similarity_percentile(distance: float, baseline_distances: np.ndarray) -> float:
    """Percent of baseline pairs that are less similar (have larger distance)."""
    p = percentileofscore(np.asarray(baseline_distances, float), distance, kind="mean")
    return float(100.0 - p)


def nearest_neighbor_tail(distance: float, baseline_distances: np.ndarray) -> float:
    b = np.asarray(baseline_distances, float)
    return float(np.mean(b <= distance))


def covariance_whiten(features: np.ndarray, ridge: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, float)
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False) + ridge * np.eye(x.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    w = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, ridge))) @ vecs.T
    return (x - mean) @ w, mean, w
