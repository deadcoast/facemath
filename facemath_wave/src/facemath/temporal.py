from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import svd
from sklearn.cross_decomposition import CCA


def trajectory_velocity(z: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(z, float), axis=0)


def trajectory_acceleration(z: np.ndarray) -> np.ndarray:
    return np.gradient(trajectory_velocity(z), axis=0)


def path_energy(z: np.ndarray) -> float:
    v = trajectory_velocity(z)
    return float(np.mean(np.sum(v * v, axis=-1)))


def path_curvature_energy(z: np.ndarray) -> float:
    a = trajectory_acceleration(z)
    return float(np.mean(np.sum(a * a, axis=-1)))


def slow_feature_analysis(x: np.ndarray, n_components: int | None = None, ridge: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    x = x - x.mean(0)
    cov = np.cov(x, rowvar=False) + ridge * np.eye(x.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    white = vecs @ np.diag(1 / np.sqrt(np.maximum(vals, ridge))) @ vecs.T
    z = x @ white
    dz = np.diff(z, axis=0)
    dcov = np.cov(dz, rowvar=False)
    dvals, dvecs = np.linalg.eigh(dcov)
    order = np.argsort(dvals)
    if n_components is not None:
        order = order[:n_components]
    components = white @ dvecs[:, order]
    return x @ components, dvals[order]


def robust_pca_pcp(x: np.ndarray, lam: float | None = None, mu: float | None = None, max_iter: int = 500, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """Principal Component Pursuit via inexact augmented Lagrangian."""
    x = np.asarray(x, float)
    m, n = x.shape
    lam = 1 / np.sqrt(max(m, n)) if lam is None else lam
    norm2 = np.linalg.norm(x, 2)
    norm_inf = np.max(np.abs(x)) / lam
    dual_norm = max(norm2, norm_inf)
    y = x / max(dual_norm, 1e-12)
    mu = (1.25 / max(norm2, 1e-12)) if mu is None else mu
    mu_bar = mu * 1e7
    rho = 1.5
    l = np.zeros_like(x)
    s = np.zeros_like(x)
    xnorm = max(np.linalg.norm(x, 'fro'), 1e-12)
    for _ in range(max_iter):
        # low-rank prox
        u, sig, vt = np.linalg.svd(x - s + y / mu, full_matrices=False)
        sig2 = np.maximum(sig - 1 / mu, 0)
        rank = np.sum(sig2 > 0)
        l = (u[:, :rank] * sig2[:rank]) @ vt[:rank] if rank else np.zeros_like(x)
        # sparse prox
        q = x - l + y / mu
        s = np.sign(q) * np.maximum(np.abs(q) - lam / mu, 0)
        z = x - l - s
        y = y + mu * z
        mu = min(mu * rho, mu_bar)
        if np.linalg.norm(z, 'fro') / xnorm < tol:
            break
    return l, s


def dynamic_mode_decomposition(x: np.ndarray, rank: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    x0, x1 = x[:-1].T, x[1:].T
    u, s, vt = svd(x0, full_matrices=False)
    r = min(len(s), rank or len(s))
    ur, sr, vr = u[:, :r], s[:r], vt[:r].T
    a_tilde = ur.T @ x1 @ vr @ np.diag(1 / np.maximum(sr, 1e-12))
    eigvals, w = np.linalg.eig(a_tilde)
    modes = x1 @ vr @ np.diag(1 / np.maximum(sr, 1e-12)) @ w
    return eigvals, modes, a_tilde


def canonical_correlation(x: np.ndarray, y: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = min(n_components, x.shape[1], y.shape[1])
    cca = CCA(n_components=n).fit(x, y)
    xa, ya = cca.transform(x, y)
    return np.array([np.corrcoef(xa[:, i], ya[:, i])[0, 1] for i in range(n)])
