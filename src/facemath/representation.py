from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA


@dataclass
class RepresentationSnapshot:
    layers: dict[str, np.ndarray] = field(default_factory=dict)
    attentions: dict[str, np.ndarray] = field(default_factory=dict)
    logits: np.ndarray | None = None
    token_labels: list[str] | None = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    hsic = np.linalg.norm(x.T @ y, "fro") ** 2
    nx = np.linalg.norm(x.T @ x, "fro")
    ny = np.linalg.norm(y.T @ y, "fro")
    return float(hsic / (nx * ny + 1e-12))


def rdm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return 1 - xn @ xn.T


def rsa(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rdm(x), rdm(y)
    iu = np.triu_indices_from(rx, 1)
    return float(spearmanr(rx[iu], ry[iu]).statistic)


def svcca(x: np.ndarray, y: np.ndarray, n_components: int | None = None) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = n_components or min(x.shape[1], y.shape[1], max(1, x.shape[0] - 1))
    n = min(n, x.shape[1], y.shape[1], max(1, x.shape[0] - 1))
    cca = CCA(n_components=n).fit(x, y)
    xa, ya = cca.transform(x, y)
    return np.array([np.corrcoef(xa[:, i], ya[:, i])[0, 1] for i in range(n)])


def attention_laplacian(attn: np.ndarray) -> np.ndarray:
    a = np.asarray(attn, float)
    if a.ndim > 2:
        a = a.mean(tuple(range(a.ndim - 2)))
    a = 0.5 * (a + a.T)
    d = a.sum(1)
    return np.diag(d) - a


def attention_spectral_distance(
    a: np.ndarray, b: np.ndarray, k: int | None = None
) -> float:
    ea = np.sort(np.linalg.eigvalsh(attention_laplacian(a)))
    eb = np.sort(np.linalg.eigvalsh(attention_laplacian(b)))
    m = min(len(ea), len(eb), k or 10**9)
    return float(np.linalg.norm(ea[:m] - eb[:m]) / np.sqrt(m))


def logit_contrast(logits: np.ndarray, positive: int, negative: int) -> float:
    l = np.asarray(logits, float)
    return float(l[..., positive].mean() - l[..., negative].mean())


class ModelAdapter:
    """Boundary for model-specific capture/intervention.

    Implementations should return RepresentationSnapshot from real model telemetry. The
    facemath core never invents activations when none are supplied.
    """

    def capture(
        self, inputs: Any
    ) -> RepresentationSnapshot:  # pragma: no cover - interface
        raise NotImplementedError

    def intervene(
        self, inputs: Any, patch: Mapping[str, np.ndarray]
    ) -> RepresentationSnapshot:  # pragma: no cover
        raise NotImplementedError
