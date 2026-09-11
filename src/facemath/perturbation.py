from __future__ import annotations
from itertools import combinations
from typing import Callable, Sequence
import math
import numpy as np


def occlusion_effect(image: np.ndarray, scorer: Callable[[np.ndarray], float], masks: Sequence[np.ndarray], fill: float = 0.0) -> np.ndarray:
    base = scorer(image)
    out = []
    for mask in masks:
        x = np.array(image, copy=True)
        m = np.asarray(mask, bool)
        if x.ndim == 3:
            x[m, :] = fill
        else:
            x[m] = fill
        out.append(base - scorer(x))
    return np.asarray(out)


def finite_difference_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps; xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2 * eps)
    return g


def finite_difference_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    n = len(x); h = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = eps
        h[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / eps**2
        for j in range(i + 1, n):
            ej = np.zeros(n); ej[j] = eps
            val = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps**2)
            h[i, j] = h[j, i] = val
    return h


def shapley_values(n_features: int, value_fn: Callable[[frozenset[int]], float]) -> np.ndarray:
    phi = np.zeros(n_features)
    fact = math.factorial
    denom = fact(n_features)
    all_features = set(range(n_features))
    for i in range(n_features):
        rest = list(all_features - {i})
        for r in range(len(rest) + 1):
            weight = fact(r) * fact(n_features - r - 1) / denom
            for subset in combinations(rest, r):
                s = frozenset(subset)
                phi[i] += weight * (value_fn(s | {i}) - value_fn(s))
    return phi


def pair_interaction(i: int, j: int, base: frozenset[int], value_fn: Callable[[frozenset[int]], float]) -> float:
    return float(value_fn(base | {i, j}) - value_fn(base | {i}) - value_fn(base | {j}) + value_fn(base))


def fisher_information(score_gradients: np.ndarray) -> np.ndarray:
    g = np.asarray(score_gradients, float)
    return (g.T @ g) / max(len(g), 1)
