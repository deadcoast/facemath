from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment


def zero_dim_persistence(points: np.ndarray) -> np.ndarray:
    """0D Vietoris-Rips persistence death times via Kruskal MST."""
    p = np.asarray(points, float)
    d = squareform(pdist(p))
    n = len(p)
    edges = sorted((d[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    deaths = []
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
            deaths.append(float(w))
            if len(deaths) == n - 1:
                break
    return np.asarray(deaths)


def persistence_distance_0d(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.sort(zero_dim_persistence(a))
    pb = np.sort(zero_dim_persistence(b))
    m = min(len(pa), len(pb))
    return float(np.sqrt(np.mean((pa[:m] - pb[:m]) ** 2)))


def relational_assignment(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Feature assignment using pairwise position cost; returns assignment and average cost."""
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    i, j = linear_sum_assignment(cost)
    return np.c_[i, j], float(cost[i, j].mean())


def gromov_relational_distortion(a: np.ndarray, b: np.ndarray, assignment: np.ndarray | None = None) -> float:
    """GW-like relational distortion under a discrete correspondence.

    Exact entropic GW requires a dedicated OT solver; this computes the core relational
    distortion objective once a one-to-one coupling is fixed.
    """
    if assignment is None:
        assignment, _ = relational_assignment(a, b)
    ia, ib = assignment[:, 0], assignment[:, 1]
    da = squareform(pdist(np.asarray(a)[ia]))
    db = squareform(pdist(np.asarray(b)[ib]))
    return float(np.mean((da - db) ** 2))
