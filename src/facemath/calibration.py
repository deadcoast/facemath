from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from .core import Direction, MetricResult
from .statistics import empirical_similarity_percentile


@dataclass(frozen=True)
class BaselineCohort:
    metric_name: str
    values: np.ndarray
    cohort_name: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.values, float)
        if v.ndim != 1 or len(v) < 3:
            raise ValueError(
                "baseline values must be a 1D sample with at least 3 observations"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("baseline values must be finite")
        object.__setattr__(self, "values", v)

    @property
    def fingerprint(self) -> str:
        h = sha256()
        h.update(self.metric_name.encode())
        h.update(self.cohort_name.encode())
        h.update(np.ascontiguousarray(self.values).tobytes())
        h.update(json.dumps(dict(self.metadata), sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]

    def percentile(self, value: float, direction: Direction) -> float:
        if direction == Direction.LOWER_IS_MORE_SIMILAR:
            return empirical_similarity_percentile(value, self.values)
        if direction == Direction.HIGHER_IS_MORE_SIMILAR:
            return float(np.mean(self.values <= value) * 100.0)
        raise ValueError("descriptive metrics cannot be calibrated")

    def bootstrap_interval(
        self,
        value: float,
        direction: Direction,
        draws: int = 1000,
        seed: int = 0,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        n = len(self.values)
        ps = np.empty(draws, float)
        for i in range(draws):
            sample = self.values[rng.integers(0, n, n)]
            if direction == Direction.LOWER_IS_MORE_SIMILAR:
                ps[i] = 100.0 - float(np.mean(sample <= value) * 100.0)
            elif direction == Direction.HIGHER_IS_MORE_SIMILAR:
                ps[i] = float(np.mean(sample <= value) * 100.0)
            else:
                raise ValueError("descriptive metrics cannot be calibrated")
        return tuple(np.quantile(ps, [alpha / 2, 1 - alpha / 2]).tolist())


@dataclass
class BaselineRegistry:
    cohorts: dict[str, BaselineCohort] = field(default_factory=dict)

    @property
    def distances(self) -> dict[str, np.ndarray]:
        return {k: v.values for k, v in self.cohorts.items()}

    def add(
        self,
        metric_name: str,
        values: np.ndarray,
        cohort_name: str = "default",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.cohorts[metric_name] = BaselineCohort(
            metric_name, np.asarray(values, float), cohort_name, metadata or {}
        )

    def calibrate(self, result: MetricResult, bootstrap_draws: int = 0) -> MetricResult:
        cohort = self.cohorts.get(result.name)
        if (
            cohort is None
            or not np.isscalar(result.value)
            or result.direction == Direction.DESCRIPTIVE
        ):
            return result
        result.calibrated_percentile = cohort.percentile(
            float(result.value), result.direction
        )
        if bootstrap_draws > 0:
            lo, hi = cohort.bootstrap_interval(
                float(result.value), result.direction, bootstrap_draws
            )
            result.uncertainty = float((hi - lo) / 2)
            result.tags = tuple(result.tags) + (
                f"baseline:{cohort.cohort_name}",
                f"baseline_fingerprint:{cohort.fingerprint}",
                f"bootstrap95:{lo:.3f}-{hi:.3f}",
            )
        else:
            result.tags = tuple(result.tags) + (
                f"baseline:{cohort.cohort_name}",
                f"baseline_fingerprint:{cohort.fingerprint}",
            )
        return result

    def calibrate_ledger(self, ledger, bootstrap_draws: int = 0):
        for result in ledger.results.values():
            self.calibrate(result, bootstrap_draws)
        return ledger
