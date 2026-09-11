from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import Direction, MetricResult
from .statistics import empirical_similarity_percentile


@dataclass
class BaselineRegistry:
    distances: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, metric_name: str, values: np.ndarray) -> None:
        self.distances[metric_name] = np.asarray(values, float)

    def calibrate(self, result: MetricResult) -> MetricResult:
        if result.name not in self.distances or not np.isscalar(result.value):
            return result
        vals = self.distances[result.name]
        if result.direction == Direction.LOWER_IS_MORE_SIMILAR:
            pct = empirical_similarity_percentile(float(result.value), vals)
        elif result.direction == Direction.HIGHER_IS_MORE_SIMILAR:
            pct = float(np.mean(vals <= float(result.value)) * 100)
        else:
            return result
        result.calibrated_percentile = pct
        return result
