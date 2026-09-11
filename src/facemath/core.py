from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


class Direction(str, Enum):
    """How to interpret a scalar metric."""

    HIGHER_IS_MORE_SIMILAR = "higher_is_more_similar"
    LOWER_IS_MORE_SIMILAR = "lower_is_more_similar"
    DESCRIPTIVE = "descriptive"


@dataclass(frozen=True)
class Provenance:
    operation: str
    inputs: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    implementation: str = "facemath"
    timestamp_unix: float = field(default_factory=time.time)


@dataclass
class MetricResult:
    name: str
    value: float | np.ndarray | list[float] | dict[str, Any]
    unit: str
    direction: Direction
    domain: str
    description: str
    provenance: Provenance
    calibrated_percentile: float | None = None
    uncertainty: float | None = None
    tags: tuple[str, ...] = ()

    def scalar(self) -> float:
        if np.isscalar(self.value):
            return float(self.value)
        raise TypeError(f"Metric {self.name!r} is not scalar")


@dataclass
class EvidenceLedger:
    results: dict[str, MetricResult] = field(default_factory=dict)

    def add(self, result: MetricResult) -> None:
        if result.name in self.results:
            raise KeyError(f"Metric already present: {result.name}")
        self.results[result.name] = result

    def extend(self, results: Iterable[MetricResult]) -> None:
        for result in results:
            self.add(result)

    def by_domain(self, domain: str) -> list[MetricResult]:
        return [r for r in self.results.values() if r.domain == domain]

    def to_jsonable(self) -> dict[str, Any]:
        def clean(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, dict):
                return {str(k): clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            return clean(asdict(v)) if hasattr(v, "__dataclass_fields__") else v

        return clean(self.results)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_jsonable(), indent=2), encoding="utf-8"
        )


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    timestamp: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("Frame.rgb must have shape [H, W, 3]")

    @property
    def gray(self) -> np.ndarray:
        x = self.rgb.astype(np.float64)
        return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]) / 255.0

    @property
    def digest(self) -> str:
        return sha256(np.ascontiguousarray(self.rgb).tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class LandmarkSet:
    xy: np.ndarray
    names: tuple[str, ...] | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        object.__setattr__(self, "xy", xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("LandmarkSet.xy must have shape [N,2]")
        if self.names is not None and len(self.names) != len(xy):
            raise ValueError("names length must match landmarks")
        if self.confidence is not None and len(self.confidence) != len(xy):
            raise ValueError("confidence length must match landmarks")

    def index(self, name: str) -> int:
        if self.names is None:
            raise KeyError("Landmarks are unnamed")
        return self.names.index(name)


@dataclass(frozen=True)
class FacePair:
    a: Frame
    b: Frame
    landmarks_a: LandmarkSet | None = None
    landmarks_b: LandmarkSet | None = None


@dataclass(frozen=True)
class CanonicalPair:
    a: Frame
    b: Frame
    landmarks_a: LandmarkSet | None
    landmarks_b: LandmarkSet | None
    transform_a_to_b: np.ndarray | None = None
    registration_rmse: float | None = None


@dataclass
class AnalysisBundle:
    pair: FacePair
    canonical: CanonicalPair | None = None
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    artifacts: dict[str, Any] = field(default_factory=dict)
