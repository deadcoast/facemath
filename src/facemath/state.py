from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .core import Frame, LandmarkSet


class SupportLevel(str, Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class Region:
    name: str
    mask: np.ndarray
    weight: float = 1.0
    parent: str | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.mask, bool)
        if m.ndim != 2:
            raise ValueError("Region.mask must be [H,W]")
        object.__setattr__(self, "mask", m)
        if self.weight < 0:
            raise ValueError("Region.weight must be non-negative")


@dataclass(frozen=True)
class CorrespondenceField:
    """Dense mapping from A coordinates to B coordinates.

    uv[y,x] is displacement in pixels. confidence is [0,1]. valid defines
    pixels permitted to contribute to geometric evidence.
    """

    uv: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    method: str

    def __post_init__(self) -> None:
        uv = np.asarray(self.uv, float)
        c = np.asarray(self.confidence, float)
        v = np.asarray(self.valid, bool)
        if uv.ndim != 3 or uv.shape[2] != 2:
            raise ValueError("uv must be [H,W,2]")
        if c.shape != uv.shape[:2] or v.shape != uv.shape[:2]:
            raise ValueError("confidence/valid must match uv spatial shape")
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "confidence", np.clip(c, 0, 1))
        object.__setattr__(self, "valid", v)

    @property
    def support_fraction(self) -> float:
        return float(np.mean(self.valid))


@dataclass(frozen=True)
class RegistrationDiagnostics:
    method: str
    matrix_a_to_b: np.ndarray
    landmark_rmse_px: float | None
    weighted_rmse_px: float | None
    inlier_fraction: float | None
    overlap_fraction: float
    ecc_correlation: float | None = None
    support: SupportLevel = SupportLevel.VALID
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FacialState:
    """Shared canonical state consumed by every Wave-2 analyzer."""

    a: Frame
    b: Frame
    landmarks_a: LandmarkSet | None
    landmarks_b: LandmarkSet | None
    registration: RegistrationDiagnostics
    face_mask: np.ndarray
    regions: Mapping[str, Region] = field(default_factory=dict)
    correspondence: CorrespondenceField | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.face_mask, bool)
        if mask.shape != self.a.gray.shape or mask.shape != self.b.gray.shape:
            raise ValueError("face_mask must match canonical frame shape")
        object.__setattr__(self, "face_mask", mask)
        for name, r in self.regions.items():
            if r.mask.shape != mask.shape:
                raise ValueError(f"region {name!r} shape mismatch")

    @property
    def valid_mask(self) -> np.ndarray:
        m = self.face_mask.copy()
        if self.correspondence is not None:
            m &= self.correspondence.valid
        return m
