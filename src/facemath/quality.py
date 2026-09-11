from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import FacialState, SupportLevel


@dataclass(frozen=True)
class QualityReport:
    support: SupportLevel
    registration_overlap: float
    correspondence_support: float | None
    correspondence_confidence: float | None
    face_coverage: float
    region_count: int
    warnings: tuple[str, ...]


def assess_state(state: FacialState) -> QualityReport:
    warnings = list(state.registration.warnings)
    face_cov = float(np.mean(state.face_mask))
    corr_support = corr_conf = None
    support = state.registration.support
    if state.correspondence is not None:
        m = state.face_mask & state.correspondence.valid
        corr_support = float(np.sum(m) / max(np.sum(state.face_mask), 1))
        corr_conf = (
            float(np.mean(state.correspondence.confidence[m])) if np.any(m) else 0.0
        )
        if corr_support < 0.45:
            warnings.append("low_correspondence_support")
            support = (
                SupportLevel.DEGRADED if support == SupportLevel.VALID else support
            )
        if corr_support < 0.20:
            support = SupportLevel.INSUFFICIENT
        if corr_conf < 0.20:
            warnings.append("low_correspondence_confidence")
    if face_cov < 0.05:
        warnings.append("face_mask_too_small")
        support = SupportLevel.INSUFFICIENT
    return QualityReport(
        support,
        state.registration.overlap_fraction,
        corr_support,
        corr_conf,
        face_cov,
        len(state.regions),
        tuple(dict.fromkeys(warnings)),
    )
