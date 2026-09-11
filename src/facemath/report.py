from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import AnalysisBundle
from .fusion import calibrated_consensus, summarize_domains


@dataclass(frozen=True)
class EvidenceReport:
    version: str
    support: str
    warnings: tuple[str, ...]
    consensus_percentile: float | None
    domain_summaries: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    interpretation_limits: tuple[str, ...]


def build_report(bundle: AnalysisBundle) -> EvidenceReport:
    q = bundle.artifacts.get("quality_report")
    support = getattr(getattr(q, "support", None), "value", "unknown")
    warnings = tuple(getattr(q, "warnings", ()))
    domains = [
        {
            "domain": s.domain,
            "calibrated_percentiles": list(s.calibrated_percentiles),
            "median_percentile": s.median_percentile,
            "metrics": list(s.metrics),
        }
        for s in summarize_domains(bundle.ledger)
    ]
    limits = []
    if calibrated_consensus(bundle.ledger) is None:
        limits.append(
            "No population-calibrated consensus is available; raw metric values must not be presented as similarity percentages."
        )
    if support != "valid":
        limits.append(
            "Input support is not fully valid; substantive facial-geometry conclusions require caution or additional correspondence evidence."
        )
    if bundle.pair.landmarks_a is None or bundle.pair.landmarks_b is None:
        limits.append(
            "No explicit anatomical landmark correspondence was supplied; geometry-specific claims are limited to signal/deformation evidence."
        )
    return EvidenceReport(
        "wave2",
        support,
        warnings,
        calibrated_consensus(bundle.ledger),
        tuple(domains),
        bundle.ledger.to_jsonable(),
        tuple(limits),
    )
