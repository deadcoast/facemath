from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import EvidenceLedger, Direction


@dataclass(frozen=True)
class DomainSummary:
    domain: str
    calibrated_percentiles: tuple[float, ...]
    median_percentile: float | None
    metrics: tuple[str, ...]


def summarize_domains(ledger: EvidenceLedger) -> list[DomainSummary]:
    domains = sorted({r.domain for r in ledger.results.values()})
    out=[]
    for d in domains:
        rs=ledger.by_domain(d)
        p=tuple(float(r.calibrated_percentile) for r in rs if r.calibrated_percentile is not None)
        out.append(DomainSummary(d,p,float(np.median(p)) if p else None,tuple(r.name for r in rs)))
    return out


def calibrated_consensus(ledger: EvidenceLedger, domains: tuple[str, ...] | None = None) -> float | None:
    summaries=summarize_domains(ledger)
    if domains is not None:
        summaries=[s for s in summaries if s.domain in domains]
    vals=[s.median_percentile for s in summaries if s.median_percentile is not None]
    if not vals:
        return None
    # domain-balanced aggregation: one vote per domain, not one vote per metric.
    return float(np.median(vals))
