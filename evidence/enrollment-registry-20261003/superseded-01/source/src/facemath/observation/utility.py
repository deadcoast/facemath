"""Diagnostic delivery and widths from the existing full16 relation graph.

No fitting, acceptance decisions or replacement of measured coordinates. The
observed ocular chart is only a conditioning/unit transform; reference-set eye
spans remain uncertain when forming geometric ratios.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .bias import ocular_basis
from .contracts import ObservationError
from .intrinsic_sets import EllipticSource, Interval, SourceRelations, displacement
from .joint import JointCalibration
from .reference import REFERENCE_NAMES
from .stratified import StratifiedJointCalibration
from .validation import LocalizationRow

FORMAT = "facemath.joint_bound_utility.v13"
METRICS = ("landmark_diameter", "eye_span", "mouth_width", "mouth_eye_ratio")


def _interval(value: Interval) -> dict:
    return {**value.to_dict(), "width": value.upper-value.lower if value.available else None}


def _missing(reason: str) -> dict:
    return {"reason": reason, "observed_eye_span_px": None,
            "landmark_diameters": [None]*8,
            **{name: _interval(Interval.missing(reason)) for name in METRICS[1:]}}


def case_utility(row: LocalizationRow, calibration: JointCalibration, stratum: str) -> dict:
    """Describe this fitted set; caller supplies the frozen enrollment stratum.

    Development/calibration rows are explicitly diagnostic, not query admission.
    Missing reference support in a validation row stays missing in this report.
    """
    if row.observer_recipe is not None and row.observer_recipe != calibration.bias.observer_recipe_sha256:
        raise ObservationError("utility observer recipe differs from full error model")
    if row.reference_kind != calibration.reference_kind:
        raise ObservationError("utility reference provenance differs from full error model")
    q = (calibration.radius(stratum).quantile if isinstance(calibration, StratifiedJointCalibration)
         else calibration.quantile)
    if q is None:
        return _missing("no_finite_category_radius" if isinstance(calibration, StratifiedJointCalibration)
                        else "no_finite_radius")
    if not np.any(row.observed):
        return _missing(row.failure or "no_observed_reference_points")
    if not np.all(row.observed[:4]):
        return _missing("unobserved_ocular_dependency")
    try:
        basis = ocular_basis(row.predicted_xy)
    except ObservationError:
        return _missing("invalid_observed_ocular_span")
    span = float(np.linalg.norm(basis[:, 0]))
    origin = row.predicted_xy[:4].mean(axis=0)
    # Same observer-defined ocular chart as intrinsic geometry. These are SET
    # centers, never emitted as replacement predictions or fitted facial state.
    centers = np.linalg.solve(basis, (row.predicted_xy-origin).T).T-calibration.bias.beta
    covariance = calibration.scatter.covariance
    blocks = covariance.reshape(8, 2, 8, 2)
    marginal = np.stack([blocks[i, :, i, :] for i in range(8)])
    source = EllipticSource(centers, marginal, row.observed, q, joint_covariance=covariance)
    relations = SourceRelations(source)
    return {"reason": "bounded" if np.all(row.observed) else "partial_reference_support",
            "observed_eye_span_px": span,
            "landmark_diameters": [float(2*r) if supported else None
                                   for r, supported in zip(source.radii, row.observed)],
            "eye_span": _interval(relations.eye_span),
            "mouth_width": _interval(relations.norm(displacement(5, 6))),
            "mouth_eye_ratio": _interval(relations.length_ratio(displacement(5, 6)))}


def _distribution(values: list[float | None], total: int) -> dict:
    finite = [v for v in values if v is not None]
    if len(values) != total or not np.isfinite(finite).all():
        raise ObservationError("utility lost its finite-width denominator")
    return {"total": total, "finite": len(finite), "unavailable": total-len(finite),
            "minimum": min(finite) if finite else None,
            "median": float(np.median(finite)) if finite else None,
            "maximum": max(finite) if finite else None}


def _summary(cases: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(case['group_id'], []).append(case)
    included = [c for c in cases if c['eligibility'] == 'included']
    def delivered(c):
        return all(d is not None for d in c['landmark_diameters'])
    def group_count(predicate):
        # Complete enrolled groups only; exclusions never silently shrink them.
        return sum(all(predicate(c) for c in members) for members in groups.values())
    return {"enrolled_cases": len(cases), "eligible_cases": len(included),
            "prospectively_excluded": len(cases)-len(included), "enrolled_groups": len(groups),
            "complete_landmark_groups": group_count(delivered),
            "finite_ratio_groups": group_count(lambda c: c['mouth_eye_ratio']['available']),
            "reasons": dict(sorted(Counter(c['reason'] for c in cases).items())),
            "landmark_diameter": _distribution([d for c in cases for d in c['landmark_diameters']], 8*len(cases)),
            **{name: _distribution([c[name]['width'] for c in cases], len(cases)) for name in METRICS[1:]}}


def joint_bound_utility(rows: tuple[LocalizationRow, ...], calibration: Any,
                        enrollment: tuple[dict, ...] | list[dict]) -> dict:
    """Complete enrollment diagnostics, always bound to the full calibration."""
    if not isinstance(calibration, JointCalibration):
        return {"format": FORMAT, "status": "unsupported_error_geometry",
                "calibration_sha256": calibration.digest, "acceptance_decision": False}
    by_id = {r.case_id: r for r in rows}
    if (len(by_id) != len(rows) or len({c['case_id'] for c in enrollment}) != len(enrollment)
            or set(by_id) != {c['case_id'] for c in enrollment if c['eligibility'] == 'included'}):
        raise ObservationError("utility rows differ from frozen eligible enrollment")
    cases = []
    for entry in enrollment:
        row = by_id.get(entry['case_id'])
        if row is not None and (row.group_id != entry['group_id'] or row.split != entry['split']):
            raise ObservationError("utility split or group differs from frozen enrollment")
        value = case_utility(row, calibration, entry['stratum']) if row is not None else _missing("prospectively_excluded")
        cases.append({**{k: entry[k] for k in ('case_id', 'group_id', 'split', 'stratum', 'eligibility')},
                      "exclusion_reason": entry.get('exclusion_reason'), **value})
    return {"format": FORMAT, "status": "diagnostic", "calibration_sha256": calibration.digest,
            "bias_sha256": calibration.bias.digest, "scatter_sha256": calibration.scatter.digest,
            "raw_coordinates_changed": False, "acceptance_decision": False,
            "width_unit": "observed_eye_spans; ratio width is dimensionless",
            "normalization": "mouth_eye_ratio divides by uncertain reference-set eye span in shared relation graph",
            "landmark_diameter_meaning": "major-axis diameter of each marginal projection of the full16 set",
            "support_limit": "validation-row observed mask includes reference availability; not a visibility claim",
            "summary_scope": "all enrolled cases, including exclusions; complete-group delivery requires every member",
            "limit": "descriptive finite widths do not establish useful accuracy, anatomical equivalence or empirical admission",
            "splits": {s: _summary([c for c in cases if c['split'] == s])
                       for s in ('development', 'calibration', 'evaluation')},
            "evaluation_strata": {s: _summary([c for c in cases if c['split'] == 'evaluation' and c['stratum'] == s])
                                  for s in sorted({c['stratum'] for c in cases})},
            "cases": cases}
