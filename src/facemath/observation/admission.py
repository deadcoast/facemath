"""Prespecified empirical acceptance of a complete observer study.

A calibrated set being finite is not evidence that held-out delivery/coverage
met a deployment requirement. This layer consumes the SAME scored rows and
frozen category inventory as qualification. It neither refits nor ranks models.
Exact binomial lower bounds require independent, comparable groups within a
category, conditional on the frozen development/calibration fit. The program
records that assumption; it cannot establish it from group names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np
from scipy.stats import beta as beta_distribution

from .contracts import ObservationError, object_digest

POLICY_FORMAT = "facemath.empirical_admission_policy.v10"
RESULT_FORMAT = "facemath.empirical_admission.v10"


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise ObservationError(f"{name} must be a finite real number strictly between zero and one")
    return float(value)


def lower_binomial(successes: int, total: int, tail: float) -> float | None:
    """One-sided exact Clopper--Pearson lower bound. Zero trials is unknown."""
    tail = _probability(tail, "tail probability")
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ObservationError("binomial counts must be integers with 0 <= successes <= total")
    if total == 0:
        return None
    if successes == 0:
        return 0.0
    result = float(beta_distribution.ppf(tail, successes, total - successes + 1))
    if not math.isfinite(result):
        raise ObservationError("nonfinite binomial lower bound")
    return result


@dataclass(frozen=True)
class AdmissionPolicy:
    required_categories: tuple[str, ...]
    minimum_groups: int = 30
    minimum_delivery: float = .8
    minimum_coverage: float = .8
    family_error: float = .05
    evidence_kind: str = "native_reference"
    independent_groups_declared: bool = False
    sampling_design: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.required_categories, (tuple, list)):
            raise ObservationError("required_categories must be an explicit category sequence")
        categories = tuple(self.required_categories)
        if (not categories or any(not isinstance(s, str) or not s.strip() for s in categories)
                or len(categories) != len(set(categories))):
            raise ObservationError("required_categories must be unique nonempty names")
        object.__setattr__(self, "required_categories", categories)
        if type(self.minimum_groups) is not int or not 1 <= self.minimum_groups <= 1000000:
            raise ObservationError("minimum_groups must be a positive bounded integer")
        for name in ("minimum_delivery", "minimum_coverage", "family_error"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.evidence_kind not in ("native_reference", "synthetic_control"):
            raise ObservationError("evidence_kind must distinguish native_reference from synthetic_control")
        if type(self.independent_groups_declared) is not bool:
            raise ObservationError("independent_groups_declared must be boolean")
        if not isinstance(self.sampling_design, str) or not self.sampling_design.strip():
            raise ObservationError("document the sampling design and dependency grouping")

    @property
    def tail(self) -> float:
        # Two separately asserted endpoints per required category. Dependence
        # between endpoints/categories does not invalidate the union bound.
        return self.family_error / (2 * len(self.required_categories))

    @property
    def perfect_minimum(self) -> int:
        """Algebraic minimum n with a pass even if every group succeeds."""
        target = max(self.minimum_delivery, self.minimum_coverage)
        candidate = max(self.minimum_groups, int(math.ceil(math.log(self.tail) / math.log(target))))
        while self.tail ** (1.0 / candidate) < target:
            candidate += 1
        return candidate

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"format": POLICY_FORMAT, "required_categories": list(self.required_categories),
                "minimum_groups": self.minimum_groups, "minimum_delivery": self.minimum_delivery,
                "minimum_coverage": self.minimum_coverage, "family_error": self.family_error,
                "evidence_kind": self.evidence_kind,
                "independent_groups_declared": self.independent_groups_declared,
                "sampling_design": self.sampling_design}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdmissionPolicy:
        if not isinstance(value, Mapping) or value.get("format") != POLICY_FORMAT:
            raise ObservationError("invalid empirical admission policy format")
        expected = {"format", *cls.__dataclass_fields__}
        if set(value) != expected:
            raise ObservationError("empirical admission policy needs exactly its documented fields")
        return cls(**{k: v for k, v in value.items() if k != "format"})


def plan_admission(policy: AdmissionPolicy, cases: Sequence[Mapping[str, Any]],
                   categories: Sequence[str], execution_kind: str) -> dict[str, Any]:
    """Before observation: detect impossible targets without guessing accuracy."""
    if not set(policy.required_categories) <= set(categories):
        raise ObservationError("admission names a category outside the frozen inventory")
    if execution_kind not in ("native_mediapipe", "synthetic_fixture"):
        raise ObservationError("admission requires a known acquisition execution kind")
    expected = "native_reference" if execution_kind == "native_mediapipe" else "synthetic_control"
    if policy.evidence_kind != expected:
        raise ObservationError("admission evidence kind contradicts observer provenance")
    group_categories: dict[str, str] = {}
    for case in cases:
        group, category = case["group_id"], case.get("stratum", "all")
        if group in group_categories and group_categories[group] != category:
            raise ObservationError("an admission group cannot cross acquisition categories")
        group_categories[group] = category
    counts = {name: len({c["group_id"] for c in cases if c["split"] == "evaluation"
              and c.get("eligibility", "included") == "included" and c.get("stratum", "all") == name})
              for name in policy.required_categories}
    return {"policy_sha256": policy.digest, "planned_evaluation_groups": counts,
            "perfect_success_minimum_groups": policy.perfect_minimum,
            "best_possible_lower": {name: lower_binomial(n, n, policy.tail) for name, n in counts.items()},
            "algebraically_feasible": all(n >= policy.perfect_minimum for n in counts.values()),
            "meaning": "sample-size feasibility only, not expected accuracy; underpowered studies remain explicitly non-admissible"}


def _score_complete(row: Any, minimum_raters: int) -> bool:
    return bool(row.simultaneous_score is not None and math.isfinite(row.simultaneous_score)
                and np.all(row.reference_visible) and np.all(row.observed)
                and np.all(row.rater_count >= minimum_raters))


def assess_admission(policy: AdmissionPolicy, study_data: Mapping[str, Any], scored_rows: Sequence[Any],
                     calibration: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Use frozen evaluation groups; failures and unavailable sets stay in n.

    `scored_rows` are the qualified model's reference evaluator rows, not
    fabricated successes. A group is successful only if EVERY eligible case
    delivers a complete finite reference8 set / is covered, respectively.
    """
    plan = plan_admission(policy, study_data["cases"], study_data["strata"],
                          study_data["observer"]["execution_kind"])
    lookup = {c["case_id"]: c for c in study_data["cases"]
              if c["split"] == "evaluation" and c.get("eligibility", "included") == "included"}
    evaluation = [r for r in scored_rows if r.split == "evaluation"]
    if len({r.case_id for r in evaluation}) != len(evaluation) or {r.case_id for r in evaluation} != set(lookup):
        raise ObservationError("admission evaluation rows differ from the complete frozen inventory")
    groups: dict[str, dict[str, Any]] = {}
    for row in evaluation:
        case = lookup[row.case_id]
        if row.group_id != case["group_id"]:
            raise ObservationError("admission scored group differs from frozen group")
        category = case["stratum"]
        q = calibration.radius(category).quantile if hasattr(calibration, "radius") else calibration.quantile
        if q is not None and (isinstance(q, bool) or not isinstance(q, (int,float)) or not math.isfinite(q) or q < 0):
            raise ObservationError("admission radius must be finite nonnegative or explicitly unavailable")
        complete = _score_complete(row, study_data["minimum_raters"])
        delivered = complete and q is not None
        covered = delivered and row.simultaneous_score <= q
        entry = groups.setdefault(row.group_id, {"group_id": row.group_id, "category": category,
                    "case_ids": [], "delivered": True, "covered": True, "radius_available": q is not None})
        if entry["category"] != category:
            raise ObservationError("evaluation group crosses admission categories")
        entry["case_ids"].append(row.case_id)
        entry["delivered"] = bool(entry["delivered"] and delivered)
        entry["covered"] = bool(entry["covered"] and covered)
    ordered = [groups[k] for k in sorted(groups)]
    for g in ordered:
        g["case_ids"].sort()
    rows = []
    for category in policy.required_categories:
        members = [g for g in ordered if g["category"] == category]
        n = len(members); d = sum(g["delivered"] for g in members); k = sum(g["covered"] for g in members)
        ld, lc = lower_binomial(d, n, policy.tail), lower_binomial(k, n, policy.tail)
        reasons = []
        if not policy.independent_groups_declared:
            reasons.append("independent comparable groups not declared; binomial interpretation withheld")
        if n < policy.minimum_groups:
            reasons.append("too few complete evaluation groups in frozen inventory")
        if ld is None or ld < policy.minimum_delivery:
            reasons.append("finite delivery lower bound below frozen target")
        if lc is None or lc < policy.minimum_coverage:
            reasons.append("reference coverage lower bound below frozen target")
        rows.append({"category": category, "groups": n, "delivered": d, "covered": k,
            "delivery_lower": ld, "coverage_lower": lc, "delivery_target": policy.minimum_delivery,
            "coverage_target": policy.minimum_coverage, "passes": not reasons, "reasons": reasons})
    passed = all(r["passes"] for r in rows)
    outside = [g for g in ordered if g["category"] not in policy.required_categories]
    report = {"format": RESULT_FORMAT, "policy": policy.to_dict(), "policy_sha256": policy.digest,
        "calibration_sha256": calibration.digest, "plan": plan,
        "status": ("admitted_synthetic_control" if policy.evidence_kind == "synthetic_control" else "meets_declared_native_study_criteria")
                  if passed else "withheld", "passed": passed,
        "all_required_categories_must_pass": True, "tail_probability_per_endpoint": policy.tail,
        "multiplicity": 2 * len(policy.required_categories), "categories": rows, "groups": ordered,
        "all_evaluation_groups": len(ordered), "all_evaluation_delivered": sum(g["delivered"] for g in ordered),
        "all_evaluation_covered": sum(g["covered"] for g in ordered),
        "outside_requested_scope_groups": len(outside),
        "outside_requested_scope_delivered": sum(g["delivered"] for g in outside),
        "outside_requested_scope_covered": sum(g["covered"] for g in outside),
        "guarantee": "one-sided exact binomial lower bounds with Bonferroni allocation across two endpoints per frozen category",
        "assumptions": "conditional on frozen training/calibration, groups must be independent and comparable within category; curator declaration is not proof",
        "limitations": ["not a per-image posterior, identity score, or guarantee against hidden acquisition shift",
                        "does not select a subset of passing categories, refit q, or repair geometry using evaluation labels",
                        "cross-study multiple testing and repeated adaptive qualification attempts are not controlled",
                        "finite-sample conformal rank statements are not strengthened or reconditioned by acceptance",
                        "self-consistency and declared reference blinding are not external attestation"]}
    report["content_sha256"] = object_digest(report)
    category_order = list(study_data["strata"])
    arrays = {"admission_group_category": np.array([category_order.index(g["category"]) for g in ordered], np.int64),
              "admission_group_case_count": np.array([len(g["case_ids"]) for g in ordered], np.int64),
              "admission_group_delivered": np.array([g["delivered"] for g in ordered], bool),
              "admission_group_covered": np.array([g["covered"] for g in ordered], bool),
              "admission_category_bounds": np.array([[r["delivery_lower"] or 0., r["coverage_lower"] or 0.] for r in rows]),
              "admission_category_bounds_available": np.array([r["groups"] > 0 for r in rows], bool)}
    return report, arrays
