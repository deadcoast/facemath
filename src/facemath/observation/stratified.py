"""Prospective acquisition-category calibration over one shared full16 error model.

Categories are curator-declared acquisition conditions, NOT inferred identities or
post-outcome clusters. Whole acquisition groups must belong to one category.
Development fits one mean/scatter. Only category-local calibration group maxima
choose radii. Query binding never looks at reference coordinates, pair similarity,
or evaluation outcomes. There is no pooled fallback for an unsupported category.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping

import numpy as np

from .bias import BiasCalibration
from .contracts import Observation, ObservationError, object_digest
from .error_model import corrected_quantile, _validate_rows
from .joint import JointCalibration
from .reference import check_digest, mapped_prediction
from .validation import LocalizationRow

FORMAT = "facemath.stratified_joint_calibration.v8"
BOUND_FORMAT = "facemath.source_bound_stratified_calibration.v8"
MODE = "development_joint_ocular_stratified"
RULE = "frozen_acquisition_category_whole_group_max_no_pooled_fallback.v8"


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ObservationError(f"{name} must be a nonempty trimmed string")
    return value


@dataclass(frozen=True)
class RoutingPolicy:
    strata: tuple[str, ...]
    definition: str
    minimum_calibration_groups: int = 9
    measurement_policy: Any | None = None

    def __post_init__(self) -> None:
        strata = tuple(_label(x, "acquisition stratum") for x in self.strata)
        if not strata or len(set(strata)) != len(strata):
            raise ObservationError("routing needs distinct prespecified acquisition strata")
        _label(self.definition, "pre-observation routing definition")
        if type(self.minimum_calibration_groups) is not int or self.minimum_calibration_groups < 1:
            raise ObservationError("minimum_calibration_groups must be a positive integer")
        object.__setattr__(self, "strata", strata)
        if self.measurement_policy is not None:
            from .acquisition import AcquisitionPolicy
            measured = (AcquisitionPolicy.from_dict(self.measurement_policy)
                        if isinstance(self.measurement_policy, Mapping) else self.measurement_policy)
            if not isinstance(measured, AcquisitionPolicy) or measured.categories != strata:
                raise ObservationError("measured rule categories differ from the frozen routing inventory")
            object.__setattr__(self, "measurement_policy", measured)

    def permits(self, category: str) -> bool:
        return self.measurement_policy is None or self.measurement_policy.permits(category)

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        result = {"format": "facemath.acquisition_routing_policy.v8", "rule": RULE,
                "strata": list(self.strata), "definition": self.definition,
                "minimum_calibration_groups": self.minimum_calibration_groups,
                "category_source": "curator-declared acquisition metadata, not learned from errors",
                "group_rule": "one stratum per declared dependent acquisition group"}
        if self.measurement_policy is not None:
            result["measurement_policy"] = self.measurement_policy.to_dict()
            result["category_source"] = "source-pixel predicates over all declared group apertures, before predictions"
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RoutingPolicy:
        if value.get("format") != "facemath.acquisition_routing_policy.v8" or value.get("rule") != RULE:
            raise ObservationError("unsupported acquisition routing policy")
        result = cls(tuple(value["strata"]), value["definition"], value["minimum_calibration_groups"], value.get("measurement_policy"))
        if result.to_dict() != dict(value):
            raise ObservationError("routing policy contains unrecognized or inconsistent fields")
        return result


def validate_routes(cases: tuple[Mapping[str, Any], ...], policy: RoutingPolicy) -> dict[str, str]:
    """Validate routing before predictions; no labels/errors are read by this function."""
    by_case: dict[str, str] = {}
    by_group: dict[str, str] = {}
    for case in cases:
        case_id = _label(case["case_id"], "case ID")
        group = _label(case["group_id"], "acquisition group")
        category = _label(case["stratum"], "acquisition stratum")
        if case_id in by_case or category not in policy.strata:
            raise ObservationError("duplicate case or undeclared acquisition stratum")
        if group in by_group and by_group[group] != category:
            raise ObservationError("dependent acquisition group crosses routing strata")
        by_case[case_id], by_group[group] = category, category
    return by_case


@dataclass(frozen=True)
class StratumRadius:
    stratum: str
    alpha: float
    minimum_groups: int
    group_scores: tuple[tuple[str, float | None], ...]
    calibration_images: tuple[str, ...]
    calibration_lineages: tuple[str, ...]
    deployment_permitted: bool = True
    quantile: float | None = field(init=False)
    rank: int = field(init=False)
    status: str = field(init=False)

    def __post_init__(self) -> None:
        _label(self.stratum, "radius stratum")
        if isinstance(self.alpha, (bool, np.bool_)) or not isinstance(self.alpha, (int, float)):
            raise ObservationError("routing alpha must be a numerical error rate")
        corrected_quantile(np.zeros(1), self.alpha)
        if type(self.minimum_groups) is not int or self.minimum_groups < 1:
            raise ObservationError("minimum category group count must be positive")
        if type(self.deployment_permitted) is not bool:
            raise ObservationError("category deployment permission must be boolean")
        groups = tuple(( _label(g, "calibration group"), v) for g, v in self.group_scores)
        if len(set(g for g, _ in groups)) != len(groups):
            raise ObservationError("duplicate stratum calibration group")
        for _, value in groups:
            if value is not None and (isinstance(value, (bool, np.bool_)) or
                    not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0):
                raise ObservationError("invalid stratum group score; null is the failure sentinel")
        images, lineages = tuple(self.calibration_images), tuple(self.calibration_lineages)
        if len(set(images)) != len(images) or len(set(lineages)) != len(lineages):
            raise ObservationError("duplicate stratum image/lineage inventory")
        for image in images:
            check_digest(image, "stratum calibration image")
        for lineage in lineages:
            _label(lineage, "stratum lineage")
        if bool(groups) != bool(images):
            raise ObservationError("stratum group/image availability differs")
        q, rank = (corrected_quantile(np.array([np.inf if v is None else v for _, v in groups]), self.alpha)
                   if groups else (None, ceil(1-self.alpha)))
        status = "supported" if q is not None else "unbounded_order_statistic"
        if not groups:
            status = "no_calibration_groups"
        elif len(groups) < self.minimum_groups:
            q, status = None, "below_frozen_minimum_groups"
        if not self.deployment_permitted:
            q, status = None, "rejected_by_measured_acquisition_policy"
        for key, value in (("group_scores", tuple(sorted(groups))), ("calibration_images", tuple(sorted(images))),
                           ("calibration_lineages", tuple(sorted(lineages))), ("quantile", q),
                           ("rank", rank), ("status", status)):
            object.__setattr__(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        result = {"format": "facemath.category_radius_fragment.v8", "stratum": self.stratum,
                "alpha": self.alpha, "minimum_groups": self.minimum_groups,
                "group_scores": [[g, v] for g, v in self.group_scores],
                "calibration_images": list(self.calibration_images),
                "calibration_lineages": list(self.calibration_lineages),
                "quantile": self.quantile, "rank": self.rank, "status": self.status,
                "calibration_groups": len(self.group_scores),
                "scope": "requires its containing joint model and frozen routing policy"}

        if not self.deployment_permitted:
            result["deployment_permitted"] = False
        return result

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> StratumRadius:
        if obj.get("format") != "facemath.category_radius_fragment.v8":
            raise ObservationError("unsupported category radius")
        out = cls(obj["stratum"], obj["alpha"], obj["minimum_groups"],
                  tuple(tuple(x) for x in obj["group_scores"]), tuple(obj["calibration_images"]),
                  tuple(obj["calibration_lineages"]), obj.get("deployment_permitted", True))
        if out.to_dict() != dict(obj):
            raise ObservationError("category radius differs from its retained order-statistic inputs")
        return out


@dataclass(frozen=True)
class SourceRoute:
    observation_sha256: str
    image_sha256: str
    group_id: str
    lineage_ids: tuple[str, ...]
    stratum: str
    measured_group_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.measured_group_sha256 is not None:
            check_digest(self.measured_group_sha256, "measured group receipt")
        check_digest(self.observation_sha256, "routed observation")
        check_digest(self.image_sha256, "routed image")
        _label(self.group_id, "routed group"); _label(self.stratum, "routed category")
        ids = tuple(_label(x, "routed lineage") for x in self.lineage_ids)
        if len(set(ids)) != len(ids):
            raise ObservationError("duplicate routed lineage")
        object.__setattr__(self, "lineage_ids", tuple(sorted(ids)))

    def to_dict(self) -> dict[str, Any]:
        result = {"observation_sha256": self.observation_sha256, "image_sha256": self.image_sha256,
                "group_id": self.group_id, "lineage_ids": list(self.lineage_ids), "stratum": self.stratum}
        if self.measured_group_sha256 is not None:
            result["measured_group_sha256"] = self.measured_group_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceRoute:
        return cls(**dict(value))


@dataclass(frozen=True)
class StratifiedJointCalibration(JointCalibration):
    policy: RoutingPolicy
    radii: tuple[StratumRadius, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        radii = tuple(self.radii)
        if tuple(r.stratum for r in radii) != self.policy.strata:
            raise ObservationError("radius order differs from frozen acquisition strata")
        groups: dict[str, float | None] = {}
        images: set[str] = set()
        lineages: set[str] = set()
        for radius in radii:
            if radius.deployment_permitted != self.policy.permits(radius.stratum):
                raise ObservationError("category deployment permission differs from frozen measurement rules")
            if radius.alpha != self.alpha or radius.minimum_groups != self.policy.minimum_calibration_groups:
                raise ObservationError("category radius differs from frozen error/minimum-group policy")
            if set(dict(radius.group_scores)) & set(groups) or set(radius.calibration_images) & images:
                raise ObservationError("calibration group/image crosses strata")
            if set(radius.calibration_lineages) & lineages:
                raise ObservationError("dependent calibration lineage crosses strata")
            groups.update(radius.group_scores); images.update(radius.calibration_images)
            lineages.update(radius.calibration_lineages)
        if (groups != dict(self.residual.group_scores) or images != set(self.residual.calibration_images)
                or lineages != set(self.residual.calibration_lineages)):
            raise ObservationError("stratified partition lost or changed pooled calibration evidence")
        object.__setattr__(self, "radii", radii)

    @property
    def quantile(self) -> None:
        # There is no single usable radius for an unbound category-conditioned model.
        return None

    def radius(self, stratum: str) -> StratumRadius:
        for radius in self.radii:
            if radius.stratum == stratum:
                return radius
        raise ObservationError(f"undeclared acquisition category: {stratum!r}; no pooled fallback")

    def require_query(self, *args: Any, **kwargs: Any) -> None:
        raise ObservationError("category-conditioned calibration requires source-bound routing")

    def source_quantile(self, observation: Observation) -> float:
        raise ObservationError("bind acquisition metadata before requesting a source radius")

    def bind(self, observations: tuple[Observation, Observation], groups: tuple[str, str],
             strata: tuple[str, str], domain: str, lineages: tuple[tuple[str, ...], tuple[str, ...]],
             *, measured_groups: tuple[Any, Any] | None = None) -> BoundStratifiedCalibration:
        if any(len(x) != 2 for x in (observations, groups, strata, lineages)):
            raise ObservationError("routing requires exactly two source declarations")
        if self.policy.measurement_policy is not None:
            from .acquisition import MeasuredGroup
            if measured_groups is None or len(measured_groups) != 2 or any(not isinstance(g, MeasuredGroup) for g in measured_groups):
                raise ObservationError("measured routing requires both original source-group pixel bundles")
            measured_groups = tuple(g.recompute() for g in measured_groups)
            derived = tuple(g.category for g in measured_groups)
            if strata != ("", "") and strata != derived:
                raise ObservationError("declared category conflicts with source-pixel routing; no manual override")
            strata = derived
        elif measured_groups is not None:
            raise ObservationError("pixel measurement groups cannot silently attach to a metadata-only model")
        routes = tuple(SourceRoute(o.digest, o.image_sha256, g, tuple(ids), category,
                                   measured_groups[i].digest if measured_groups is not None else None)
                       for i, (o, g, category, ids) in enumerate(zip(observations, groups, strata, lineages)))
        bound = BoundStratifiedCalibration(self.residual, self.bias, self.scatter, self, routes, measured_groups)
        for observation, group, ids in zip(observations, groups, lineages):
            bound.require_query(observation, group, domain, tuple(ids))
        return bound

    def to_dict(self) -> dict[str, Any]:
        # JointCalibration.to_dict uses the stored pooled residual, not quantile.
        return {"format": FORMAT, "rule": RULE, "routing_policy": self.policy.to_dict(),
                "shared_joint_model": JointCalibration.to_dict(self),
                "category_radii": [r.to_dict() for r in self.radii],
                "pooled_model_use": "diagnostic only; never a fallback for an unsupported query",
                "raw_geometry_changed": False}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> StratifiedJointCalibration:
        if obj.get("format") != FORMAT or obj.get("rule") != RULE:
            raise ObservationError("unsupported stratified calibration")
        model = JointCalibration.from_dict(obj["shared_joint_model"])
        result = cls(model.residual, model.bias, model.scatter, RoutingPolicy.from_dict(obj["routing_policy"]),
                     tuple(StratumRadius.from_dict(v) for v in obj["category_radii"]))
        if object_digest(result.to_dict()) != object_digest(dict(obj)):
            raise ObservationError("stratified model contains inconsistent/unknown data")
        return result


@dataclass(frozen=True)
class BoundStratifiedCalibration(JointCalibration):
    model: StratifiedJointCalibration
    routes: tuple[SourceRoute, SourceRoute]
    measured_groups: tuple[Any, Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.residual.digest != self.model.residual.digest or self.bias.digest != self.model.bias.digest
                or self.scatter.digest != self.model.scatter.digest):
            raise ObservationError("bound source routes lost their shared fitted error model")
        routes = tuple(self.routes)
        if self.model.policy.measurement_policy is not None:
            if any(r.measured_group_sha256 is None for r in routes):
                raise ObservationError("measured source route lost its original-pixel group binding")
        elif self.measured_groups is not None or any(r.measured_group_sha256 is not None for r in routes):
            raise ObservationError("unmeasured model cannot claim measured source bindings")
        if len(routes) != 2:
            raise ObservationError("bound pair needs exactly two source routes")
        for route in routes:
            if self.model.radius(route.stratum).quantile is None:
                raise ObservationError(f"acquisition category {route.stratum!r} is unsupported: "
                                       f"{self.model.radius(route.stratum).status}; no pooled fallback")
        left, right = routes
        if (left.image_sha256 == right.image_sha256 or left.group_id == right.group_id
                or set(left.lineage_ids) & set(right.lineage_ids)) and left.stratum != right.stratum:
            raise ObservationError("related query sources cannot claim different routing categories")
        if left.group_id == right.group_id and left.measured_group_sha256 != right.measured_group_sha256:
            raise ObservationError("same group cannot supply different pixel inventories to a pair")
        if left.observation_sha256 == right.observation_sha256 and left != right:
            raise ObservationError("same observed source has conflicting route declarations")
        object.__setattr__(self, "routes", routes)

    @property
    def quantile(self) -> float | None:
        values = [self.model.radius(r.stratum).quantile for r in self.routes]
        return values[0] if values[0] == values[1] else None

    def _route(self, observation: Observation) -> SourceRoute:
        for route in self.routes:
            if route.observation_sha256 == observation.digest and route.image_sha256 == observation.image_sha256:
                return route
        raise ObservationError("observation substitution in bound acquisition route")

    def source_quantile(self, observation: Observation) -> float:
        if self.model.policy.measurement_policy is not None and self.measured_groups is None:
            raise ObservationError("saved measured route needs original-pixel revalidation, not just a stored category")
        value = self.model.radius(self._route(observation).stratum).quantile
        if value is None:  # Constructor makes this unreachable for a valid object.
            raise ObservationError("unsupported source radius")
        return value

    def require_query(self, observation: Observation, group_id: str, domain: str,
                      lineage_ids: tuple[str, ...] = ()) -> None:
        route = self._route(observation)
        if self.model.policy.measurement_policy is not None:
            if self.measured_groups is None:
                raise ObservationError("saved measured route needs original-pixel revalidation")
            group = next((g for g in self.measured_groups if g.digest == route.measured_group_sha256), None)
            if (group is None or group.policy.digest != self.model.policy.measurement_policy.digest
                    or group.group_id != route.group_id or group.lineage_ids != route.lineage_ids
                    or group.category != route.stratum
                    or observation.image_sha256 not in {s.image_sha256 for s in group.sources}):
                raise ObservationError("measured group conflicts with source route or complete lineage inventory")
            # Check ALL measured group context, not only the selected query image.
            for source in group.sources:
                if (source.image_sha256 in self.residual.calibration_images
                        or set(source.lineage_ids) & set(self.residual.calibration_lineages)):
                    raise ObservationError("measured context member leaks calibration evidence")
                self.bias.require_disjoint(source.image_sha256, group.group_id, source.lineage_ids)
        if route.group_id != group_id or route.lineage_ids != tuple(sorted(lineage_ids)):
            raise ObservationError("query grouping/lineage differs from its bound route")
        # Validate against ALL calibration categories, not only the selected one.
        self.residual.require_query_context(observation, group_id, domain, lineage_ids)
        self.bias.require_disjoint(observation.image_sha256, group_id, lineage_ids)
        if not np.all(mapped_prediction(observation)[2][:4]):
            raise ObservationError("routed bias deployment requires four observable eye corners")

    def routing_evidence(self) -> dict[str, Any]:
        result = {"policy_sha256": self.model.policy.digest, "model_sha256": self.model.digest,
                "definition": self.model.policy.definition, "rule": RULE,
                "sources": [{**r.to_dict(), "quantile": self.model.radius(r.stratum).quantile,
                             "calibration_groups": len(self.model.radius(r.stratum).group_scores)} for r in self.routes],
                "claim": "conditional on correctly declared acquisition category and exchangeable whole groups within it",
                "not": "not automated domain recognition; a matching label cannot detect hidden within-category shift",
                "availability_limit": "the rank theorem averages calibration/new-group randomness; selecting only finite delivered outputs does not acquire an extra conditional coverage guarantee"}
        if self.model.policy.measurement_policy is not None:
            result["measurement_policy_sha256"] = self.model.policy.measurement_policy.digest
            result["category_source"] = "recomputed source pixels in all declared group apertures"
            result["claim"] = "conditional on exchangeable whole groups under the frozen measured partition; not hidden-shift detection"
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"format": BOUND_FORMAT, "model": self.model.to_dict(),
                "routes": [r.to_dict() for r in self.routes],
                "source_quantiles": [self.model.radius(r.stratum).quantile for r in self.routes]}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> BoundStratifiedCalibration:
        if obj.get("format") != BOUND_FORMAT:
            raise ObservationError("unsupported source-bound calibration")
        model = StratifiedJointCalibration.from_dict(obj["model"])
        result = cls(model.residual, model.bias, model.scatter, model,
                     tuple(SourceRoute.from_dict(x) for x in obj["routes"]))
        if object_digest(result.to_dict()) != object_digest(dict(obj)):
            raise ObservationError("bound source radii differ from frozen category calibration")
        return result


def stratify_joint(model: JointCalibration, scored_rows: tuple[LocalizationRow, ...],
                    cases: tuple[Mapping[str, Any], ...], policy: RoutingPolicy) -> StratifiedJointCalibration:
    """Reuse the development fit and case scores; only calibration partition changes."""
    _validate_rows(scored_rows)
    routes = validate_routes(cases, policy)
    selected = tuple(r for r in scored_rows if r.split == "calibration")
    if set(r.case_id for r in selected) != {c["case_id"] for c in cases if c["split"] == "calibration" and c.get("eligibility", "included") == "included"}:
        raise ObservationError("category calibration rows differ from frozen included calibration inventory")
    by_id = {c["case_id"]: c for c in cases}
    for row in selected:
        if row.group_id != by_id[row.case_id]["group_id"]:
            raise ObservationError("calibration row has changed acquisition group")
    radii = []
    for name in policy.strata:
        rows = [r for r in selected if routes[r.case_id] == name]
        scores: dict[str, float] = {}
        for row in rows:
            value = row.simultaneous_score
            ready = bool(value is not None and np.all(row.reference_visible) and np.all(row.observed)
                         and np.all(row.rater_count >= model.residual.minimum_raters))
            scores[row.group_id] = max(scores.get(row.group_id, 0.), value if ready else np.inf)
        radii.append(StratumRadius(name, model.alpha, policy.minimum_calibration_groups,
            tuple((g, s if np.isfinite(s) else None) for g, s in sorted(scores.items())),
            tuple(r.image_sha256 for r in rows), tuple(sorted({v for r in rows for v in r.lineage_ids})),
            policy.permits(name)))
    return StratifiedJointCalibration(model.residual, model.bias, model.scatter, policy, tuple(radii))


def evaluate_stratified(model: StratifiedJointCalibration, rows: tuple[LocalizationRow, ...],
                        cases: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    from .qualification import binomial_interval
    _validate_rows(rows)
    routes = validate_routes(cases, model.policy)
    by_id = {c["case_id"]: c for c in cases}
    evaluation = tuple(r for r in rows if r.split == "evaluation")
    expected = {c["case_id"] for c in cases if c["split"] == "evaluation" and c.get("eligibility", "included") == "included"}
    if not evaluation or {r.case_id for r in evaluation} != expected:
        raise ObservationError("routed evaluation must retain the entire frozen evaluation inventory")
    group_results: dict[str, dict[str, Any]] = {}
    outcomes = []
    for row in evaluation:
        if row.group_id != by_id[row.case_id]["group_id"]:
            raise ObservationError("evaluation group differs from frozen routing")
        if (row.image_sha256 in model.residual.calibration_images or row.group_id in dict(model.residual.group_scores)
                or set(row.lineage_ids) & set(model.residual.calibration_lineages)):
            raise ObservationError("routed evaluation leaks a calibration category")
        model.bias.require_disjoint(row.image_sha256, row.group_id, row.lineage_ids)
        if row.observer_recipe and row.observer_recipe != model.residual.observer_recipe_sha256:
            raise ObservationError("routed evaluation observer recipe differs")
        if row.reference_kind != model.reference_kind:
            raise ObservationError("routed evaluation reference kind differs")
        category = routes[row.case_id]; radius = model.radius(category)
        ready = bool(row.simultaneous_score is not None and np.all(row.observed)
                     and np.all(row.reference_visible) and np.all(row.rater_count >= model.residual.minimum_raters))
        finite = radius.quantile is not None
        hit = bool(finite and ready and row.simultaneous_score <= radius.quantile)
        record = group_results.setdefault(row.group_id, {"group_id": row.group_id, "stratum": category,
            "finite": finite, "covered": True, "cases": 0, "assessable": True})
        record["covered"] &= hit; record["assessable"] &= ready; record["cases"] += 1
        outcomes.append({"case_id": row.case_id, "group_id": row.group_id, "stratum": category,
            "score": row.simultaneous_score, "quantile": radius.quantile, "assessable": ready,
            "finite": finite, "covered": hit if finite else None, "route_status": radius.status})
    slices = []
    for radius in model.radii:
        groups = [g for g in group_results.values() if g["stratum"] == radius.stratum]
        count = len(groups); finite = radius.quantile is not None
        hits = sum(g["covered"] for g in groups)
        slices.append({"stratum": radius.stratum, "radius": radius.to_dict(), "evaluation": {
            "cases": sum(g["cases"] for g in groups), "groups": count,
            "covered_groups": hits if finite else None,
            "empirical_coverage": hits/count if count and finite else None,
            "binomial_95_interval": binomial_interval(hits, count) if count and finite else None,
            "finite_and_covered_fraction": hits/count if count else None,
            "case_count_per_group": {g["group_id"]: g["cases"] for g in groups}}})
    groups = list(group_results.values()); n = len(groups); hits = sum(g["covered"] for g in groups)
    finite_groups = sum(g["finite"] for g in groups)
    return {"calibration_sha256": model.digest, "nominal_coverage": 1-model.alpha,
        "evaluation_cases": len(evaluation), "evaluation_groups": n, "covered_groups": hits,
        "empirical_group_coverage": hits/n, "finite_envelope": finite_groups == n,
        "finite_envelope_groups": finite_groups, "unsupported_groups": n-finite_groups,
        "supported_group_empirical_coverage": hits/finite_groups if finite_groups else None,
        "failures": sum(not c["assessable"] for c in outcomes), "cases": outcomes,
        "groups": groups, "strata": slices,
        "meaning": "aggregate = fraction of ALL evaluation groups with finite delivered bounds AND covered references; abstentions are not deleted",
        "guarantee_scope": "category-local conformal rank under exchangeable groups in each frozen category; not per-image, not hidden-shift immunity",
        "evaluation_used_to_select_radii": False,
        "availability_limit": "unbounded theoretical error sets are not delivered as numerical geometry; conditioning on finite delivery is not a separate conformal guarantee"}
