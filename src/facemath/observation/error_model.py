"""Group-level split-conformal localization envelopes, separate from similarity.

The order-statistic correction is ceil((n+1)*(1-alpha)); rank n+1 is unbounded,
NOT a clipped maximum. Scores maximize over landmarks, then over declared group
members. Guarantees require exchangeable groups under the frozen acquisition and
annotation protocol. A calibration artifact is not evidence those assumptions hold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from .contracts import ObservationError, frozen_array, object_digest
from .reference import (
    REFERENCE_NAMES,
    REFERENCE_PROTOCOL_SHA256,
    check_digest,
    mapped_prediction,
    observer_recipe,
)
from .validation import LocalizationRow


def corrected_quantile(scores: np.ndarray, alpha: float) -> tuple[float | None, int]:
    scores = np.asarray(scores, float)
    if (
        scores.ndim != 1
        or not len(scores)
        or np.isnan(scores).any()
        or np.any(scores < 0)
    ):
        raise ObservationError(
            "scores must be a nonempty nonnegative vector; +infinity represents failure"
        )
    if not np.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ObservationError("alpha must be in (0, .5)")
    k = ceil((len(scores) + 1) * (1 - alpha))
    if k > len(scores):
        return None, k
    value = float(np.partition(scores, k - 1)[k - 1])
    return (value if np.isfinite(value) else None), k


@dataclass(frozen=True)
class LocalizationCalibration:
    alpha: float
    quantile: float | None
    rank: int
    group_scores: tuple[tuple[str, float | None], ...]
    observer_recipe_sha256: str
    study_sha256: str
    domain: str
    reference_kind: str
    calibration_images: tuple[str, ...]
    calibration_lineages: tuple[str, ...]
    reference_digests: tuple[str, ...]
    minimum_raters: int = 1

    def __post_init__(self) -> None:
        check_digest(self.observer_recipe_sha256, "observer recipe")
        check_digest(self.study_sha256, "study digest")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ObservationError("calibration domain must be explicitly declared")
        if self.reference_kind not in (
            "human_blind",
            "external_dataset",
            "synthetic_fixture",
        ):
            raise ObservationError(
                "exposed manual labels cannot calibrate localization accuracy"
            )
        groups = tuple((str(k), v) for k, v in self.group_scores)
        if any(
            not k or (v is not None and (not np.isfinite(v) or v < 0))
            for k, v in groups
        ):
            raise ObservationError("invalid group scores")
        if len({k for k, _ in groups}) != len(groups):
            raise ObservationError("duplicate calibration group")
        q, k = corrected_quantile(
            np.array([np.inf if v is None else v for _, v in groups]), self.alpha
        )
        if q != self.quantile or k != self.rank:
            raise ObservationError(
                "calibration order statistic is inconsistent with retained group scores"
            )
        for image in self.calibration_images:
            check_digest(image, "calibration image")
        if type(self.minimum_raters) is not int or self.minimum_raters < 1:
            raise ObservationError("minimum_raters must be a positive integer")
        object.__setattr__(self, "group_scores", groups)
        for key in ("calibration_images", "calibration_lineages", "reference_digests"):
            object.__setattr__(self, key, tuple(getattr(self, key)))

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "format": "facemath.localization_calibration.v4",
            **asdict(self),
            "protocol_sha256": REFERENCE_PROTOCOL_SHA256,
            "names": list(REFERENCE_NAMES),
            "scope": (
                "synthetic_only"
                if self.reference_kind == "synthetic_fixture"
                else "declared_reference_domain_only"
            ),
            "score": "group_max_of_case_max_reference8_mahalanobis",
            "guarantee": "marginal new-group simultaneous-reference coverage under exchangeability and frozen protocol; not per-image posterior certainty",
            "unbounded": self.quantile is None,
            "missingness": "failed or partially labeled calibration cases contribute +infinity, serialized as null",
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> LocalizationCalibration:
        if (
            obj.get("format") != "facemath.localization_calibration.v4"
            or obj.get("protocol_sha256") != REFERENCE_PROTOCOL_SHA256
        ):
            raise ObservationError("unsupported localization calibration")
        if tuple(obj.get("names", ())) != REFERENCE_NAMES:
            raise ObservationError("calibration name order differs")
        from dataclasses import fields

        return cls(**{f.name: obj[f.name] for f in fields(cls)})

    def require_query(
        self,
        observation: Any,
        group_id: str,
        domain: str,
        lineage_ids: tuple[str, ...] = (),
    ) -> None:
        if self.quantile is None:
            raise ObservationError(
                "localization calibration is unbounded: insufficient groups or failure mass"
            )
        if not group_id or domain != self.domain:
            raise ObservationError(
                "explicit query group and matching acquisition domain required"
            )
        if observer_recipe(observation) != self.observer_recipe_sha256:
            raise ObservationError("localization observer recipe mismatch")
        if observation.image_sha256 in self.calibration_images or group_id in dict(
            self.group_scores
        ):
            raise ObservationError("localization calibration/query leakage")
        if set(lineage_ids) & set(self.calibration_lineages):
            raise ObservationError("localization lineage leakage")
        if (
            self.reference_kind == "synthetic_fixture"
            and "synthetic" not in observation.provider
        ):
            raise ObservationError(
                "synthetic calibration cannot qualify a human/model observation"
            )


def _validate_rows(rows: tuple[LocalizationRow, ...]) -> None:
    """Public numerical entry points enforce split contracts too, not just the CLI."""
    if not rows or len({r.case_id for r in rows}) != len(rows):
        raise ObservationError("localization rows must have unique case IDs")
    if len({r.image_sha256 for r in rows}) != len(rows):
        raise ObservationError("duplicate localization source image")
    tokens: dict[str, tuple[str, str]] = {}
    for r in rows:
        for token in [f"group:{r.group_id}"] + [f"lineage:{x}" for x in r.lineage_ids]:
            if old := tokens.get(token):
                if old[1] != r.split:
                    raise ObservationError("localization row split leakage")
                if old[0] != r.group_id:
                    raise ObservationError("dependent lineage split across groups")
            tokens[token] = (r.group_id, r.split)


def fit_localization_calibration(
    rows: tuple[LocalizationRow, ...],
    *,
    study_sha256: str,
    alpha: float,
    domain: str,
    minimum_raters: int = 1,
) -> LocalizationCalibration:
    _validate_rows(rows)
    selected = tuple(r for r in rows if r.split == "calibration")
    if not selected:
        raise ObservationError("no calibration-split cases")
    recipes = {r.observer_recipe for r in rows if r.observer_recipe}
    kinds = {r.reference_kind for r in rows}
    if len(recipes) != 1 or len(kinds) != 1:
        raise ObservationError(
            "need one observed provider recipe and one reference provenance"
        )
    groups: dict[str, float] = {}
    for r in selected:
        # Max over all eight *prespecified* labels; absent labels are not silently removed.
        score = (
            r.simultaneous_score
            if np.all(r.reference_visible)
            and np.all(r.rater_count >= minimum_raters)
            and r.simultaneous_score is not None
            else np.inf
        )
        groups[r.group_id] = max(groups.get(r.group_id, 0), score)
    q, k = corrected_quantile(np.array(list(groups.values())), alpha)
    return LocalizationCalibration(
        alpha,
        q,
        k,
        tuple((g, s if np.isfinite(s) else None) for g, s in sorted(groups.items())),
        next(iter(recipes)),
        study_sha256,
        domain,
        next(iter(kinds)),
        tuple(sorted(r.image_sha256 for r in selected)),
        tuple(sorted({x for r in selected for x in r.lineage_ids})),
        tuple(sorted({d for r in selected for d in r.reference_digests})),
        minimum_raters,
    )


def evaluate_coverage(
    calibration: LocalizationCalibration, rows: tuple[LocalizationRow, ...]
) -> dict[str, Any]:
    _validate_rows(rows)
    evaluation = tuple(r for r in rows if r.split == "evaluation")
    if not evaluation:
        raise ObservationError("held-out evaluation split is empty")
    for row in evaluation:
        if row.image_sha256 in calibration.calibration_images or row.group_id in dict(
            calibration.group_scores
        ):
            raise ObservationError(
                "coverage evaluation leaks calibration images/groups"
            )
        if set(row.lineage_ids) & set(calibration.calibration_lineages):
            raise ObservationError("coverage evaluation leaks calibration lineages")
        if (
            row.observer_recipe
            and row.observer_recipe != calibration.observer_recipe_sha256
        ):
            raise ObservationError("coverage observer recipe differs")
        if row.reference_kind != calibration.reference_kind:
            raise ObservationError("coverage reference provenance differs")
    group_results: dict[str, bool] = {}
    outcomes = []
    for r in evaluation:
        assessable = bool(
            r.simultaneous_score is not None
            and np.all(r.reference_visible)
            and np.all(r.rater_count >= calibration.minimum_raters)
        )
        covered = (
            assessable
            and calibration.quantile is not None
            and r.simultaneous_score <= calibration.quantile
        )
        group_results[r.group_id] = group_results.get(r.group_id, True) and covered
        outcomes.append(
            {
                "case_id": r.case_id,
                "group_id": r.group_id,
                "score": r.simultaneous_score,
                "assessable": assessable,
                "covered": covered if calibration.quantile is not None else None,
            }
        )
    return {
        "calibration_sha256": calibration.digest,
        "nominal_coverage": 1 - calibration.alpha,
        "finite_envelope": calibration.quantile is not None,
        "evaluation_cases": len(evaluation),
        "evaluation_groups": len(group_results),
        "covered_groups": (
            sum(group_results.values()) if calibration.quantile is not None else None
        ),
        "empirical_group_coverage": (
            float(np.mean(list(group_results.values())))
            if calibration.quantile is not None
            else None
        ),
        "failures": sum(not c["assessable"] for c in outcomes),
        "cases": outcomes,
        "interpretation": "held-out reference coverage, not facial resemblance or a guarantee of deployment coverage",
    }


@dataclass(frozen=True)
class DirectionalBounds:
    """Conservative Euclidean gap bounds for a translated Minkowski sum of ellipses.

    Finite normal sampling gives a lower distance bound. For the upper bound,
    the maximum sampled support is divided by cos(pi/M): every vector lies within
    pi/M of a sampled normal. Merely taking sampled maxima would under-bound.
    These bounds are intersected with the simpler enclosing-circle bounds.
    """

    normalized_covariances: np.ndarray
    normals: np.ndarray
    combined_support: np.ndarray
    lower_ipd: np.ndarray
    upper_ipd: np.ndarray

    def __post_init__(self) -> None:
        for key in (
            "normalized_covariances",
            "normals",
            "combined_support",
            "lower_ipd",
            "upper_ipd",
        ):
            object.__setattr__(self, key, frozen_array(getattr(self, key)))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            f"localization_ellipse_{key}": getattr(self, key)
            for key in (
                "normalized_covariances",
                "normals",
                "combined_support",
                "lower_ipd",
                "upper_ipd",
            )
        }


def ellipse_gap_bounds(
    delta: np.ndarray, covariances: np.ndarray, quantile: float, directions: int = 128
) -> DirectionalBounds:
    """Two [N,2,2] covariance fields, already pushed into normalized canonical units."""
    d, c = np.asarray(delta, float), np.asarray(covariances, float)
    if (
        d.ndim != 2
        or d.shape[1] != 2
        or c.shape != (2, len(d), 2, 2)
        or not np.isfinite(d).all()
        or not np.isfinite(c).all()
    ):
        raise ObservationError(
            "ellipse bounds require finite [N,2] deltas and [2,N,2,2] covariance"
        )
    if not np.allclose(c, c.swapaxes(-1, -2), rtol=1e-10, atol=1e-12) or np.any(
        np.linalg.eigvalsh(c) <= 0
    ):
        raise ObservationError("ellipse covariance must be symmetric positive definite")
    if (
        not np.isfinite(quantile)
        or quantile < 0
        or type(directions) is not int
        or not 8 <= directions <= 4096
    ):
        raise ObservationError("invalid ellipse radius or angular resolution")
    angles = np.arange(directions) * (2 * np.pi / directions)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    variance = np.einsum("mi,anij,mj->anm", normals, c, normals)
    support = quantile * np.sqrt(np.maximum(0, variance)).sum(axis=0)
    projection = d @ normals.T
    gap = np.linalg.norm(d, axis=1)
    radii = quantile * np.sqrt(np.linalg.eigvalsh(c)[..., -1]).sum(axis=0)
    roundoff = 64 * np.finfo(float).eps * np.maximum(1, gap + radii)
    lower = np.maximum(
        0, np.maximum(gap - radii, np.max(projection - support, axis=1)) - roundoff
    )
    # Every point in the uncertainty set projects at least ||point|| cos(pi/M)
    # onto its closest sampled normal. The support bounds every such projection.
    angular_upper = np.max(projection + support, axis=1) / np.cos(np.pi / directions)
    upper = np.minimum(gap + radii, angular_upper) + roundoff
    return DirectionalBounds(c, normals, support, lower, upper)


@dataclass(frozen=True)
class LocalizationEnvelope:
    calibration: LocalizationCalibration
    query_groups: tuple[str, str]
    domain: str
    source_radii_px: np.ndarray
    canonical_radii_px: np.ndarray
    canonical_points: np.ndarray
    radial_budget_ipd: np.ndarray
    observed_gap_ipd: np.ndarray
    lower_ipd: np.ndarray
    upper_ipd: np.ndarray
    supported: np.ndarray
    tolerance_ipd: float
    query_lineages: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    directional: DirectionalBounds | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_radii_px",
            "canonical_radii_px",
            "canonical_points",
            "radial_budget_ipd",
            "observed_gap_ipd",
            "lower_ipd",
            "upper_ipd",
        ):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        object.__setattr__(self, "supported", frozen_array(self.supported, bool))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            f"localization_{n}": getattr(self, n)
            for n in (
                "source_radii_px",
                "canonical_radii_px",
                "canonical_points",
                "radial_budget_ipd",
                "observed_gap_ipd",
                "lower_ipd",
                "upper_ipd",
                "supported",
            )
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_sha256": self.calibration.digest,
            "names": list(REFERENCE_NAMES),
            "bound_method": "ellipse_support_outer_bounds_with_finite_angle_correction_and_radial_intersection",
            "angular_directions": (
                len(self.directional.normals) if self.directional else None
            ),
            "query_groups": list(self.query_groups),
            "query_lineages": [list(x) for x in self.query_lineages],
            "domain": self.domain,
            "nominal_per_source_coverage": 1 - self.calibration.alpha,
            "pair_union_bound": max(0, 1 - 2 * self.calibration.alpha),
            "scope": (
                "synthetic_only"
                if self.calibration.reference_kind == "synthetic_fixture"
                else "calibration_reference_domain"
            ),
            "gauge": "observed_transforms_and_observed_normalizer_held_fixed; NOT re-fitted_true_shape_geometry",
            "interpretation": "conservative geometric gap bounds from calibrated reference ellipses; full covariance pushforward with fixed observed gauge; no extrapolation to unannotated vertices",
            "rows": [
                {
                    "name": name,
                    "observed_gap_ipd": (
                        float(self.observed_gap_ipd[i]) if self.supported[i] else None
                    ),
                    "lower_ipd": (
                        float(self.lower_ipd[i]) if self.supported[i] else None
                    ),
                    "upper_ipd": (
                        float(self.upper_ipd[i]) if self.supported[i] else None
                    ),
                    "status": (
                        "withheld"
                        if not self.supported[i]
                        else (
                            "within_tolerance_under_envelope"
                            if self.upper_ipd[i] <= self.tolerance_ipd
                            else (
                                "outside_tolerance_under_envelope"
                                if self.lower_ipd[i] > self.tolerance_ipd
                                else "unresolved_at_tolerance"
                            )
                        )
                    ),
                }
                for i, name in enumerate(REFERENCE_NAMES)
            ],
        }


def assess_pair(
    state: Any,
    calibration: LocalizationCalibration,
    *,
    query_groups: tuple[str, str],
    domain: str,
    query_lineages: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
) -> LocalizationEnvelope:
    from .provider import apply_points

    if len(query_groups) != 2 or not all(query_groups):
        raise ObservationError(
            "provide two explicit acquisition-group IDs; no identity is inferred"
        )
    if len(query_lineages) != 2 or any(
        not isinstance(x, str) or not x for group in query_lineages for x in group
    ):
        raise ObservationError("query lineages require two collections of nonempty IDs")
    points, radii, mapped_radii, valid, pushed_covariances = [], [], [], [], []
    for observation, matrix, group, lineages in zip(
        (state.observation_a, state.observation_b),
        (state.source_to_canonical_a, state.source_to_canonical_b),
        query_groups,
        query_lineages,
    ):
        calibration.require_query(observation, group, domain, tuple(lineages))
        xy, cov, support = mapped_prediction(observation)
        points.append(apply_points(xy, matrix))
        pushed_covariances.append(
            np.einsum("ij,njk,lk->nil", matrix[:2, :2], cov, matrix[:2, :2])
        )
        radius = calibration.quantile * np.sqrt(np.linalg.eigvalsh(cov)[:, -1])
        # Similarity map has exactly one isotropic scale. Largest singular value
        # remains a safe bound should a general affine gauge be supplied later.
        radii.append(radius)
        mapped_radii.append(radius * np.linalg.svd(matrix[:2, :2], compute_uv=False)[0])
        valid.append(support)
    pts, sr, cr = np.stack(points), np.stack(radii), np.stack(mapped_radii)
    support = np.logical_and(*valid)
    span = state.normalization_span_px
    gap = np.linalg.norm(pts[0] - pts[1], axis=1) / span
    budget = cr.sum(axis=0) / span
    directional = ellipse_gap_bounds(
        (pts[0] - pts[1]) / span,
        np.stack(pushed_covariances) / span**2,
        calibration.quantile,
    )
    lower, upper = directional.lower_ipd.copy(), directional.upper_ipd.copy()
    lower[~support], upper[~support] = 0, 0
    return LocalizationEnvelope(
        calibration,
        tuple(query_groups),
        domain,
        sr,
        cr,
        pts,
        budget,
        gap,
        lower,
        upper,
        support,
        state.config.tolerance_ipd,
        tuple(tuple(x) for x in query_lineages),
        directional,
    )
