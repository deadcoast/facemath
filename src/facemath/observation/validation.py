"""Source-pixel localization evaluation, with failed attempts in denominators.

No registration is fitted to reference labels. Error-to-reference, annotation
agreement and perturbation repeatability remain distinct measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import Observation, ObservationError, frozen_array, object_digest
from .reference import (
    REFERENCE_NAMES,
    REFERENCE_PROTOCOL_SHA256,
    ReferenceAnnotation,
    consensus,
    mapped_prediction,
    observer_recipe,
)


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    group_id: str
    split: str
    references: tuple[ReferenceAnnotation, ...]
    observation: Observation | None
    failure: str | None = None
    lineage_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.case_id
            or not self.group_id
            or self.split not in ("development", "calibration", "evaluation")
        ):
            raise ObservationError(
                "case needs nonempty IDs and an explicit development/calibration/evaluation split"
            )
        if not self.references:
            raise ObservationError(
                "every validation attempt requires independent references"
            )
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "lineage_ids", tuple(self.lineage_ids))
        if not all(self.lineage_ids):
            raise ObservationError("lineage IDs cannot be empty")
        if (self.observation is None) != bool(self.failure):
            raise ObservationError(
                "failed attempts need a reason; successful attempts cannot have one"
            )
        c = consensus(self.references)
        c.eye_span_px  # require a usable independently labeled denominator
        if self.observation:
            r = self.references[0]
            if (
                r.image_sha256 != self.observation.image_sha256
                or r.image_shape != self.observation.image_shape
            ):
                raise ObservationError("prediction/reference source binding mismatch")

    @property
    def image_sha256(self) -> str:
        return self.references[0].image_sha256


@dataclass(frozen=True)
class LocalizationRow:
    case_id: str
    group_id: str
    split: str
    image_sha256: str
    lineage_ids: tuple[str, ...]
    reference_xy: np.ndarray
    predicted_xy: np.ndarray
    covariance: np.ndarray
    reference_visible: np.ndarray
    observed: np.ndarray
    error_vectors_px: np.ndarray
    error_norm_px: np.ndarray
    mahalanobis: np.ndarray
    rater_count: np.ndarray
    rater_disagreement_px: np.ndarray
    normalization_span_px: float
    nme: float | None
    simultaneous_score: float | None
    failure: str | None
    reference_kind: str
    observer_recipe: str | None
    reference_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "reference_xy",
            "predicted_xy",
            "covariance",
            "error_vectors_px",
            "error_norm_px",
            "mahalanobis",
            "rater_disagreement_px",
        ):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        for name in ("reference_visible", "observed"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), bool))
        object.__setattr__(
            self, "rater_count", frozen_array(self.rater_count, np.int64)
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
        return data


def evaluate_case(case: ValidationCase, minimum_raters: int = 1) -> LocalizationRow:
    ref = consensus(case.references, minimum_raters)
    span = ref.eye_span_px
    n = len(REFERENCE_NAMES)
    predicted, cov, observed = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros(n, bool)
    residual, error, mahal = np.zeros((n, 2)), np.zeros(n), np.zeros(n)
    recipe = None
    if case.observation is not None:
        predicted, cov, supported = mapped_prediction(case.observation)
        observed = supported & ref.visible
        residual[observed] = predicted[observed] - ref.xy[observed]
        error[observed] = np.linalg.norm(residual[observed], axis=1)
        # Do not pseudo-invert invalid covariance into false certainty.
        if np.any(np.linalg.eigvalsh(cov) <= 0):
            raise ObservationError("prediction covariance must be positive definite")
        mahal[observed] = np.sqrt(
            np.maximum(
                0,
                np.einsum(
                    "ni,nij,nj->n",
                    residual[observed],
                    np.linalg.inv(cov[observed]),
                    residual[observed],
                ),
            )
        )
        recipe = observer_recipe(case.observation)
    complete = bool(np.all(observed[ref.visible]))
    failure = case.failure or (
        None if complete else "unobservable_predicted_reference_points"
    )
    return LocalizationRow(
        case.case_id,
        case.group_id,
        case.split,
        case.image_sha256,
        case.lineage_ids,
        ref.xy,
        predicted,
        cov,
        ref.visible,
        observed,
        residual,
        error,
        mahal,
        ref.rater_count,
        ref.disagreement_rms_px,
        span,
        float(error[observed].mean() / span) if complete else None,
        float(mahal[ref.visible].max()) if complete else None,
        failure,
        ref.kind,
        recipe,
        ref.annotation_digests,
    )


def validate_study(cases: tuple[ValidationCase, ...]) -> None:
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ObservationError("study requires unique nonempty cases")
    if len({c.image_sha256 for c in cases}) != len(cases):
        raise ObservationError(
            "duplicate source image; aggregate raters inside one case"
        )
    groups_by_token: dict[str, tuple[str, str]] = {}
    for c in cases:
        # No automatic identity recognition: grouping is supplied by the study curator.
        tokens = [f"group:{c.group_id}", f"image:{c.image_sha256}"] + [
            f"lineage:{s}" for s in c.lineage_ids
        ]
        for token in tokens:
            previous = groups_by_token.get(token)
            if previous and previous[1] != c.split:
                raise ObservationError(
                    f"split leakage: {token} appears in {previous[1]} and {c.split}"
                )
            if previous and token.startswith("lineage:") and previous[0] != c.group_id:
                raise ObservationError(
                    "same lineage assigned to different groups; merge the dependent group"
                )
            groups_by_token[token] = (c.group_id, c.split)
    recipes = {
        observer_recipe(c.observation) for c in cases if c.observation is not None
    }
    if len(recipes) > 1:
        raise ObservationError("mixed observer recipes; validate separately")
    if len({r.kind for c in cases for r in c.references}) != 1:
        raise ObservationError(
            "mixed reference provenance; do not pool synthetic and independent labels"
        )


def summarize(
    rows: tuple[LocalizationRow, ...],
    pck_threshold: float = 0.05,
    auc_threshold: float = 0.1,
) -> dict[str, Any]:
    if not rows or not 0 < pck_threshold <= 1 or not 0 < auc_threshold <= 1:
        raise ObservationError(
            "nonempty rows and positive normalized thresholds required"
        )
    normalized = np.stack([r.error_norm_px / r.normalization_span_px for r in rows])
    eligible, observed = (
        np.stack([r.reference_visible for r in rows]),
        np.stack([r.observed for r in rows]),
    )
    hit = eligible & observed & (normalized <= pck_threshold)
    per_landmark = []
    for i, name in enumerate(REFERENCE_NAMES):
        total, supported = int(eligible[:, i].sum()), int(observed[:, i].sum())
        per_landmark.append(
            {
                "name": name,
                "eligible": total,
                "observed": supported,
                "within_tolerance": int(hit[:, i].sum()),
                "pck": float(hit[:, i].sum() / total) if total else None,
                "median_error_px": (
                    float(
                        np.median([r.error_norm_px[i] for r in rows if r.observed[i]])
                    )
                    if supported
                    else None
                ),
            }
        )
    successful = [r.nme for r in rows if r.nme is not None]
    return {
        "attempted_cases": len(rows),
        "complete_cases": len(successful),
        "failed_cases": len(rows) - len(successful),
        "groups": len({r.group_id for r in rows}),
        "reference_kind": rows[0].reference_kind,
        "successful_case_mean_nme": float(np.mean(successful)) if successful else None,
        "unconditional_auc": float(
            np.mean(
                [
                    max(0, 1 - r.nme / auc_threshold) if r.nme is not None else 0
                    for r in rows
                ]
            )
        ),
        "auc_threshold_ipd": auc_threshold,
        "pck_threshold_ipd": pck_threshold,
        "eligible_landmarks": int(eligible.sum()),
        "observed_landmarks": int(observed.sum()),
        "pck_including_failures": float(hit.sum() / eligible.sum()),
        "per_landmark": per_landmark,
        "failure_policy": "every failed attempt stays in case/PCK/AUC denominators; NME conditional on complete cases is labeled",
        "reference_limit": "errors are relative to recorded annotations; blinding is declared, not independently attested",
        "covariance_limit": "perturbation repeatability is not localization accuracy",
        "rater_agreement_limit": "disagreement needs at least two raters; zero with one rater is not evidence of agreement",
    }


def evaluate_study(
    cases: tuple[ValidationCase, ...], minimum_raters: int = 1
) -> tuple[dict[str, Any], tuple[LocalizationRow, ...]]:
    validate_study(cases)
    rows = tuple(evaluate_case(c, minimum_raters) for c in cases)
    report = {
        "format": "facemath.localization_validation.v4",
        "reference_protocol_sha256": REFERENCE_PROTOCOL_SHA256,
        "minimum_raters": minimum_raters,
        "normalization": "reference_eye_corner_midpoints_source_pixels",
        "alignment": "NONE: detector error measured in the original image coordinate system",
        "observer_recipe": next(
            (r.observer_recipe for r in rows if r.observer_recipe), None
        ),
        "splits": {
            split: summarize(tuple(r for r in rows if r.split == split))
            for split in ("development", "calibration", "evaluation")
            if any(r.split == split for r in rows)
        },
        "cases": [r.to_dict() for r in rows],
        "independence": "curator-supplied group/lineage IDs; no identity inference",
    }
    report["content_sha256"] = object_digest(report)
    return report, rows
