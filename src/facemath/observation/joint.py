"""Development-fitted full-shape error geometry, calibrated as ONE scalar score.

The 16 coordinates are ordered [point0.x, point0.y, ..., point7.y]. The matrix
is an equal-group residual second moment with FROZEN regularization, not an
assumption of Gaussian errors and not the observer's perturbation covariance.
Independent group calibration sets the radius. Nothing here edits an Observation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping
from numbers import Real

import numpy as np

from .bias import BiasCalibration, DevelopmentBias, fit_development_bias, ocular_basis
from .contracts import Observation, ObservationError, frozen_array, object_digest
from .error_model import LocalizationCalibration, _validate_rows, fit_localization_calibration
from .reference import REFERENCE_NAMES, REFERENCE_PROTOCOL_SHA256, check_digest, mapped_prediction
from .validation import LocalizationRow

JOINT_MODE = "development_joint_ocular"
JOINT_SCORE = "group_max_reference8_full16_mahalanobis_after_development_only_mean_and_scatter.v7"
SCATTER_RECIPE = "equal_group_mean_case_residual_outer_product_plus_fixed_isotropic_shrinkage_and_floor.v7"
DIMENSION = 16


def regularized_scatter(moments: np.ndarray, shrinkage: float, floor: float) -> np.ndarray:
    """No Bessel correction: this is a score-shape estimator, not an unbiased claim."""
    if (not isinstance(shrinkage, Real) or not isinstance(floor, Real)
            or isinstance(shrinkage, (bool, np.bool_)) or isinstance(floor, (bool, np.bool_))
            or not np.isfinite([shrinkage, floor]).all() or not 0 <= shrinkage <= 1 or floor <= 0):
        raise ObservationError("joint shrinkage must be in [0,1] and ocular floor strictly positive")
    m = np.asarray(moments, float)
    if m.ndim != 3 or m.shape[1:] != (DIMENSION, DIMENSION) or not len(m) or not np.isfinite(m).all():
        raise ObservationError("joint second moments require finite [groups,16,16]")
    if not np.allclose(m, m.swapaxes(-1, -2), rtol=1e-12, atol=1e-15):
        raise ObservationError("joint second moments must be symmetric")
    if np.linalg.eigvalsh(m).min() < -1e-12 * max(1., float(np.max(np.abs(m)))):
        raise ObservationError("joint second moments must be positive semidefinite")
    try:
        with np.errstate(over="raise", invalid="raise"):
            empirical = m.mean(axis=0)
            covariance = ((1-shrinkage)*empirical
                          + (shrinkage*np.trace(empirical)/DIMENSION + np.square(np.float64(floor)))*np.eye(DIMENSION))
    except (FloatingPointError, OverflowError) as exc:
        raise ObservationError("joint scatter regularization overflowed its numerical scale") from exc
    covariance = (covariance+covariance.T)/2
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
        raise ObservationError("regularized joint scatter is not numerically positive definite")
    return covariance


@dataclass(frozen=True)
class DevelopmentScatter:
    bias_sha256: str
    group_ids: tuple[str, ...]
    group_case_counts: tuple[int, ...]
    group_second_moments: np.ndarray
    covariance: np.ndarray
    shrinkage: float = .02
    floor_eye_spans: float = .001

    def __post_init__(self) -> None:
        check_digest(self.bias_sha256, "joint scatter bias binding")
        groups, counts = tuple(self.group_ids), tuple(self.group_case_counts)
        if (not groups or any(not isinstance(g, str) or not g for g in groups)
                or len(set(groups)) != len(groups) or len(counts) != len(groups)
                or any(type(n) is not int or n < 1 for n in counts)):
            raise ObservationError("joint scatter group inventory is invalid")
        moments, covariance = frozen_array(self.group_second_moments), frozen_array(self.covariance)
        if moments.shape != (len(groups), DIMENSION, DIMENSION) or covariance.shape != (DIMENSION, DIMENSION):
            raise ObservationError("joint scatter array shapes differ from group inventory")
        expected = regularized_scatter(moments, self.shrinkage, self.floor_eye_spans)
        if not np.allclose(expected, covariance, atol=1e-15, rtol=1e-12):
            raise ObservationError("joint scatter differs from retained moments and frozen regularization")
        for name, value in (("group_ids", groups), ("group_case_counts", counts),
                            ("group_second_moments", moments), ("covariance", covariance)):
            object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"format": "facemath.development_scatter.v7", "recipe": SCATTER_RECIPE,
                "protocol_sha256": REFERENCE_PROTOCOL_SHA256, "coordinate_order": "reference8_interleaved_xy",
                "bias_sha256": self.bias_sha256, "group_ids": list(self.group_ids),
                "group_case_counts": list(self.group_case_counts),
                "group_second_moments": self.group_second_moments.tolist(),
                "covariance": self.covariance.tolist(), "shrinkage": self.shrinkage,
                "floor_eye_spans": self.floor_eye_spans,
                "meaning": "development residual score-shape matrix; not perturbation repeatability or a Gaussian posterior"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> DevelopmentScatter:
        from dataclasses import fields
        if (obj.get("format") != "facemath.development_scatter.v7" or obj.get("recipe") != SCATTER_RECIPE
                or obj.get("protocol_sha256") != REFERENCE_PROTOCOL_SHA256
                or obj.get("coordinate_order") != "reference8_interleaved_xy"):
            raise ObservationError("unsupported joint scatter representation")
        return cls(**{f.name: obj[f.name] for f in fields(cls)})


def _eligible(row: LocalizationRow, minimum_raters: int) -> bool:
    return bool(row.failure is None and np.all(row.observed) and np.all(row.reference_visible)
                and np.all(row.rater_count >= minimum_raters))


def fit_development_scatter(rows: tuple[LocalizationRow, ...], bias: DevelopmentBias, *,
                            shrinkage: float = .02, floor_eye_spans: float = .001) -> DevelopmentScatter:
    _validate_rows(rows)
    if any(r.split != "development" for r in rows):
        raise ObservationError("joint scatter accepts development rows ONLY")
    if (tuple(sorted({r.group_id for r in rows})) != bias.development_groups
            or tuple(sorted(r.image_sha256 for r in rows)) != bias.development_images):
        raise ObservationError("scatter and bias development inventories differ")
    replayed_bias = fit_development_bias(rows, minimum_groups=bias.minimum_groups,
                                         minimum_raters=bias.minimum_raters)
    if replayed_bias.digest != bias.digest:
        raise ObservationError("scatter rows differ from the bound development bias fit")
    groups: dict[str, list[np.ndarray]] = {}
    for row in rows:
        if row.observer_recipe and row.observer_recipe != bias.observer_recipe_sha256:
            raise ObservationError("joint scatter observer recipe mismatch")
        if row.reference_kind != bias.reference_kind:
            raise ObservationError("joint scatter reference provenance mismatch")
        if not _eligible(row, bias.minimum_raters):
            continue
        try:
            residual = (np.linalg.solve(ocular_basis(row.predicted_xy), row.error_vectors_px.T).T-bias.beta).reshape(DIMENSION)
        except ObservationError:
            continue
        groups.setdefault(row.group_id, []).append(np.outer(residual, residual))
    names = tuple(sorted(groups))
    if names != bias.fitted_groups:
        raise ObservationError("joint scatter and bias fitted group sets differ")
    moments = np.stack([np.mean(groups[g], axis=0) for g in names])
    return DevelopmentScatter(bias.digest, names, tuple(len(groups[g]) for g in names), moments,
                              regularized_scatter(moments, shrinkage, floor_eye_spans), shrinkage, floor_eye_spans)


@dataclass(frozen=True)
class JointCalibration(BiasCalibration):
    """One 16D ellipsoid per source; the two sources are NOT assumed independent."""
    scatter: DevelopmentScatter

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.scatter.bias_sha256 != self.bias.digest or self.scatter.group_ids != self.bias.fitted_groups
                or sum(self.scatter.group_case_counts) != self.bias.fitted_cases):
            raise ObservationError("joint scatter has lost its development-bias context")

    def source_joint_covariance(self, observation: Observation) -> np.ndarray:
        xy, _, _ = mapped_prediction(observation)
        transport = np.kron(np.eye(8), ocular_basis(xy))
        return frozen_array(transport @ self.scatter.covariance @ transport.T)

    def source_covariance(self, observation: Observation) -> np.ndarray:
        covariance = self.source_joint_covariance(observation).reshape(8, 2, 8, 2)
        return frozen_array(np.stack([covariance[i, :, i, :] for i in range(8)]))

    def to_dict(self) -> dict[str, Any]:
        residual = self.residual.to_dict()
        residual.update(format="facemath.joint_residual_calibration.v7", score=JOINT_SCORE,
                        requires_bias_sha256=self.bias.digest, requires_scatter_sha256=self.scatter.digest)
        return {"format": "facemath.localization_calibration.v7", "score": JOINT_SCORE,
                "bias": self.bias.to_dict(), "scatter": self.scatter.to_dict(),
                "residual_calibration": residual, "raw_geometry_changed": False,
                "uncertainty_set": "c + q L z, ||z||_2 <= 1, L L^T = source full16 covariance",
                "not": "not independent marginal ellipses, not new facial coordinates, not Gaussian probability"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> JointCalibration:
        if (obj.get("format") != "facemath.localization_calibration.v7" or obj.get("score") != JOINT_SCORE
                or obj.get("raw_geometry_changed") is not False):
            raise ObservationError("unsupported joint error calibration")
        bias = DevelopmentBias.from_dict(obj["bias"])
        scatter = DevelopmentScatter.from_dict(obj["scatter"])
        residual = dict(obj["residual_calibration"])
        if (residual.get("format") != "facemath.joint_residual_calibration.v7"
                or residual.get("score") != JOINT_SCORE
                or residual.pop("requires_bias_sha256", None) != bias.digest
                or residual.pop("requires_scatter_sha256", None) != scatter.digest):
            raise ObservationError("joint residual fragment detached from required bias/scatter")
        residual["format"] = "facemath.localization_calibration.v4"
        return cls(LocalizationCalibration.from_dict(residual), bias, scatter)


def joint_scored_rows(rows: tuple[LocalizationRow, ...], bias: DevelopmentBias,
                       scatter: DevelopmentScatter) -> tuple[LocalizationRow, ...]:
    _validate_rows(rows)
    if scatter.bias_sha256 != bias.digest:
        raise ObservationError("joint score scatter/bias mismatch")
    chol = np.linalg.cholesky(scatter.covariance)
    result = []
    marginal = np.stack([scatter.covariance[2*i:2*i+2, 2*i:2*i+2] for i in range(8)])
    for row in rows:
        if row.split == "development":
            raise ObservationError("joint calibration and coverage exclude development rows")
        bias.require_disjoint(row.image_sha256, row.group_id, row.lineage_ids)
        if row.observer_recipe and row.observer_recipe != bias.observer_recipe_sha256:
            raise ObservationError("joint score observer recipe mismatch")
        if row.reference_kind != bias.reference_kind:
            raise ObservationError("joint score reference provenance mismatch")
        score, marginal_score = None, np.zeros(8)
        if _eligible(row, bias.minimum_raters):
            try:
                residual = np.linalg.solve(ocular_basis(row.predicted_xy), row.error_vectors_px.T).T-bias.beta
                score = float(np.linalg.norm(np.linalg.solve(chol, residual.reshape(DIMENSION))))
                marginal_score = np.sqrt(np.maximum(0., np.einsum("ni,ni->n", residual,
                    np.linalg.solve(marginal, residual[..., None])[..., 0])))
            except ObservationError:
                pass  # Invalid source gauge remains unavailable, never a zero score.
        # The joint score is NOT max(marginal_score). Raw vectors and NME remain raw.
        result.append(replace(row, simultaneous_score=score, mahalanobis=marginal_score))
    return tuple(result)


def fit_joint_calibration(rows: tuple[LocalizationRow, ...], *, study_sha256: str, domain: str,
                          alpha: float = .1, minimum_raters: int = 1, minimum_development_groups: int = 5,
                          shrinkage: float = .02, floor_eye_spans: float = .001
                          ) -> tuple[JointCalibration, tuple[LocalizationRow, ...]]:
    _validate_rows(rows)
    development = tuple(r for r in rows if r.split == "development")
    bias = fit_development_bias(development, minimum_groups=minimum_development_groups,
                                minimum_raters=minimum_raters)
    scatter = fit_development_scatter(development, bias, shrinkage=shrinkage, floor_eye_spans=floor_eye_spans)
    scored = joint_scored_rows(tuple(r for r in rows if r.split != "development"), bias, scatter)
    residual = fit_localization_calibration(scored, study_sha256=study_sha256, domain=domain,
                                            alpha=alpha, minimum_raters=minimum_raters)
    return JointCalibration(residual, bias, scatter), scored
