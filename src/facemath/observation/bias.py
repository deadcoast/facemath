"""Development-only error centering in an observer-defined ocular frame.

This is an error model, NOT a shape correction. Raw Observation/CanonicalState
coordinates never change. Independent calibration scores the remaining error;
the deployed reference ellipse may be translated but the geometry stays raw.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, cast

import numpy as np

from .contracts import Observation, ObservationError, frozen_array, object_digest
from .error_model import (
    DirectionalBounds, LocalizationCalibration, LocalizationEnvelope, _validate_rows, assess_pair,
    ellipse_gap_bounds, evaluate_coverage, fit_localization_calibration,
)
from .reference import (
    REFERENCE_NAMES, REFERENCE_PROTOCOL_SHA256, check_digest, mapped_prediction,
    observer_recipe,
)
from .validation import LocalizationRow


BIAS_RECIPE = "reference8_equal_group_mean_observed_ocular_vector_frame.v1"
SCORE_RECIPE = "group_max_centered_reference8_mahalanobis_after_development_only_fit"


def ocular_basis(predicted_xy: np.ndarray) -> np.ndarray:
    """Columns are the predicted eye-midpoint vector and its +90 degree rotation.

    Multiplying by this matrix transports a dimensionless vector into source
    pixels. No reference label or reference normalizer is needed at deployment.
    """
    p = np.asarray(predicted_xy, dtype=float)
    if p.shape != (8, 2) or not np.isfinite(p).all():
        raise ObservationError("ocular frame requires finite reference8 predictions")
    v = p[2:4].mean(axis=0) - p[:2].mean(axis=0)
    if np.linalg.norm(v) < 8:
        raise ObservationError("observed eye-midpoint span must be at least eight pixels")
    return np.column_stack((v, (-v[1], v[0])))


@dataclass(frozen=True)
class DevelopmentBias:
    """Frozen equal-group estimator; failed development cases remain in accounting."""

    beta: np.ndarray
    group_vectors: np.ndarray
    fitted_groups: tuple[str, ...]
    development_groups: tuple[str, ...]
    development_images: tuple[str, ...]
    development_lineages: tuple[str, ...]
    development_reference_digests: tuple[str, ...]
    attempted_cases: int
    fitted_cases: int
    rejected_cases: tuple[tuple[str, str], ...]
    observer_recipe_sha256: str
    reference_kind: str
    minimum_raters: int
    minimum_groups: int

    def __post_init__(self) -> None:
        beta = frozen_array(self.beta)
        gv = frozen_array(self.group_vectors)
        if beta.shape != (8, 2) or gv.shape != (len(self.fitted_groups), 8, 2):
            raise ObservationError("bias requires reference8 group vectors")
        if (type(self.minimum_groups) is not int or self.minimum_groups < 2
                or len(self.fitted_groups) < self.minimum_groups):
            raise ObservationError("insufficient independent development groups for bias fitting")
        if type(self.minimum_raters) is not int or self.minimum_raters < 1:
            raise ObservationError("minimum_raters must be positive")
        if (type(self.attempted_cases) is not int or type(self.fitted_cases) is not int
                or self.fitted_cases < len(self.fitted_groups)
                or self.attempted_cases != self.fitted_cases + len(self.rejected_cases)):
            raise ObservationError("inconsistent development attempt accounting")
        if not np.allclose(beta, gv.mean(axis=0), atol=1e-14, rtol=1e-12):
            raise ObservationError("bias mean does not match retained equal-group inputs")
        check_digest(self.observer_recipe_sha256, "bias observer recipe")
        if self.reference_kind not in ("human_blind", "external_dataset", "synthetic_fixture"):
            raise ObservationError("unsupported development reference provenance")
        for name in ("fitted_groups", "development_groups", "development_images",
                     "development_lineages", "development_reference_digests"):
            values = tuple(getattr(self, name))
            if any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
                raise ObservationError(f"invalid or duplicate {name}")
            object.__setattr__(self, name, values)
        if not set(self.fitted_groups) <= set(self.development_groups):
            raise ObservationError("fitted group absent from development inventory")
        for digest in self.development_images + self.development_reference_digests:
            check_digest(digest, "development source")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "group_vectors", gv)
        object.__setattr__(self, "rejected_cases", tuple(tuple(x) for x in self.rejected_cases))

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        d = asdict(self)
        d["beta"], d["group_vectors"] = self.beta.tolist(), self.group_vectors.tolist()
        return {"format": "facemath.development_bias.v5", "recipe": BIAS_RECIPE,
                "protocol_sha256": REFERENCE_PROTOCOL_SHA256, **d,
                "interpretation": "expected prediction-minus-reference error, not true anatomy; fitted only on development groups"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> DevelopmentBias:
        from dataclasses import fields
        if (obj.get("format") != "facemath.development_bias.v5" or obj.get("recipe") != BIAS_RECIPE
                or obj.get("protocol_sha256") != REFERENCE_PROTOCOL_SHA256):
            raise ObservationError("unsupported development bias recipe")
        return cls(**{f.name: obj[f.name] for f in fields(cls)})

    def require_disjoint(self, image: str, group: str, lineages: tuple[str, ...]) -> None:
        if (image in self.development_images or group in self.development_groups
                or set(lineages) & set(self.development_lineages)):
            raise ObservationError("development bias/query or held-out split leakage")

    def vectors(self, predicted_xy: np.ndarray) -> np.ndarray:
        return frozen_array(self.beta @ ocular_basis(predicted_xy).T)


def fit_development_bias(rows: tuple[LocalizationRow, ...], *, minimum_groups: int = 5,
                         minimum_raters: int = 1) -> DevelopmentBias:
    """Reject rather than silently ingest any calibration/evaluation row."""
    _validate_rows(rows)
    if any(r.split != "development" for r in rows):
        raise ObservationError("bias fit accepts development rows ONLY")
    recipes = {r.observer_recipe for r in rows if r.observer_recipe}
    kinds = {r.reference_kind for r in rows}
    if len(recipes) != 1 or len(kinds) != 1:
        raise ObservationError("bias fit requires one observer recipe and reference kind")
    grouped: dict[str, list[np.ndarray]] = {}
    rejected: list[tuple[str, str]] = []
    fitted = 0
    for r in rows:
        if not (np.all(r.observed) and np.all(r.reference_visible)
                and np.all(r.rater_count >= minimum_raters) and r.failure is None):
            rejected.append((r.case_id, r.failure or "incomplete_reference8"))
            continue
        try:
            basis = ocular_basis(r.predicted_xy)
        except ObservationError as exc:
            rejected.append((r.case_id, str(exc)))
            continue
        local = np.linalg.solve(basis, r.error_vectors_px.T).T
        grouped.setdefault(r.group_id, []).append(local)
        fitted += 1
    names = tuple(sorted(grouped))
    if len(names) < minimum_groups:
        raise ObservationError("insufficient independent development groups for bias fitting")
    group_vectors = np.stack([np.mean(grouped[g], axis=0) for g in names])
    return DevelopmentBias(group_vectors.mean(axis=0), group_vectors, names,
        tuple(sorted({r.group_id for r in rows})), tuple(sorted(r.image_sha256 for r in rows)),
        tuple(sorted({s for r in rows for s in r.lineage_ids})),
        tuple(sorted({s for r in rows for s in r.reference_digests})), len(rows), fitted,
        tuple(rejected), next(iter(recipes)), next(iter(kinds)), minimum_raters, minimum_groups)


def centered_rows(rows: tuple[LocalizationRow, ...], bias: DevelopmentBias) -> tuple[LocalizationRow, ...]:
    """New error records; raw predictions and error vectors remain untouched.

    Only mahalanobis/simultaneous_score are replaced for the centered conformal
    score. Keeping raw NME and raw coordinates avoids reporting an adjusted
    prediction as if the detector had actually produced it.
    """
    _validate_rows(rows)
    out = []
    for r in rows:
        if r.split == "development":
            raise ObservationError("centered calibration/coverage cannot use development rows")
        bias.require_disjoint(r.image_sha256, r.group_id, r.lineage_ids)
        if r.observer_recipe and r.observer_recipe != bias.observer_recipe_sha256:
            raise ObservationError("bias observer recipe mismatch")
        if r.reference_kind != bias.reference_kind:
            raise ObservationError("bias reference kind mismatch")
        maha = np.zeros(8)
        complete = bool(np.all(r.observed) and np.all(r.reference_visible)
                        and np.all(r.rater_count >= bias.minimum_raters) and r.failure is None)
        score = None
        if complete:
            try:
                residual = r.error_vectors_px - bias.vectors(r.predicted_xy)
                maha = np.sqrt(np.maximum(0, np.einsum("ni,nij,nj->n", residual,
                                                   np.linalg.inv(r.covariance), residual)))
                score = float(maha.max())
            except ObservationError:
                # This is an invalid observed gauge, not an invented zero score.
                pass
        out.append(replace(r, mahalanobis=maha, simultaneous_score=score))
    return tuple(out)


@dataclass(frozen=True)
class BiasCalibration:
    """A v5 artifact cannot be mistaken for a zero-centered v4 calibration."""

    residual: LocalizationCalibration
    bias: DevelopmentBias

    def __post_init__(self) -> None:
        if (self.residual.observer_recipe_sha256 != self.bias.observer_recipe_sha256
                or self.residual.minimum_raters != self.bias.minimum_raters
                or self.residual.reference_kind != self.bias.reference_kind):
            raise ObservationError("bias and residual calibration contracts differ")
        if (set(self.residual.calibration_images) & set(self.bias.development_images)
                or set(dict(self.residual.group_scores)) & set(self.bias.development_groups)
                or set(self.residual.calibration_lineages) & set(self.bias.development_lineages)):
            raise ObservationError("development/calibration leakage")

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    @property
    def quantile(self) -> float | None:
        return self.residual.quantile

    @property
    def alpha(self) -> float:
        return self.residual.alpha

    @property
    def reference_kind(self) -> str:
        return self.residual.reference_kind

    def require_query(self, observation: Observation, group_id: str, domain: str,
                      lineage_ids: tuple[str, ...] = ()) -> None:
        self.residual.require_query(observation, group_id, domain, lineage_ids)
        self.bias.require_disjoint(observation.image_sha256, group_id, lineage_ids)
        _, _, valid = mapped_prediction(observation)
        if not np.all(valid[:4]):
            raise ObservationError("bias deployment requires four observable eye corners")

    def to_dict(self) -> dict[str, Any]:
        residual = self.residual.to_dict()
        # A detached fragment must NOT be accepted by the zero-centered loader.
        # Its quantile is meaningful only after this particular development fit.
        residual["format"] = "facemath.centered_residual_calibration.v5"
        residual["requires_bias_sha256"] = self.bias.digest
        return {"format": "facemath.localization_calibration.v5", "bias": self.bias.to_dict(),
                "residual_calibration": residual,
                "score": SCORE_RECIPE,
                "raw_geometry_changed": False}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> BiasCalibration:
        if (obj.get("format") != "facemath.localization_calibration.v5"
                or obj.get("score") != SCORE_RECIPE or obj.get("raw_geometry_changed") is not False):
            raise ObservationError("unsupported bias-qualified calibration")
        bias = DevelopmentBias.from_dict(obj["bias"])
        residual = dict(obj["residual_calibration"])
        if (residual.get("format") != "facemath.centered_residual_calibration.v5"
                or residual.pop("requires_bias_sha256", None) != bias.digest):
            raise ObservationError("centered residual calibration lost its required bias context")
        residual["format"] = "facemath.localization_calibration.v4"
        return cls(LocalizationCalibration.from_dict(residual), bias)


def fit_bias_calibration(rows: tuple[LocalizationRow, ...], *, study_sha256: str, domain: str,
                         alpha: float = .1, minimum_raters: int = 1,
                         minimum_development_groups: int = 5) -> tuple[BiasCalibration, tuple[LocalizationRow, ...]]:
    _validate_rows(rows)
    bias = fit_development_bias(tuple(r for r in rows if r.split == "development"),
                               minimum_groups=minimum_development_groups, minimum_raters=minimum_raters)
    scored = centered_rows(tuple(r for r in rows if r.split != "development"), bias)
    residual = fit_localization_calibration(scored, study_sha256=study_sha256, domain=domain,
                                           alpha=alpha, minimum_raters=minimum_raters)
    return BiasCalibration(residual, bias), scored


@dataclass(frozen=True)
class BiasEnvelope:
    base: LocalizationEnvelope
    source_bias_px: np.ndarray
    canonical_bias_px: np.ndarray
    reference_center_points: np.ndarray
    model_centered_gap_ipd: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.base.calibration, BiasCalibration) or self.base.directional is None:
            raise ObservationError("bias envelope requires a bias calibration and directional bounds")
        expected_shapes = {"source_bias_px": (2, 8, 2), "canonical_bias_px": (2, 8, 2),
                           "reference_center_points": (2, 8, 2), "model_centered_gap_ipd": (8,)}
        for name, shape in expected_shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ObservationError(f"bias envelope {name} has an invalid shape")
        for key in ("source_bias_px", "canonical_bias_px", "reference_center_points", "model_centered_gap_ipd"):
            object.__setattr__(self, key, frozen_array(getattr(self, key)))

    @property
    def calibration(self) -> BiasCalibration:
        return cast(BiasCalibration, self.base.calibration)

    @property
    def directional(self) -> DirectionalBounds:
        return cast(DirectionalBounds, self.base.directional)

    @property
    def supported(self) -> np.ndarray:
        return self.base.supported

    def arrays(self) -> dict[str, np.ndarray]:
        return {**self.base.arrays(), **{"localization_" + k: getattr(self, k) for k in
                ("source_bias_px", "canonical_bias_px", "reference_center_points", "model_centered_gap_ipd")}}

    def to_dict(self) -> dict[str, Any]:
        d = self.base.to_dict()
        d.update(bias_sha256=self.calibration.bias.digest, raw_geometry_changed=False,
                 centering="reference ellipse centers are prediction minus development-only bias; displayed geometry remains raw")
        for i, row in enumerate(d["rows"]):
            row["model_centered_gap_ipd"] = float(self.model_centered_gap_ipd[i]) if self.supported[i] else None
        return d


def assess_bias_pair(state: Any, calibration: BiasCalibration, **kwargs: Any) -> BiasEnvelope:
    base = assess_pair(state, calibration, **kwargs)
    biases, canonical = [], []
    for obs, matrix in zip((state.observation_a, state.observation_b),
                           (state.source_to_canonical_a, state.source_to_canonical_b)):
        xy, _, _ = mapped_prediction(obs)
        b = calibration.bias.vectors(xy)
        biases.append(b)
        canonical.append(b @ matrix[:2, :2].T)
    canonical = np.stack(canonical)
    centers = base.canonical_points - canonical
    delta = (centers[0] - centers[1]) / state.normalization_span_px
    directional = ellipse_gap_bounds(delta, base.directional.normalized_covariances, calibration.quantile)
    lo, hi = directional.lower_ipd.copy(), directional.upper_ipd.copy()
    lo[~base.supported], hi[~base.supported] = 0, 0
    revised = replace(base, lower_ipd=lo, upper_ipd=hi, directional=directional)
    return BiasEnvelope(revised, np.stack(biases), canonical, centers, np.linalg.norm(delta, axis=1))


def load_localization_calibration(obj: Mapping[str, Any]) -> LocalizationCalibration | BiasCalibration:
    if obj.get("format") == "facemath.localization_calibration.v5":
        return BiasCalibration.from_dict(obj)
    return LocalizationCalibration.from_dict(obj)
