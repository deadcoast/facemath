"""Registration-independent reference-shape relations inside the shared instrument.

The primary canonical profile is unchanged. This profile consumes its exact
Observation objects and optional already-qualified localization envelope. It
never aligns points to each other before constructing intrinsic relations.

The two sources have uncertain reference eye spans. Calibrated error sets are
propagated into distances, angles, signed area and bilateral contrasts. A
quotient-space bound also covers a fresh optimal proper-rotation fit for EVERY
admissible pair of normalized reference shapes; no fitted transform is assumed
known. Only the existing eight-point reference protocol is covered.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
from math import pi
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bias import BiasEnvelope, ocular_basis
from .canonical import CanonicalState
from .contracts import ObservationError, frozen_array, object_digest
from .error_model import LocalizationEnvelope
from .intrinsic_sets import (
    Array, EYE_VECTOR, N_POINTS, EllipticSource, Interval, SourceRelations,
    absolute_difference, displacement, pad,
)
from .reference import REFERENCE_NAMES, REFERENCE_PROTOCOL_SHA256, mapped_prediction

FAMILIES = ("distance", "angle", "signed_double_area", "asymmetry")
STATUS = {0: "no_localization_calibration", 1: "within_tolerance_under_envelope",
          2: "outside_tolerance_under_envelope", 3: "unresolved_at_tolerance",
          4: "unbounded_or_unobserved_dependency"}


@dataclass(frozen=True)
class Relation:
    key: str
    family: str
    indices: tuple[int, ...]
    a: Array
    b: Array
    unit: str
    tolerance: float
    description: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES or not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ObservationError("invalid intrinsic relation definition")
        for name in ("a", "b"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))

    def metadata(self) -> dict:
        return {"key": self.key, "family": self.family, "indices": list(self.indices),
                "point_names": [REFERENCE_NAMES[i] for i in self.indices],
                "unit": self.unit, "tolerance": self.tolerance, "description": self.description}


def relation_registry(config: Any) -> tuple[Relation, ...]:
    specs: list[Relation] = []
    zero = np.zeros(N_POINTS)
    for i, j in combinations(range(N_POINTS), 2):
        specs.append(Relation(f"distance.{i}.{j}", "distance", (i, j), displacement(i, j), zero,
            "own_reference_eye_spans", config.intrinsic_length_tolerance,
            "distance divided by this source's uncertain reference eye-midpoint span"))
    for j in range(N_POINTS):
        for i, k in combinations([x for x in range(N_POINTS) if x != j], 2):
            specs.append(Relation(f"angle.{i}.{j}.{k}", "angle", (i, j, k),
                displacement(i, j), displacement(k, j), "radians", config.intrinsic_angle_tolerance_rad,
                "unsigned included angle at the middle listed landmark"))
    for i, j, k in combinations(range(N_POINTS), 3):
        specs.append(Relation(f"area.{i}.{j}.{k}", "signed_double_area", (i, j, k),
            displacement(j, i), displacement(k, i), "own_reference_eye_spans_squared",
            config.intrinsic_area_tolerance,
            "oriented double triangle area; proper-rotation invariant, reflection sensitive under fixed labels"))
    left_eye = np.array([.5, .5, 0., 0., 0., 0., 0., 0.])
    right_eye = np.array([0., 0., .5, .5, 0., 0., 0., 0.])
    nose, chin = np.eye(N_POINTS)[4], np.eye(N_POINTS)[7]
    definitions = (
        ("eye_width", displacement(0, 1), displacement(3, 2), (0, 1, 2, 3)),
        ("nose_to_eye", nose-left_eye, nose-right_eye, (0, 1, 2, 3, 4)),
        ("nose_to_mouth", displacement(4, 5), displacement(4, 6), (4, 5, 6)),
        ("chin_to_eye", chin-left_eye, chin-right_eye, (0, 1, 2, 3, 7)),
    )
    for name, a, b, indices in definitions:
        specs.append(Relation(f"asymmetry.{name}", "asymmetry", indices, a, b,
            "signed_length_contrast", config.intrinsic_asymmetry_tolerance,
            "(viewer-left length - viewer-right length) / sum of both lengths"))
    return tuple(specs)


def _nominal(points: Array, support: NDArray[np.bool_], spec: Relation) -> tuple[float, bool]:
    a, b = spec.a, spec.b
    required = (a != 0) | (b != 0)
    if spec.family in ("distance", "signed_double_area"):
        required |= EYE_VECTOR != 0
    if not np.all(support[required]):
        return 0., False
    u, v = a @ points, b @ points
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    d = float(np.linalg.norm(EYE_VECTOR @ points))
    if spec.family == "distance":
        return (lu/d, True) if d > 0 else (0., False)
    if spec.family == "angle":
        if min(lu, lv) <= 1e-14:
            return 0., False
        # atan2 is numerically better than acos near collinear arms.
        return float(np.arctan2(abs(u[0]*v[1]-u[1]*v[0]), u@v)), True
    if spec.family == "signed_double_area":
        return (float((u[0]*v[1]-u[1]*v[0])/d**2), True) if d > 0 else (0., False)
    return ((lu-lv)/(lu+lv), True) if lu+lv > 1e-14 else (0., False)


def nominal_values(points: Array, supported: NDArray[np.bool_], relations: tuple[Relation, ...]) -> tuple[Array, NDArray[np.bool_]]:
    pairs = [_nominal(points, supported, r) for r in relations]
    return np.array([x[0] for x in pairs]), np.array([x[1] for x in pairs], bool)


def _bound(evaluator: SourceRelations, spec: Relation) -> Interval:
    if spec.family == "distance":
        return evaluator.length_ratio(spec.a)
    if spec.family == "angle":
        return evaluator.angle(spec.a, spec.b)
    if spec.family == "signed_double_area":
        return evaluator.signed_area_ratio(spec.a, spec.b)
    return evaluator.asymmetry(spec.a, spec.b)


def normalized_centered(points: Array) -> Array:
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        raise ObservationError("proper shape fit requires finite reference8 points")
    d = float(np.linalg.norm(EYE_VECTOR @ points))
    if d <= 1e-14:
        raise ObservationError("shape normalization eye span collapses")
    return (points-points.mean(axis=0))/d


def proper_shape_distance(a: Array, b: Array) -> tuple[float, Array, Array]:
    """Eye-normalized proper-rotation Procrustes RMS; NO reflection or scale fit.

    Translation is removed by centering. Separate eye-span normalization already
    removes scale. The returned row-vector rotation minimizes ||A R - B||_F.
    This does not overwrite the instrument's original robust registration.
    """
    x, y = normalized_centered(a), normalized_centered(b)
    u, _, vt = np.linalg.svd(x.T @ y)
    sign = 1. if np.linalg.det(u @ vt) >= 0 else -1.
    rotation = u @ np.diag([1., sign]) @ vt
    residual = x @ rotation-y
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))), rotation, residual


@dataclass(frozen=True)
class ShapeBound:
    observed_rms: float | None
    centered_rms: float | None
    lower: float | None
    upper: float | None
    available: bool
    reason: str
    edm_lower: float | None
    lipschitz_lower: float | None
    lipschitz_upper: float | None
    candidate_upper: float | None
    source_deviation_radii: Array
    candidate_rotation: Array

    def __post_init__(self) -> None:
        for name in ("source_deviation_radii", "candidate_rotation"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))

    def to_dict(self) -> dict:
        return {"metric": "intrinsic.eye_normalized_proper_procrustes_rms",
                "observed_rms": self.observed_rms, "reference_set_center_rms": self.centered_rms,
                "lower": self.lower, "upper": self.upper, "available": self.available,
                "reason": self.reason, "distance_matrix_lower": self.edm_lower,
                "quotient_lipschitz_lower": self.lipschitz_lower,
                "quotient_lipschitz_upper": self.lipschitz_upper,
                "candidate_rotation_upper": self.candidate_upper,
                "meaning": "outer range of optimally proper-rotation-refitted RMS over admissible reference shapes, with their own uncertain eye spans",
                "not": "not fixed observed-gauge displacement, not exact extrema, not a resemblance percentage",
                "arrays": ["intrinsic_shape_deviation_radii", "intrinsic_shape_candidate_rotation"]}


def bound_refitted_shape(raw: Array, evaluators: tuple[SourceRelations, SourceRelations],
                        relation_gap_lower: Array, calibrated: bool) -> ShapeBound:
    all_visible = all(np.all(x.source.supported) for x in evaluators)
    observed = proper_shape_distance(*raw)[0] if all_visible else None
    centers = [x.source.centers for x in evaluators]
    d0 = [float(np.linalg.norm(EYE_VECTOR @ p)) for p in centers]
    centered, rotation, residual = (None, np.eye(2), np.zeros((8, 2)))
    if all_visible and min(d0) > 1e-14:
        centered, rotation, residual = proper_shape_distance(*centers)
    radii = np.zeros((2, N_POINTS))
    reason = ("no_localization_calibration" if not calibrated else
              "reference8_incomplete" if not all_visible else
              "normalizer_not_separated_from_zero" if any(
                  not e.eye_span.available or e.eye_span.lower <= 0 for e in evaluators) else "bounded")
    if reason != "bounded" or centered is None:
        return ShapeBound(observed, centered, None, None, False, reason, None, None, None, None, radii, rotation)
    for s, evaluator in enumerate(evaluators):
        d = evaluator.eye_span
        reciprocal_error = max(abs(1/d.lower-1/d0[s]), abs(1/d.upper-1/d0[s]))
        for i in range(N_POINTS):
            a = np.eye(N_POINTS)[i] - np.ones(N_POINTS)/N_POINTS
            center = a @ evaluator.source.centers
            radii[s, i] = evaluator.radius(a)/d.lower + np.linalg.norm(center)*reciprocal_error
    budget = float(np.sqrt(np.mean(radii[0]**2)) + np.sqrt(np.mean(radii[1]**2)))
    # |D_ij(X)-D_ij(Y)| <= ||e_i-e_j|| and centered sum identity give
    # d_Procrustes >= sqrt(sum_{i<j} gap_ij**2) / n.
    edm_lower = float(np.linalg.norm(relation_gap_lower[:28])/N_POINTS)
    lipschitz_lower = max(0., centered-budget)
    lipschitz_upper = centered+budget
    candidate_upper = float(np.sqrt(np.mean((np.linalg.norm(residual, axis=1)+radii.sum(axis=0))**2)))
    lo = max(0., max(edm_lower, lipschitz_lower)-pad(edm_lower, centered, budget))
    hi = min(lipschitz_upper, candidate_upper)+pad(lipschitz_upper, candidate_upper)
    if not np.isfinite([lo, hi]).all():
        return ShapeBound(observed, centered, None, None, False, "nonfinite_arithmetic", None, None, None, None, radii, rotation)
    return ShapeBound(observed, centered, lo, hi, True, "bounded", edm_lower,
                      lipschitz_lower, lipschitz_upper, candidate_upper, radii, rotation)


@dataclass(frozen=True)
class JointRelationAudit:
    """Same calibrated set, deliberately relaxed to its marginal product for comparison."""
    covariance: Array
    marginal_covariance: Array
    marginal_lower: Array
    marginal_upper: Array
    marginal_available: NDArray[np.bool_]
    marginal_shape: ShapeBound
    mode_eigenvalues: Array = field(init=False, repr=False)
    mode_vectors: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shapes = {"covariance": (2,16,16), "marginal_covariance": (2,8,2,2),
                  "marginal_lower": (2,256), "marginal_upper": (2,256), "marginal_available": (2,256)}
        for name, shape in shapes.items():
            value = frozen_array(getattr(self, name), bool if name == "marginal_available" else float)
            if value.shape != shape:
                raise ObservationError(f"joint relation audit {name} has wrong shape")
            object.__setattr__(self, name, value)
        if np.any(self.marginal_lower > self.marginal_upper):
            raise ObservationError("joint relaxation interval endpoints reversed")
        eigenvalues, vectors = np.linalg.eigh(self.covariance)
        eigenvalues, vectors = eigenvalues[:, ::-1], vectors[:, :, ::-1]
        # Deterministic display sign; tied eigenbases are not unique mechanisms.
        for source in range(2):
            for k in range(16):
                v = vectors[source, :, k]
                if v[np.argmax(np.abs(v))] < 0:
                    vectors[source, :, k] *= -1
        object.__setattr__(self, "mode_eigenvalues", frozen_array(eigenvalues))
        object.__setattr__(self, "mode_vectors", frozen_array(vectors))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"intrinsic_joint_"+name: getattr(self, name) for name in
                ("covariance", "marginal_covariance", "marginal_lower", "marginal_upper", "marginal_available",
                 "mode_eigenvalues", "mode_vectors")}

    def to_dict(self) -> dict[str, Any]:
        translation = np.tile(np.eye(2), (8,1))/np.sqrt(8.)
        total = np.trace(self.covariance, axis1=1, axis2=2)
        fractions = [float(np.trace(translation.T@c@translation)/t) if t > 0 else 0.
                     for c, t in zip(self.covariance, total)]
        return {"set": "one calibrated full16 ellipsoid per source; shared latent ball across all points",
                "coordinate_order": "reference8_interleaved_xy", "common_translation_variance_fraction": fractions,
                "relaxation": "product of the SAME set's 8 marginal ellipses, SAME q; not a separately fitted model",
                "relaxed_shape_distance": self.marginal_shape.to_dict(),
                "statistical_status": "no Gaussian assumption or independent landmark/source errors; no extra coverage from tighter algebra",
                "source_covariance_array": "intrinsic_joint_covariance[2,16,16]",
                "raw_repeatability_preserved": "intrinsic_source_covariance remains the observed perturbation covariance"}


@dataclass(frozen=True)
class IntrinsicProfile:
    relations: tuple[Relation, ...]
    raw_points: Array
    source_centers: Array
    source_covariance: Array
    source_chart: Array
    source_supported: NDArray[np.bool_]
    quantile: float | None
    calibration_sha256: str | None
    conditional_pair_coverage: float | None
    nominal: Array
    nominal_valid: NDArray[np.bool_]
    centered: Array
    centered_valid: NDArray[np.bool_]
    source_lower: Array
    source_upper: Array
    source_available: NDArray[np.bool_]
    source_reasons: tuple[tuple[str, ...], tuple[str, ...]]
    eye_span_bounds: Array
    eye_span_available: NDArray[np.bool_]
    gap_lower: Array
    gap_upper: Array
    gap_available: NDArray[np.bool_]
    statuses: NDArray[np.int64]
    shape: ShapeBound
    observation_digests: tuple[str, str]
    scope: str
    directions: int
    joint: JointRelationAudit | None = None
    recipe_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        masks = {"source_supported", "nominal_valid", "centered_valid", "source_available",
                 "eye_span_available", "gap_available"}
        for name in ("raw_points", "source_centers", "source_covariance", "source_chart", "source_supported",
                     "nominal", "nominal_valid", "centered", "centered_valid", "source_lower", "source_upper",
                     "source_available", "eye_span_bounds", "eye_span_available", "gap_lower", "gap_upper",
                     "gap_available", "statuses"):
            dtype = bool if name in masks else np.int64 if name == "statuses" else float
            object.__setattr__(self, name, frozen_array(getattr(self, name), dtype))
        n = len(self.relations)
        shapes = {
            "raw_points": (2,8,2), "source_centers": (2,8,2),
            "source_covariance": (2,8,2,2), "source_chart": (2,3,3),
            "source_supported": (2,8), "eye_span_bounds": (2,2),
            "eye_span_available": (2,),
        }
        shapes.update({k: (2,n) for k in ("nominal", "nominal_valid", "centered", "centered_valid",
                      "source_lower", "source_upper", "source_available")})
        shapes.update({k: (n,) for k in ("gap_lower", "gap_upper", "gap_available", "statuses")})
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ObservationError(f"intrinsic {name} must have shape {shape}")
        if n != 256 or len({r.key for r in self.relations}) != n:
            raise ObservationError("intrinsic reference8 registry must have 256 distinct relations")
        if (len(self.source_reasons) != 2 or any(len(r) != n for r in self.source_reasons)
                or np.any(~np.isin(self.statuses, list(STATUS)))):
            raise ObservationError("invalid intrinsic status/reason inventory")
        if np.any(self.source_lower > self.source_upper) or np.any(self.gap_lower > self.gap_upper):
            raise ObservationError("intrinsic interval endpoints are reversed")
        if self.quantile is not None and (not np.isfinite(self.quantile) or self.quantile < 0):
            raise ObservationError("invalid intrinsic localization multiplier")
        recipe = {"implementation": "intrinsic.reference8.outer_relations.v7",
                  "error_set": "joint_full16" if self.joint is not None else "product_marginals", "protocol": REFERENCE_PROTOCOL_SHA256,
                  "directions": self.directions, "relations": [r.metadata() for r in self.relations]}
        object.__setattr__(self, "recipe_sha256", object_digest(recipe))

    def arrays(self) -> dict[str, np.ndarray]:
        names = ("raw_points", "source_centers", "source_covariance", "source_chart", "source_supported",
                 "nominal", "nominal_valid", "centered", "centered_valid", "source_lower", "source_upper",
                 "source_available", "eye_span_bounds", "eye_span_available", "gap_lower", "gap_upper",
                 "gap_available", "statuses")
        result = {"intrinsic_"+name: getattr(self, name) for name in names}
        result.update(intrinsic_relation_a=frozen_array(np.stack([r.a for r in self.relations])),
                      intrinsic_relation_b=frozen_array(np.stack([r.b for r in self.relations])),
                      intrinsic_shape_deviation_radii=self.shape.source_deviation_radii,
                      intrinsic_shape_candidate_rotation=self.shape.candidate_rotation)
        theta = np.arange(self.directions)*(2*pi/self.directions)
        result["intrinsic_support_normals"] = frozen_array(np.column_stack((np.cos(theta), np.sin(theta))))
        if self.joint is not None:
            result.update(self.joint.arrays())
        return result

    @property
    def reference_error_covariance(self) -> Array:
        return self.joint.marginal_covariance if self.joint is not None else self.source_covariance

    def to_dict(self) -> dict[str, Any]:
        rows = []
        for i, spec in enumerate(self.relations):
            raw_ok = bool(np.all(self.nominal_valid[:, i]))
            rows.append({**spec.metadata(), "index": i,
                "observed_a": float(self.nominal[0, i]) if self.nominal_valid[0, i] else None,
                "observed_b": float(self.nominal[1, i]) if self.nominal_valid[1, i] else None,
                "observed_absolute_gap": float(abs(self.nominal[0, i]-self.nominal[1, i])) if raw_ok else None,
                "reference_set_center_a": float(self.centered[0, i]) if self.centered_valid[0, i] else None,
                "reference_set_center_b": float(self.centered[1, i]) if self.centered_valid[1, i] else None,
                "source_intervals": [{"lower": float(self.source_lower[s, i]) if self.source_available[s, i] else None,
                    "upper": float(self.source_upper[s, i]) if self.source_available[s, i] else None,
                    "reason": self.source_reasons[s][i]} for s in range(2)],
                "reference_gap_lower": float(self.gap_lower[i]) if self.gap_available[i] else None,
                "reference_gap_upper": float(self.gap_upper[i]) if self.gap_available[i] else None,
                "status": STATUS[int(self.statuses[i])],
                "array_columns": {"observed": f"intrinsic_nominal[:,{i}]",
                                  "bounds": f"intrinsic_gap_lower[{i}],intrinsic_gap_upper[{i}]"}})
        families = []
        for family in FAMILIES:
            ids = [i for i, r in enumerate(self.relations) if r.family == family]
            code = self.statuses[ids]
            inside, outside = int(np.sum(code == 1)), int(np.sum(code == 2))
            families.append({"family": family, "prespecified_relations": len(ids),
                "nominally_observable": int(np.all(self.nominal_valid[:, ids], axis=0).sum()),
                "bounded_relations": int(self.gap_available[ids].sum()),
                "within": inside, "outside": outside, "unresolved_or_unavailable": len(ids)-inside-outside,
                "within_tolerance_fraction_bounds": [inside/len(ids), (len(ids)-outside)/len(ids)] if self.calibration_sha256 is not None else None,
                "fraction_meaning": "deterministic fraction of these prespecified relations; dependencies retained, not resemblance probability"})
        return {"format": "facemath.intrinsic_profile.v7",
            "joint_error_geometry": self.joint.to_dict() if self.joint is not None else None, "recipe_sha256": self.recipe_sha256,
            "observation_digests": list(self.observation_digests), "reference_names": list(REFERENCE_NAMES),
            "calibration_sha256": self.calibration_sha256, "scope": self.scope,
            "calibrated": self.calibration_sha256 is not None, "quantile": self.quantile,
            "calibration_bound_state": "absent" if self.calibration_sha256 is None else "unbounded" if self.quantile is None else "finite",
            "conditional_pair_coverage_lower_bound": self.conditional_pair_coverage,
            "coverage_condition": "only if both sources satisfy their frozen calibration assumptions; no new coverage from more relations",
            "gauge": "intrinsic source relations; uncertain reference eye-span normalization, NOT a fixed observed pair alignment",
            "normalizer_intervals_in_source_charts": [{"lower": float(self.eye_span_bounds[s, 0]) if self.eye_span_available[s] else None,
                "upper": float(self.eye_span_bounds[s, 1]) if self.eye_span_available[s] else None,
                "separated_from_zero": bool(self.eye_span_available[s] and self.eye_span_bounds[s, 0] > 0)} for s in range(2)],
            "shape_distance": self.shape.to_dict(), "families": families, "rows": rows,
            "non_independence": "256 algebraically related summaries of 8 landmarks per source; never fused into 256 independent votes",
            "missingness": "false masks identify zero array sentinels; unknowns remain in every prespecified fraction denominator",
            "numerics": "analytic outer bounds with float64 padding and finite-angle correction; not formal interval arithmetic or exact extrema",
            "raw_geometry_changed": False, "population_similarity_percent": None}


def measure_intrinsic(state: CanonicalState, localization: LocalizationEnvelope | BiasEnvelope | None) -> IntrinsicProfile:
    """Consume the same image-bound observations and already-checked error model."""
    specs = relation_registry(state.config)
    raw, centers, covariances, charts, supports = [], [], [], [], []
    error_covariances, joint_covariances = [], []
    q = localization.calibration.quantile if localization is not None else None
    unavailable_reason = "no_localization_calibration" if localization is None else "unbounded_localization_calibration"
    for s, observation in enumerate((state.observation_a, state.observation_b)):
        xy, covariance, support = mapped_prediction(observation)
        if not np.all(support[:4]):
            raise ObservationError("intrinsic source chart requires observed four eye corners")
        inverse = np.linalg.inv(ocular_basis(xy))
        origin = xy[:4].mean(axis=0)
        chart = np.eye(3)
        chart[:2, :2], chart[:2, 2] = inverse, -inverse @ origin
        point_chart = (xy-origin) @ inverse.T
        source_bias = localization.source_bias_px[s] if isinstance(localization, BiasEnvelope) else np.zeros((8, 2))
        center = point_chart-source_bias @ inverse.T
        raw.append(point_chart)
        centers.append(center)
        covariances.append(np.einsum("ij,njk,lk->nil", inverse, covariance, inverse))
        model_cov = localization.calibration.source_covariance(observation) if localization is not None else covariance
        error_covariances.append(np.einsum("ij,njk,lk->nil", inverse, model_cov, inverse))
        joint = localization.calibration.source_joint_covariance(observation) if localization is not None else None
        transport = np.kron(np.eye(8), inverse)
        joint_covariances.append(transport@joint@transport.T if joint is not None else None)
        charts.append(chart)
        supports.append(support)
    raw, centers, covariance, charts, supports = map(np.stack, (raw, centers, covariances, charts, supports))
    evaluators = tuple(SourceRelations(EllipticSource(centers[s], error_covariances[s], supports[s], q if q is not None else 0.,
                                                     state.config.intrinsic_directions, joint_covariances[s])) for s in range(2))
    nominal, nominal_valid, center_values, center_valid = [], [], [], []
    lower, upper, available = np.zeros((2, len(specs))), np.zeros((2, len(specs))), np.zeros((2, len(specs)), bool)
    reasons: list[tuple[str, ...]] = []
    for s, evaluator in enumerate(evaluators):
        values, valid = nominal_values(raw[s], supports[s], specs)
        nominal.append(values); nominal_valid.append(valid)
        values, valid = nominal_values(centers[s], supports[s], specs)
        center_values.append(values); center_valid.append(valid)
        bounds = [_bound(evaluator, r) if q is not None else Interval.missing(unavailable_reason) for r in specs]
        lower[s] = [b.lower for b in bounds]
        upper[s] = [b.upper for b in bounds]
        available[s] = [b.available for b in bounds]
        reasons.append(tuple(b.reason for b in bounds))
    gap_lower, gap_upper, gap_available = np.zeros(len(specs)), np.zeros(len(specs)), np.zeros(len(specs), bool)
    statuses = np.zeros(len(specs), np.int64)
    for i, spec in enumerate(specs):
        if q is None:
            statuses[i] = 0 if localization is None else 4
            continue
        if not np.all(available[:, i]):
            statuses[i] = 4
            continue
        gap = absolute_difference(Interval(lower[0, i], upper[0, i]), Interval(lower[1, i], upper[1, i]))
        gap_lower[i], gap_upper[i], gap_available[i] = gap.lower, gap.upper, gap.available
        statuses[i] = 1 if gap.upper <= spec.tolerance else 2 if gap.lower > spec.tolerance else 3
    shape = bound_refitted_shape(raw, evaluators, gap_lower, q is not None)
    if localization is not None and q is None:
        shape = replace(shape, reason=unavailable_reason)
    span_intervals = [e.eye_span if q is not None else Interval.missing(unavailable_reason) for e in evaluators]
    joint_audit = None
    if all(c is not None for c in joint_covariances):
        marginal_evaluators = tuple(SourceRelations(EllipticSource(centers[s], error_covariances[s],
            supports[s], q if q is not None else 0., state.config.intrinsic_directions)) for s in range(2))
        marginal_bounds = [[_bound(e, r) if q is not None else Interval.missing(unavailable_reason)
                            for r in specs] for e in marginal_evaluators]
        ml = np.array([[b.lower for b in row] for row in marginal_bounds])
        mu = np.array([[b.upper for b in row] for row in marginal_bounds])
        mv = np.array([[b.available for b in row] for row in marginal_bounds])
        marginal_gaps = np.array([absolute_difference(marginal_bounds[0][i], marginal_bounds[1][i]).lower
                                 for i in range(len(specs))])
        marginal_shape = bound_refitted_shape(raw, marginal_evaluators, marginal_gaps, q is not None)
        joint_audit = JointRelationAudit(np.stack(joint_covariances), np.stack(error_covariances),
                                        ml, mu, mv, marginal_shape)
    return IntrinsicProfile(specs, raw, centers, covariance, charts, supports, q,
        localization.calibration.digest if localization is not None else None,
        max(0., 1-2*localization.calibration.alpha) if localization is not None else None,
        np.stack(nominal), np.stack(nominal_valid), np.stack(center_values), np.stack(center_valid),
        lower, upper, available, tuple(reasons),
        np.array([[b.lower, b.upper] for b in span_intervals]), np.array([b.available for b in span_intervals]),
        gap_lower, gap_upper, gap_available, statuses, shape,
        (state.observation_a.digest, state.observation_b.digest),
        "nominal_observation_only" if localization is None else
        "synthetic_only" if localization.calibration.reference_kind == "synthetic_fixture" else "conditional_reference_domain",
        state.config.intrinsic_directions, joint=joint_audit)
