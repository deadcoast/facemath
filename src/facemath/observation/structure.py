"""Named projected structure, computed on the instrument's existing relation graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import ObservationError, frozen_array, object_digest
from .intrinsic_sets import Array, EYE_VECTOR, Interval, SourceRelations, absolute_difference, displacement, outer
from .reference import REFERENCE_NAMES, REFERENCE_PROTOCOL_SHA256


@dataclass(frozen=True)
class StructuralMeasurement:
    key: str
    label: str
    operation: str
    a: Array
    b: Array
    tolerance: float

    def __post_init__(self) -> None:
        if self.operation not in {"width", "height", "offset", "asymmetry"}:
            raise ObservationError("unknown structural operation")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ObservationError("structural tolerance must be finite and positive")
        for name in ("a", "b"):
            value = frozen_array(getattr(self, name))
            if value.shape != (8,) or abs(value.sum()) > 1e-14:
                raise ObservationError("structural expressions must be translation invariant reference8 vectors")
            object.__setattr__(self, name, value)

    @property
    def dependencies(self) -> Array:
        mask = (self.a != 0) | (self.b != 0)
        if self.operation != "asymmetry":
            mask |= EYE_VECTOR != 0
        return mask

    def metadata(self) -> dict:
        return {"key": self.key, "label": self.label, "operation": self.operation,
                "unit": "signed_length_contrast" if self.operation == "asymmetry" else "own_reference_eye_spans",
                "tolerance": self.tolerance, "required_points": [REFERENCE_NAMES[i] for i in np.flatnonzero(self.dependencies)],
                "a_coefficients": self.a.tolist(), "b_coefficients": self.b.tolist()}


def structural_registry(config: Any) -> tuple[StructuralMeasurement, ...]:
    point = np.eye(8)
    eye = point[:4].mean(axis=0)
    mouth = (point[5]+point[6])/2
    zero = np.zeros(8)
    specs = [
        ("left_eye_width", "Viewer-left eye width", "width", displacement(1, 0)),
        ("right_eye_width", "Viewer-right eye width", "width", displacement(3, 2)),
        ("inner_eye_gap", "Inner-eye gap", "width", displacement(2, 1)),
        ("mouth_width", "Mouth width", "width", displacement(6, 5)),
        ("eye_to_nose_height", "Eye line to nose height", "height", point[4]-eye),
        ("eye_to_mouth_height", "Eye line to mouth height", "height", mouth-eye),
        ("eye_to_chin_height", "Eye line to chin height", "height", point[7]-eye),
        ("nose_to_mouth_height", "Nose to mouth height", "height", mouth-point[4]),
        ("mouth_to_chin_height", "Mouth to chin height", "height", point[7]-mouth),
        ("nose_lateral_offset", "Nose lateral offset", "offset", point[4]-eye),
        ("mouth_lateral_offset", "Mouth lateral offset", "offset", mouth-eye),
        ("chin_lateral_offset", "Chin lateral offset", "offset", point[7]-eye),
        ("mouth_corner_height_difference", "Right minus left mouth height", "height", displacement(6, 5)),
    ]
    result = [StructuralMeasurement(k, label, op, a, zero, config.intrinsic_length_tolerance)
              for k, label, op, a in specs]
    result.append(StructuralMeasurement("eye_width_asymmetry", "Eye-width asymmetry", "asymmetry",
        displacement(1, 0), displacement(3, 2), config.intrinsic_asymmetry_tolerance))
    return tuple(result)


def observed_value(points: Array, supported: Array, spec: StructuralMeasurement) -> tuple[float, bool]:
    if not np.all(supported[spec.dependencies]):
        return 0., False
    vector = spec.a @ points
    if spec.operation == "asymmetry":
        left, right = float(np.linalg.norm(vector)), float(np.linalg.norm(spec.b @ points))
        return ((left-right)/(left+right), True) if left+right > 0 else (0., False)
    eye = EYE_VECTOR @ points
    squared = float(eye @ eye)
    if squared <= 0:
        return 0., False
    if spec.operation == "width":
        return float(np.linalg.norm(vector)/np.sqrt(squared)), True
    if spec.operation == "height":
        return float((eye[0]*vector[1]-eye[1]*vector[0])/squared), True
    return float(eye @ vector/squared), True


def reference_bound(evaluator: SourceRelations, spec: StructuralMeasurement) -> Interval:
    if spec.operation == "width":
        return evaluator.length_ratio(spec.a)
    if spec.operation == "asymmetry":
        return evaluator.asymmetry(spec.a, spec.b)
    return evaluator.projected_ratio(spec.a, perpendicular=spec.operation == "height")


@dataclass(frozen=True)
class StructuralComparison:
    measurements: tuple[StructuralMeasurement, ...]
    nominal: Array
    nominal_valid: Array
    source_lower: Array
    source_upper: Array
    source_available: Array
    source_reasons: tuple[tuple[str, ...], tuple[str, ...]]
    delta_lower: Array
    delta_upper: Array
    gap_lower: Array
    gap_upper: Array
    pair_available: Array
    calibration_sha256: str | None
    source_quantiles: tuple[float | None, float | None]
    recipe_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.measurements)
        if n != 14 or len({m.key for m in self.measurements}) != n:
            raise ObservationError("structural registry must retain fourteen distinct measurements")
        for name in ("nominal", "nominal_valid", "source_lower", "source_upper", "source_available",
                     "delta_lower", "delta_upper", "gap_lower", "gap_upper", "pair_available"):
            value = frozen_array(getattr(self, name), bool if name.endswith(("valid", "available")) else float)
            shape = (2, n) if name.startswith(("nominal", "source")) else (n,)
            if value.shape != shape:
                raise ObservationError(f"structural {name} must have shape {shape}")
            object.__setattr__(self, name, value)
        if len(self.source_reasons) != 2 or any(len(r) != n for r in self.source_reasons):
            raise ObservationError("structural reasons must retain both complete inventories")
        for low, high in ((self.source_lower, self.source_upper), (self.delta_lower, self.delta_upper), (self.gap_lower, self.gap_upper)):
            if np.any(low > high):
                raise ObservationError("structural interval endpoints reversed")
        if not np.array_equal(self.pair_available, np.all(self.source_available, axis=0)):
            raise ObservationError("structural pair bounds require both source bounds")
        if len(self.source_quantiles) != 2 or any(q is not None and (q < 0 or not np.isfinite(q)) for q in self.source_quantiles):
            raise ObservationError("structural radii require explicit per-source finite values or missingness")
        object.__setattr__(self, "recipe_sha256", object_digest({"implementation": "reference8.structure.v18",
            "protocol": REFERENCE_PROTOCOL_SHA256, "eye_vector": EYE_VECTOR.tolist(),
            "measurements": [m.metadata() for m in self.measurements]}))

    def arrays(self) -> dict[str, Array]:
        names = ("nominal", "nominal_valid", "source_lower", "source_upper", "source_available",
                 "delta_lower", "delta_upper", "gap_lower", "gap_upper", "pair_available")
        return {**{"structure_"+k: getattr(self, k) for k in names},
                "structure_expression_a": frozen_array(np.stack([m.a for m in self.measurements])),
                "structure_expression_b": frozen_array(np.stack([m.b for m in self.measurements]))}

    def to_dict(self) -> dict:
        rows = []
        for i, spec in enumerate(self.measurements):
            observed = bool(np.all(self.nominal_valid[:, i]))
            delta = float(self.nominal[1, i]-self.nominal[0, i]) if observed else None
            status = ("unavailable" if not self.pair_available[i] else
                      "within_tolerance_under_envelope" if self.gap_upper[i] <= spec.tolerance else
                      "outside_tolerance_under_envelope" if self.gap_lower[i] > spec.tolerance else "unresolved_at_tolerance")
            rows.append({**spec.metadata(), "index": i,
                "observed_a": float(self.nominal[0, i]) if self.nominal_valid[0, i] else None,
                "observed_b": float(self.nominal[1, i]) if self.nominal_valid[1, i] else None,
                "observed_delta_b_minus_a": delta, "observed_absolute_gap": abs(delta) if delta is not None else None,
                "observed_status": "unavailable" if delta is None else "within_tolerance" if abs(delta) <= spec.tolerance else "outside_tolerance",
                "reference_status": status,
                "source_intervals": [{"lower": float(self.source_lower[s, i]) if self.source_available[s, i] else None,
                    "upper": float(self.source_upper[s, i]) if self.source_available[s, i] else None,
                    "reason": self.source_reasons[s][i]} for s in range(2)],
                "reference_delta_b_minus_a": [float(self.delta_lower[i]), float(self.delta_upper[i])] if self.pair_available[i] else None,
                "reference_absolute_gap": [float(self.gap_lower[i]), float(self.gap_upper[i])] if self.pair_available[i] else None})
        return {"format": "facemath.structural_comparison.v18", "recipe_sha256": self.recipe_sha256,
                "prespecified_measurements": 14, "calibration_sha256": self.calibration_sha256,
                "source_quantiles": list(self.source_quantiles),
                "observed_counts": {k: sum(r["observed_status"] == k for r in rows)
                                    for k in ("within_tolerance", "outside_tolerance", "unavailable")},
                "reference_counts": {k: sum(r["reference_status"] == k for r in rows) for k in
                    ("within_tolerance_under_envelope", "outside_tolerance_under_envelope", "unresolved_at_tolerance", "unavailable")},
                "rows": rows, "delta_convention": "B minus A",
                "axis_convention": "viewer-left to viewer-right eye midpoint; positive height is clockwise perpendicular in stored-raster xy",
                "normalization": "each source's own uncertain reference eye span; eye-width asymmetry uses sum of eye widths",
                "meaning": "projected configuration under image pose and expression; dependent measurements, not identity probability or physical depth",
                "bounds_meaning": "analytic outer intervals from the existing shared reference-error sets; no refit, new alignment or independent votes",
                "missingness": "all fourteen measurements remain; false array masks mark zero sentinels"}


def measure_structure(raw: Array, supported: Array, evaluators: tuple[SourceRelations, SourceRelations],
                      quantiles: tuple[float | None, float | None], calibration_sha256: str | None,
                      config: Any) -> StructuralComparison:
    specs = structural_registry(config)
    nominal, valid = np.zeros((2, 14)), np.zeros((2, 14), bool)
    bounds = []
    for s, evaluator in enumerate(evaluators):
        for i, spec in enumerate(specs):
            nominal[s, i], valid[s, i] = observed_value(raw[s], supported[s], spec)
        reason = "no_localization_calibration" if calibration_sha256 is None else "unbounded_localization_calibration"
        bounds.append([reference_bound(evaluator, spec) if quantiles[s] is not None and calibration_sha256 is not None
                       else Interval.missing(reason) for spec in specs])
    lower, upper = np.array([[b.lower for b in row] for row in bounds]), np.array([[b.upper for b in row] for row in bounds])
    available = np.array([[b.available for b in row] for row in bounds])
    pair_available = np.all(available, axis=0)
    dl, du, gl, gu = (np.zeros(14) for _ in range(4))
    for i in np.flatnonzero(pair_available):
        a, b = bounds[0][i], bounds[1][i]
        delta = outer(b.lower-a.upper, b.upper-a.lower)
        gap = absolute_difference(a, b)
        dl[i], du[i], gl[i], gu[i] = delta.lower, delta.upper, gap.lower, gap.upper
    return StructuralComparison(specs, nominal, valid, lower, upper, available,
        tuple(tuple(b.reason for b in row) for row in bounds), dl, du, gl, gu, pair_available,
        calibration_sha256, quantiles)
