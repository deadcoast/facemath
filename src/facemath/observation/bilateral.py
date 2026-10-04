"""Eye-defined bilateral configuration on the shared reference8 relation graph."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import ObservationError, frozen_array, object_digest
from .intrinsic_sets import EYE_VECTOR, Interval, SourceRelations, outer
from .reference import REFERENCE_NAMES, REFERENCE_PROTOCOL_SHA256


ROWS = (("outer_eyes", "Outer eye corners", 0, 3),
        ("inner_eyes", "Inner eye corners", 1, 2),
        ("mouth", "Mouth corners", 5, 6),
        ("nose", "Nose midline offset", 4, 4),
        ("chin", "Chin midline offset", 7, 7))


def expressions() -> np.ndarray:
    points = np.eye(8)
    center = points[:4].mean(axis=0)
    return np.array([(points[l] + points[r] - 2 * center, points[r] - points[l])
                     if l != r else (points[l] - center, np.zeros(8))
                     for _, _, l, r in ROWS])


def nominal_components(points: np.ndarray, supported: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result, valid = np.zeros((5, 2)), np.zeros(5, bool)
    eye = EYE_VECTOR @ points
    square = float(eye @ eye)
    if not np.all(supported[:4]) or square <= 0:
        return result, valid
    for i, (_, _, left, right) in enumerate(ROWS):
        if supported[left] and supported[right]:
            horizontal, vertical = expressions()[i] @ points
            result[i] = (eye @ horizontal / square,
                         (eye[0] * vertical[1] - eye[1] * vertical[0]) / square)
            valid[i] = True
    return result, valid


def norm_box(lower: np.ndarray, upper: np.ndarray) -> Interval:
    """Euclidean norm extrema over an interval rectangle (an outer relaxation)."""
    near = np.maximum(np.maximum(lower, -upper), 0.)
    far = np.maximum(np.abs(lower), np.abs(upper))
    return outer(float(np.linalg.norm(near)), float(np.linalg.norm(far)), nonnegative=True)


@dataclass(frozen=True)
class BilateralComparison:
    nominal: np.ndarray
    nominal_valid: np.ndarray
    source_lower: np.ndarray
    source_upper: np.ndarray
    source_available: np.ndarray
    source_reasons: tuple
    source_segments: np.ndarray
    source_axes: np.ndarray
    tolerance: float
    calibration_sha256: str | None
    source_quantiles: tuple[float | None, float | None]
    coefficients: np.ndarray = field(default_factory=expressions)
    recipe_sha256: str = field(init=False)

    def __post_init__(self):
        shapes = {"nominal": (2, 5, 2), "nominal_valid": (2, 5),
                  "source_lower": (2, 5, 2), "source_upper": (2, 5, 2),
                  "source_available": (2, 5), "source_segments": (2, 5, 2, 2),
                  "source_axes": (2, 2, 2), "coefficients": (5, 2, 8)}
        for key, shape in shapes.items():
            value = frozen_array(getattr(self, key), bool if key.endswith(("valid", "available")) else float)
            if value.shape != shape:
                raise ObservationError(f"bilateral {key} must have shape {shape}")
            object.__setattr__(self, key, value)
        if np.any(self.source_lower > self.source_upper):
            raise ObservationError("bilateral reference interval endpoints reversed")
        if len(self.source_reasons) != 2 or any(len(row) != 5 for row in self.source_reasons):
            raise ObservationError("bilateral missingness must retain all five rows for each source")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ObservationError("bilateral tolerance must be finite and positive")
        object.__setattr__(self, "recipe_sha256", object_digest({
            "implementation": "reference8.bilateral.v23", "protocol": REFERENCE_PROTOCOL_SHA256,
            "rows": ROWS, "coefficients": self.coefficients.tolist(), "tolerance": self.tolerance}))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"bilateral_" + key: getattr(self, key) for key in
                ("nominal", "nominal_valid", "source_lower", "source_upper", "source_available",
                 "source_segments", "source_axes", "coefficients")}

    def to_dict(self) -> dict:
        rows = []
        for i, (key, label, left, right) in enumerate(ROWS):
            components = [self.nominal[s, i].tolist() if self.nominal_valid[s, i] else None for s in range(2)]
            magnitudes = [float(np.linalg.norm(c)) if c is not None else None for c in components]
            valid = bool(np.all(self.nominal_valid[:, i]))
            delta = self.nominal[1, i] - self.nominal[0, i] if valid else None
            change = float(np.linalg.norm(delta)) if valid else None
            sources = []
            for s in range(2):
                bounded = self.source_available[s, i]
                magnitude = norm_box(self.source_lower[s, i], self.source_upper[s, i]) if bounded else None
                sources.append({"components": np.column_stack((self.source_lower[s, i], self.source_upper[s, i])).tolist() if bounded else None,
                    "magnitude": [magnitude.lower, magnitude.upper] if bounded else None,
                    "reason": self.source_reasons[s][i]})
            delta_bounds, change_bounds = None, None
            if np.all(self.source_available[:, i]):
                intervals = [outer(self.source_lower[1, i, c] - self.source_upper[0, i, c],
                                   self.source_upper[1, i, c] - self.source_lower[0, i, c]) for c in range(2)]
                delta_bounds = [[v.lower, v.upper] for v in intervals]
                change_interval = norm_box(np.array([v.lower for v in intervals]), np.array([v.upper for v in intervals]))
                change_bounds = [change_interval.lower, change_interval.upper]
            observed_status = "unavailable" if change is None else "within_tolerance" if change <= self.tolerance else "outside_tolerance"
            reference_status = ("unavailable" if change_bounds is None else
                "within_tolerance_under_envelope" if change_bounds[1] <= self.tolerance else
                "outside_tolerance_under_envelope" if change_bounds[0] > self.tolerance else "unresolved_at_tolerance")
            rows.append({"key": key, "label": label, "index": i, "kind": "paired_corners" if left != right else "midline_offset",
                "points": [REFERENCE_NAMES[j] for j in sorted({left, right})],
                "observed_a": components[0], "observed_b": components[1],
                "observed_magnitude_a": magnitudes[0], "observed_magnitude_b": magnitudes[1],
                "observed_delta_b_minus_a": delta.tolist() if valid else None,
                "observed_magnitude_delta_b_minus_a": magnitudes[1] - magnitudes[0] if valid else None,
                "observed_configuration_change": change, "observed_status": observed_status,
                "source_intervals": sources, "reference_delta_b_minus_a": delta_bounds,
                "reference_configuration_change": change_bounds, "reference_status": reference_status,
                "source_segments": [self.source_segments[s, i].tolist() if self.nominal_valid[s, i] else None for s in range(2)]})
        return {"format": "facemath.bilateral_comparison.v23", "recipe_sha256": self.recipe_sha256,
            "rows": rows, "prespecified_rows": 5, "unit": "own_reference_eye_spans", "tolerance": self.tolerance,
            "components": ["horizontal", "vertical"], "delta_convention": "B minus A",
            "calibration_sha256": self.calibration_sha256, "source_quantiles": list(self.source_quantiles),
            "source_axes": self.source_axes.tolist(),
            "observed_counts": {status: sum(r["observed_status"] == status for r in rows) for status in
                                ("within_tolerance", "outside_tolerance", "unavailable")},
            "reference_counts": {status: sum(r["reference_status"] == status for r in rows) for status in
                                 ("within_tolerance_under_envelope", "outside_tolerance_under_envelope", "unresolved_at_tolerance", "unavailable")},
            "meaning": "paired right-minus-reflected-left components; central points use signed midline offset, not doubled reflection distance",
            "axis_convention": "existing eye-midpoint line and clockwise perpendicular; no fitted symmetry axis or reflection registration",
            "bounds_meaning": "analytic outer rectangles from the same source error graph and uncertain eye span; dependent rows, not resemblance probability",
            "limits": "projected configuration under pose and expression; no anatomical accuracy or physical 3D symmetry claim",
            "missingness": "five fixed rows; array zeros under false masks are unavailable, not symmetric"}


def measure_bilateral(raw: np.ndarray, supported: np.ndarray, evaluators: tuple[SourceRelations, SourceRelations],
                      quantiles: tuple, calibration_sha256: str | None, config) -> BilateralComparison:
    values, valid = zip(*(nominal_components(raw[s], supported[s]) for s in range(2)))
    lower, upper, available = np.zeros((2, 5, 2)), np.zeros((2, 5, 2)), np.zeros((2, 5), bool)
    reasons, segments, axes = [], np.zeros((2, 5, 2, 2)), np.zeros((2, 2, 2))
    coefficients = expressions()
    for s, evaluator in enumerate(evaluators):
        source_reasons = []
        for i in range(5):
            if calibration_sha256 is None or quantiles[s] is None:
                source_reasons.append("no_localization_calibration" if calibration_sha256 is None else "unbounded_localization_calibration")
                continue
            bounds = [evaluator.projected_ratio(coefficients[i, c], perpendicular=c == 1) for c in range(2)]
            available[s, i] = all(v.available for v in bounds)
            if available[s, i]:
                lower[s, i], upper[s, i] = [v.lower for v in bounds], [v.upper for v in bounds]
            source_reasons.append("analytic_outer_bound" if available[s, i] else next(v.reason for v in bounds if not v.available))
        reasons.append(tuple(source_reasons))
        eye = EYE_VECTOR @ raw[s]
        span = float(np.linalg.norm(eye))
        if np.all(supported[s, :4]) and span > 0:
            center, unit = raw[s, :4].mean(axis=0), eye / span
            perpendicular = np.array([-unit[1], unit[0]])
            axes[s] = np.array([center - span * perpendicular, center + 3 * span * perpendicular])
            for i, (_, _, left, right) in enumerate(ROWS):
                if valid[s][i]:
                    point = raw[s, left]
                    expected = point - (2 if left != right else 1) * ((point - center) @ unit) * unit
                    segments[s, i] = np.array([expected, raw[s, right]])
    return BilateralComparison(np.array(values), np.array(valid), lower, upper, available, tuple(reasons),
        segments, axes, config.intrinsic_length_tolerance, calibration_sha256, quantiles)
