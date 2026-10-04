"""All measurements consume the same canonical state and anatomical field.

No similarity probability is manufactured. A fraction within a declared
geometric tolerance IS a valid empirical percentage; a population percentile is
a separate operation. Sensitivity intervals are not accuracy confidence bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from ..geometry import pairwise_distance_matrix

from .anatomy import eye_span
from .canonical import CanonicalState, fit_anatomy, symmetric_normalizer
from .contracts import ObservationError, frozen_array
from .field import AnatomicalField, FlowDiagnostics
from .provider import apply_points
from .regional import RegionalComparison, compare_regions

REGIONS = ("left_eye", "right_eye", "nose", "mouth", "jaw", "central", "all")


@dataclass(frozen=True)
class Measurement:
    name: str
    value: float | None
    unit: str
    support: int
    total: int
    family: str
    interpretation: str
    interval: tuple[float, float] | None = None
    interval_kind: str | None = None
    exclusion: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and not np.isfinite(self.value):
            raise ValueError("measurement values must be finite or explicitly missing")
        if self.support < 0 or self.total < self.support:
            raise ValueError("invalid measurement support counts")
        if self.interval is not None and (len(self.interval) != 2 or not np.isfinite(self.interval).all()
                                          or self.interval[0] > self.interval[1]):
            raise ValueError("invalid measurement interval")
        if self.value is None and not self.exclusion:
            raise ValueError("missing measurement requires an exclusion reason")

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True)
class MeasurementProfile:
    measurements: tuple[Measurement, ...]
    residual_vectors: np.ndarray
    relational_distortion: np.ndarray
    heldout_residuals: np.ndarray
    heldout_labels: tuple[str, ...]
    sensitivity_draws: np.ndarray
    sensitivity_regions: tuple[str, ...]
    warnings: tuple[str, ...]
    heldout_fit_supported: np.ndarray
    heldout_failures: tuple[str | None, ...]
    sensitivity_draw_supported: np.ndarray
    sensitivity_failures: tuple[tuple[int, str], ...]
    regional: RegionalComparison

    def __post_init__(self) -> None:
        for name in ("residual_vectors", "relational_distortion", "heldout_residuals", "sensitivity_draws"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        for name in ("heldout_fit_supported", "sensitivity_draw_supported"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), bool))


def _residual(a: np.ndarray, b: np.ndarray, state: CanonicalState, excluded: str | None = None) -> np.ndarray:
    supported = state.observation_a.in_bounds & state.observation_b.in_bounds
    m, _, _ = fit_anatomy(a, b, state.anatomy, state.config, excluded, supported=supported)
    return (apply_points(a, m) - b) / symmetric_normalizer(a, b, state.anatomy, m)


def _sensitivity_experiment(state: CanonicalState) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, str], ...]]:
    """Whole-shape augmentation resampling + declared pixel noise, with refitting.

    Preserves cross-landmark augmentation dependence. Independent pixel noise is
    only the explicitly assumed residual floor. The Monte Carlo distribution is
    conditional on this observation/error model, not a population calibration.
    """
    rng = np.random.default_rng(state.config.seed)
    a, b = state.observation_a, state.observation_b
    centered_a = a.samples - a.samples.mean(axis=0)
    centered_b = b.samples - b.samples.mean(axis=0)
    valid = a.in_bounds & b.in_bounds
    out = np.zeros((state.config.uncertainty_draws, len(REGIONS)))
    supported = np.zeros(len(out), bool)
    failures = []
    for draw in range(len(out)):
        aa = a.xy + centered_a[rng.integers(len(centered_a))] + rng.normal(0, a.noise_floor_px, a.xy.shape)
        bb = b.xy + centered_b[rng.integers(len(centered_b))] + rng.normal(0, b.noise_floor_px, b.xy.shape)
        try:
            residual = np.linalg.norm(_residual(aa, bb, state), axis=1)
        except ObservationError as exc:
            failures.append((draw, str(exc)))
            continue
        supported[draw] = True
        for column, region in enumerate(REGIONS):
            ids = state.anatomy.region_indices(region)
            ids = ids[valid[ids]]
            # Missing regions are separately excluded in the ledger; zero is never published for them.
            out[draw, column] = np.sqrt(np.mean(residual[ids]**2)) if len(ids) else 0
    return out, supported, tuple(failures)


def sensitivity(state: CanonicalState) -> np.ndarray:
    """Retain the array-only API's complete-experiment requirement."""
    draws, supported, failures = _sensitivity_experiment(state)
    if not supported.all():
        raise ObservationError(f"sensitivity refit unavailable at draw {failures[0][0]}: {failures[0][1]}")
    return draws


def measure(state: CanonicalState, field: AnatomicalField, flow: FlowDiagnostics | None) -> MeasurementProfile:
    span = state.normalization_span_px
    residual = (state.xy_a - state.xy_b) / span
    magnitudes = np.linalg.norm(residual, axis=1)
    visible = state.observation_a.in_bounds & state.observation_b.in_bounds
    draw_values, draw_supported, draw_failures = _sensitivity_experiment(state)
    measurements = []
    warnings = list(state.warnings)
    if draw_failures:
        warnings.append(f"sensitivity_intervals_unavailable:{len(draw_failures)}_of_{len(draw_values)}_refits_failed")
    for column, region in enumerate(REGIONS):
        ids = state.anatomy.region_indices(region)
        selected = ids[visible[ids]]
        if len(selected) < 2:
            measurements.append(Measurement(f"geometry.{region}.rms", None, "registered_eye_span_geomean", len(selected), len(ids),
                                            "landmarks", "positional residual after proper similarity registration",
                                            exclusion="fewer_than_two_visible_landmarks"))
            continue
        value = float(np.sqrt(np.mean(magnitudes[selected]**2)))
        interval = (tuple(float(x) for x in np.quantile(draw_values[:, column], [.025, .975]))
                    if draw_supported.all() else None)
        measurements.append(Measurement(f"geometry.{region}.rms", value, "registered_eye_span_geomean", len(selected), len(ids),
            "landmarks", "lower residual means closer projected layout under this registration gauge",
            interval, "95_percent_conditional_observation_sensitivity_not_accuracy_CI" if interval else None))
        fraction = float(np.mean(magnitudes[selected] <= state.config.tolerance_ipd))
        measurements.append(Measurement(f"geometry.{region}.within_tolerance", fraction, "fraction_of_evaluated_landmarks",
            len(selected), len(ids), "landmarks",
            f"fraction within {state.config.tolerance_ipd:g} geometric-mean registered eye spans; not identity probability or rarity"))
        if region in ("left_eye", "right_eye", "nose", "mouth", "jaw"):
            # Paired model-contour samples retain their original ordering; no per-region warp fit.
            delta = residual[selected]
            centered = delta - delta.mean(axis=0)
            measurements.append(Measurement(f"contour.{region}.centered_rms", float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
                "registered_eye_span_geomean", len(selected), len(ids), "landmarks",
                "model-predicted contour mismatch after removing translation only; not independent edge evidence"))
    da = pairwise_distance_matrix(state.xy_a, span)
    db = pairwise_distance_matrix(state.xy_b, span)
    distortion = da - db
    mask = np.triu(np.ones(distortion.shape, bool), 1) & visible[:, None] & visible[None, :]
    measurements.append(Measurement("geometry.relational_rms", float(np.sqrt(np.mean(distortion[mask]**2))),
        "registered_eye_span_geomean", int(mask.sum()), int(np.triu(np.ones_like(mask), 1).sum()),
        "landmarks", "pairwise distances share landmarks and are not independent observations"))
    labels = ("eyes", "nose", "mouth", "jaw")
    heldout = np.zeros((len(labels), *residual.shape))
    heldout_supported = np.zeros(len(labels), bool)
    heldout_failures = []
    for i, label in enumerate(labels):
        try:
            heldout[i] = _residual(state.observation_a.xy, state.observation_b.xy, state, label)
            heldout_supported[i] = True
            heldout_failures.append(None)
        except ObservationError as exc:
            heldout_failures.append(str(exc))
            warnings.append(f"heldout_{label}_registration_unavailable:{exc}")
    for i, label in enumerate(labels):
        regions = ("left_eye", "right_eye") if label == "eyes" else (label,)
        ids = np.unique(np.concatenate([state.anatomy.region_indices(r) for r in regions]))
        total_ids = len(ids)
        ids = ids[visible[ids]]
        if not len(ids) or not heldout_supported[i]:
            for suffix in ("rms", "change"):
                measurements.append(Measurement(
                    f"registration.heldout_{label}.{suffix}", None,
                    "registered_eye_span_geomean", len(ids), total_ids, "registration_sensitivity",
                    "region excluded from alignment anchors", exclusion=("no_supported_region_landmarks" if not len(ids)
                    else "holdout_registration_unavailable: " + heldout_failures[i])))
            continue
        hold_value = float(np.sqrt(np.mean(np.sum(heldout[i, ids]**2, axis=1))))
        base_value = float(np.sqrt(np.mean(magnitudes[ids]**2)))
        measurements.append(Measurement(f"registration.heldout_{label}.rms", hold_value, "registered_eye_span_geomean",
            len(ids), total_ids, "registration_sensitivity",
            "evaluation region excluded from alignment anchors; exposes alignment-induced resemblance"))
        measurements.append(Measurement(f"registration.heldout_{label}.change", hold_value - base_value, "registered_eye_span_geomean",
            len(ids), total_ids, "registration_sensitivity", "signed change from the primary registration residual"))
    # Regions use an A-domain field and explicitly record its interpolation support.
    for region, region_mask in state.masks_a.items():
        observed_region = visible[state.anatomy.region_indices(region)].sum() >= 3
        selected = region_mask & field.valid & observed_region
        total, count = int(region_mask.sum()), int(selected.sum())
        if count < 8:
            measurements.append(Measurement(f"deformation.{region}.log_stretch_rms", None, "log_stretch",
                count, total, "interpolated_anatomy", "finite-rotation-invariant local stretch",
                exclusion="insufficient_observed_region_landmarks" if not observed_region else "fewer_than_eight_supported_pixels"))
            continue
        ids = field.simplex_index[selected]
        value = float(np.sqrt(np.mean(np.sum(field.principal_log_stretch[ids]**2, axis=1))))
        measurements.append(Measurement(f"deformation.{region}.log_stretch_rms", value, "log_stretch", count, total,
            "interpolated_anatomy", "piecewise-affine shape-model distortion; not physical strain or independent evidence"))
    measurements.append(Measurement("quality.folded_triangle_fraction", field.folded_fraction, "fraction", len(field.triangles),
                                    len(field.triangles), "quality", "folded triangles retained in denominator and excluded from interpolation"))
    if field.folded_fraction > 0:
        warnings.append("anatomical_field_has_folded_triangles")
    if flow is not None:
        for region, region_mask in state.masks_a.items():
            total = int(region_mask.sum())
            observed_region = visible[state.anatomy.region_indices(region)].sum() >= 3
            valid = region_mask & flow.admissible & field.valid & observed_region
            count = int(valid.sum())
            measurements.append(Measurement(f"flow.{region}.support_fraction", count / total if total and observed_region else None,
                "fraction_of_region_pixels", count, total, "image_correspondence", "texture/cycle-qualified overlap, not match probability",
                exclusion="empty_region" if not total else ("insufficient_observed_region_landmarks" if not observed_region else None)))
            measurements.append(Measurement(f"flow.{region}.anatomical_disagreement", float(np.median(flow.anatomical_disagreement_px[valid]) / span) if count else None,
                "registered_eye_span_geomean", count, total, "image_correspondence", "independent image-flow vs interpolated-anatomy disagreement; never fused into shape score",
                exclusion="no_texture_cycle_supported_pixels" if not count else None))
    regional = compare_regions(state, field)
    for row in regional.rows:
        movement, deformation = row.displacement, row.deformation
        for axis, index in (("x", 0), ("y", 1)):
            measurements.append(Measurement(f"regional.{row.region}.shift_{axis}",
                movement.shift_b_minus_a[index] if movement.shift_b_minus_a else None,
                "registered_eye_span_geomean", movement.supported, movement.total, "regional_displacement",
                "mean B-minus-A displacement in the existing canonical chart; no local alignment",
                exclusion=movement.unavailable_reason))
        for name, unit in (("rotation_rms_rad", "radians"), ("log_area_mean", "log_area_ratio"),
                           ("log_area_rms", "log_area_ratio"), ("anisotropic_log_strain_rms", "log_stretch"),
                           ("chart_shear_mean", "log_stretch"), ("chart_shear_rms", "log_stretch")):
            measurements.append(Measurement(f"regional.{row.region}.{name}", getattr(deformation, name), unit,
                deformation.supported_pixels, deformation.total_pixels, "regional_deformation",
                "component of the existing A-to-B inferred anatomical field; supported A-pixel weighting",
                exclusion=deformation.unavailable_reason))
    return MeasurementProfile(tuple(measurements), residual, distortion, heldout, labels,
        draw_values, REGIONS, tuple(warnings), heldout_supported, tuple(heldout_failures),
        draw_supported, draw_failures, regional)
