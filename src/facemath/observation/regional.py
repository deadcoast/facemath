"""Movement and finite-deformation components of the existing canonical pair.

There is no local registration or new correspondence here. Polar factors
describe the existing triangle maps; regional translations describe measured
landmark displacements. These are projected model geometry, not tissue strain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from .canonical import CanonicalState
from .contracts import ObservationError, frozen_array, object_digest
from .field import AnatomicalField

REGIONAL_NAMES = ("left_eye", "right_eye", "nose", "mouth", "jaw", "central", "all")


@dataclass(frozen=True)
class DisplacementComponents:
    total: int
    supported: int
    within_tolerance: int
    tolerance: float
    within_tolerance_fraction_bounds: tuple[float, float]
    shift_b_minus_a: tuple[float, float] | None
    translation_magnitude: float | None
    total_rms: float | None
    centered_rms: float | None
    maximum_displacement: float | None
    unavailable_reason: str | None


def displacement_components(a: np.ndarray, b: np.ndarray, supported: np.ndarray,
                            normalizer: float, tolerance: float) -> DisplacementComponents:
    """Exact mean/residual decomposition with a frozen landmark denominator."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.asarray(supported)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape or not len(a):
        raise ObservationError("regional displacement needs matching nonempty [N,2] coordinates")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ObservationError("regional coordinates must be finite")
    if mask.dtype != bool or mask.shape != (len(a),):
        raise ObservationError("regional support must be a boolean landmark mask")
    if any(isinstance(x, (bool, np.bool_)) or not np.isfinite(x) or x <= 0 for x in (normalizer, tolerance)):
        raise ObservationError("regional normalizer and tolerance must be finite and positive")
    delta = (b[mask] - a[mask]) / normalizer
    magnitudes = np.linalg.norm(delta, axis=1)
    total, count = len(a), len(delta)
    within = int(np.count_nonzero(magnitudes <= tolerance))
    bounds = (within / total, (within + total - count) / total)
    if count < 2:
        return DisplacementComponents(total, count, within, float(tolerance), bounds,
            None, None, None, None, None, "fewer_than_two_supported_landmarks")
    shift = delta.mean(axis=0)
    centered = delta - shift
    return DisplacementComponents(total, count, within, float(tolerance), bounds,
        tuple(float(x) for x in shift), float(np.linalg.norm(shift)),
        float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
        float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))), float(magnitudes.max()), None)


@dataclass(frozen=True)
class PolarComponents:
    supported: np.ndarray
    rotation: np.ndarray
    stretch: np.ndarray
    log_stretch: np.ndarray
    rotation_rad: np.ndarray
    log_area: np.ndarray
    deviatoric_log_stretch: np.ndarray

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, frozen_array(getattr(self, name), bool if name == "supported" else float))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"regional_triangle_" + name: getattr(self, name) for name in self.__dataclass_fields__}


def polar_components(jacobians: np.ndarray, supported: np.ndarray) -> PolarComponents:
    """Right polar decomposition J=R U and H=log(U) for existing valid maps.

    Invalid slots remain zero sentinels with a false support mask. Reflections
    are excluded, never repaired into rotations. The 1e-8 cutoffs match the
    anatomical field's nonsingular, positive-determinant contract.
    """
    maps, requested = np.asarray(jacobians, float), np.asarray(supported)
    if maps.ndim != 3 or maps.shape[1:] != (2, 2) or not len(maps) or not np.isfinite(maps).all():
        raise ObservationError("polar decomposition needs finite nonempty [N,2,2] maps")
    if requested.dtype != bool or requested.shape != (len(maps),):
        raise ObservationError("polar support must be a boolean triangle mask")
    rotations = np.zeros_like(maps); stretches = np.zeros_like(maps)
    logarithms = np.zeros_like(maps); deviators = np.zeros_like(maps)
    angles = np.zeros(len(maps)); log_area = np.zeros(len(maps))
    valid = np.zeros(len(maps), bool)
    ids = np.flatnonzero(requested & (np.linalg.det(maps) > 1e-8))
    if len(ids):
        left, singular, right_t = np.linalg.svd(maps[ids])
        keep = singular[:, -1] > 1e-8
        ids, left, singular, right_t = ids[keep], left[keep], singular[keep], right_t[keep]
        right = np.swapaxes(right_t, -1, -2)
        logs = np.log(singular)
        rotations[ids] = left @ right_t
        stretches[ids] = (right * singular[:, None, :]) @ right_t
        logarithms[ids] = (right * logs[:, None, :]) @ right_t
        angles[ids] = np.arctan2(rotations[ids, 1, 0], rotations[ids, 0, 0])
        log_area[ids] = logs.sum(axis=1)
        deviators[ids] = logarithms[ids] - .5 * log_area[ids, None, None] * np.eye(2)
        valid[ids] = True
    return PolarComponents(valid, rotations, stretches, logarithms, angles, log_area, deviators)


@dataclass(frozen=True)
class DeformationComponents:
    total_pixels: int
    supported_pixels: int
    support_fraction: float | None
    rotation_mean_rad: float | None
    rotation_rms_rad: float | None
    rotation_resultant_length: float | None
    log_area_mean: float | None
    log_area_rms: float | None
    area_ratio_mean: float | None
    anisotropic_log_strain_rms: float | None
    chart_shear_mean: float | None
    chart_shear_rms: float | None
    log_stretch_rms: float | None
    unavailable_reason: str | None
    rotation_mean_unavailable_reason: str | None = None


def deformation_components(polar: PolarComponents, triangle_ids: np.ndarray,
                           total_pixels: int, *, exclusion: str | None = None) -> DeformationComponents:
    """Aggregate retained A-domain pixels; repeated triangle IDs are pixel weights."""
    ids = np.asarray(triangle_ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= len(polar.supported))):
        raise ObservationError("deformation pixel lookup must contain valid triangle indices")
    if not isinstance(total_pixels, int) or isinstance(total_pixels, bool) or total_pixels < len(ids):
        raise ObservationError("deformation denominator must retain every supported pixel")
    if not np.all(polar.supported[ids]):
        raise ObservationError("unsupported triangle cannot enter regional deformation")
    count = len(ids)
    fraction = count / total_pixels if total_pixels else None
    if exclusion or count < 8:
        return DeformationComponents(total_pixels, count, fraction, *([None] * 10),
            exclusion or "fewer_than_eight_supported_pixels")
    theta, area = polar.rotation_rad[ids], polar.log_area[ids]
    sine, cosine = float(np.sin(theta).mean()), float(np.cos(theta).mean())
    resultant = float(np.hypot(sine, cosine))
    mean_angle = float(np.arctan2(sine, cosine)) if resultant > 1e-12 else None
    shear = polar.log_stretch[ids, 0, 1]
    def rms(values):
        return float(np.sqrt(np.mean(values**2)))
    return DeformationComponents(total_pixels, count, fraction, mean_angle, rms(theta), resultant,
        float(area.mean()), rms(area), float(np.exp(area).mean()),
        float(np.sqrt(np.mean(np.sum(polar.deviatoric_log_stretch[ids]**2, axis=(1, 2))))),
        float(shear.mean()), rms(shear),
        float(np.sqrt(np.mean(np.sum(polar.log_stretch[ids]**2, axis=(1, 2))))), None,
        "circular_mean_cancels" if mean_angle is None else None)


@dataclass(frozen=True)
class RegionalRow:
    region: str
    displacement: DisplacementComponents
    deformation: DeformationComponents


@dataclass(frozen=True)
class RegionalComparison:
    rows: tuple[RegionalRow, ...]
    displacement_b_minus_a: np.ndarray
    landmark_supported: np.ndarray
    membership: np.ndarray
    polar: PolarComponents
    recipe_sha256: str

    def __post_init__(self) -> None:
        for name in ("displacement_b_minus_a", "landmark_supported", "membership"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), float if name == "displacement_b_minus_a" else bool))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"regional_displacement_b_minus_a": self.displacement_b_minus_a,
                "regional_landmark_supported": self.landmark_supported,
                "regional_membership": self.membership, **self.polar.arrays()}

    def to_dict(self) -> dict:
        return {"format": "facemath.regional_comparison.v19", "recipe_sha256": self.recipe_sha256,
            "rows": [asdict(row) for row in self.rows],
            "displacement_units": "geometric_mean_registered_eye_span",
            "direction": "B minus A in the existing canonical chart; x viewer-right, y image-down",
            "deformation_direction": "existing A-to-B piecewise-affine map, uniformly weighted supported A-domain pixels",
            "shear_definition": "off-diagonal entry of log(U) in the canonical chart, not engineering shear or an axis-invariant scalar",
            "fraction_bounds_meaning": "missingness range over the declared landmark inventory, not a confidence interval",
            "meaning": "observed projected model geometry, not physical strain or independent anatomical validation",
            "localization_uncertainty_propagated": False,
            "missingness": "all seven regions retained; zero array sentinels require the associated support mask"}


def regional_pixel_support(state: CanonicalState, field: AnatomicalField,
                           polar: PolarComponents, region: str, visible: np.ndarray) -> np.ndarray:
    """The shared A-domain selector for aggregation and spatial inspection."""
    region_mask = state.masks_a["face" if region == "all" else region]
    observed_region = np.count_nonzero(visible[state.anatomy.region_indices(region)]) >= 3
    return (region_mask & field.valid & observed_region
            & polar.supported[np.maximum(field.simplex_index, 0)])


def compare_regions(state: CanonicalState, field: AnatomicalField) -> RegionalComparison:
    visible = state.observation_a.in_bounds & state.observation_b.in_bounds
    polar = polar_components(field.jacobians, field.valid_triangles)
    span = state.normalization_span_px
    displacement = np.zeros_like(state.xy_a)
    displacement[visible] = (state.xy_b[visible] - state.xy_a[visible]) / span
    membership = np.zeros((len(REGIONAL_NAMES), len(visible)), bool)
    rows = []
    for index, region in enumerate(REGIONAL_NAMES):
        ids = state.anatomy.region_indices(region)
        membership[index, ids] = True
        measured = displacement_components(state.xy_a[ids], state.xy_b[ids], visible[ids], span, state.config.tolerance_ipd)
        region_mask = state.masks_a["face" if region == "all" else region]
        observed_region = np.count_nonzero(visible[ids]) >= 3
        selected = regional_pixel_support(state, field, polar, region, visible)
        deformed = deformation_components(polar, field.simplex_index[selected], int(region_mask.sum()),
            exclusion="fewer_than_three_supported_region_landmarks" if not observed_region else None)
        rows.append(RegionalRow(region, measured, deformed))
    recipe = object_digest({"format": "facemath.regional_recipe.v19", "canonical_recipe_sha256": state.recipe_sha256,
        "regions": REGIONAL_NAMES, "tolerance": state.config.tolerance_ipd,
        "displacement": "B-A; regional mean and mean-centered residual; no refit",
        "deformation": "right_polar_log_stretch; existing_valid_A_domain_pixels; minimum_8",
        "denominators": "declared_landmarks_and_A_region_pixels"})
    return RegionalComparison(tuple(rows), displacement, visible, membership, polar, recipe)
