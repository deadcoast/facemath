"""Read-only spatial inspection of the one existing regional comparison.

Colors are display encodings, not acceptance rules. Numerical values never
clip to the display range. The same pixel selector feeds regional aggregation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from .regional import regional_pixel_support

if TYPE_CHECKING:
    from .instrument import InstrumentResult


MAP_SCALES = (
    {"key": "rotation", "label": "Local rotation", "low": -30., "high": 30.,
     "ticks": ["−30°", "0°", "+30°"], "units": "degrees, positive clockwise in image coordinates"},
    {"key": "area", "label": "Local area ratio B/A", "low": -1., "high": 1.,
     "ticks": ["0.5×", "1×", "2×"], "units": "color coordinate log2(area ratio)"},
    {"key": "anisotropy", "label": "Anisotropic logarithmic stretch", "low": 0., "high": float(np.log(2)),
     "ticks": ["0", "0.347", "0.693"], "units": "Frobenius norm of deviatoric log(U)"},
)
BLUE = np.array([82, 152, 255.])
NEUTRAL = np.array([216, 226, 235.])
ORANGE = np.array([255, 156, 66.])


def _triangle_values(result: InstrumentResult) -> dict[str, np.ndarray]:
    polar = result.profile.regional.polar
    return {"rotation": np.rad2deg(polar.rotation_rad),
            "area": polar.log_area / np.log(2),
            "anisotropy": np.linalg.norm(polar.deviatoric_log_stretch, axis=(1, 2))}


def regional_inspection(result: InstrumentResult) -> dict:
    """Export exact locations and weights; no new fit or observed-state arrays."""
    state, field, regional = result.state, result.field, result.profile.regional
    values = _triangle_values(result)
    rows = []
    for i, row in enumerate(regional.rows):
        selected = regional_pixel_support(state, field, regional.polar, row.region, regional.landmark_supported)
        ids, weights = np.unique(field.simplex_index[selected], return_counts=True)
        landmarks = []
        for index in np.flatnonzero(regional.membership[i]):
            supported = bool(regional.landmark_supported[index])
            delta = regional.displacement_b_minus_a[index] if supported else None
            residual = (delta - row.displacement.shift_b_minus_a
                        if supported and row.displacement.shift_b_minus_a is not None else None)
            landmarks.append({"index": int(index), "name": state.observation_a.names[index],
                "supported": supported, "a": state.xy_a[index].tolist() if supported else None,
                "b": state.xy_b[index].tolist() if supported else None,
                "delta": delta.tolist() if supported else None,
                "centered_delta": residual.tolist() if residual is not None else None,
                "magnitude": float(np.linalg.norm(delta)) if supported else None,
                "within_tolerance": bool(np.linalg.norm(delta) <= row.displacement.tolerance) if supported else None})
        maps = []
        for scale in MAP_SCALES:
            v = values[scale["key"]][ids]
            maps.append({**scale, "image": f'regional_{row.region}_{scale["key"]}.png',
                "below_scale_pixels": int(weights[v < scale["low"]].sum()),
                "above_scale_pixels": int(weights[v > scale["high"]].sum())})
        triangles = [{"index": int(index), "vertices": field.triangles[index].tolist(),
            "pixels": int(weight), "rotation_deg": float(values["rotation"][index]),
            "area_ratio": float(np.exp(regional.polar.log_area[index])),
            "anisotropy": float(values["anisotropy"][index]),
            "chart_shear": float(regional.polar.log_stretch[index, 0, 1])}
            for index, weight in zip(ids, weights)]
        rows.append({"region": row.region, "landmarks": landmarks, "triangles": triangles,
            "maps": maps, "maps_available": row.deformation.unavailable_reason is None,
            "unavailable_reason": row.deformation.unavailable_reason,
            "unresolved_pixels": row.deformation.total_pixels - row.deformation.supported_pixels})
    return {"format": "facemath.regional_inspection.v24", "canvas_size": state.config.canvas_size,
        "normalization_span_px": state.normalization_span_px, "rows": rows,
        "coordinates": "existing canonical stored-raster pixel centers; x viewer-right, y image-down",
        "display_limits_are_tolerances": False,
        "arrays": ["canonical_a_xy", "canonical_b_xy", "regional_membership", "regional_landmark_supported",
                   "simplex_index", "anatomical_support", "regional_triangle_supported", "regional_triangle_log_stretch"],
        "meaning": "observed projected model geometry; no localization uncertainty propagated"}


def regional_map_images(result: InstrumentResult):
    """Yield RGB rasters with exact support and fixed, labeled color scales."""
    state, field, regional = result.state, result.field, result.profile.regional
    values = _triangle_values(result)
    yy, xx = np.indices(field.valid.shape)
    hatch = ((xx + yy) // 3) % 2 == 0
    for row in regional.rows:
        if row.deformation.unavailable_reason is not None:
            continue
        region = state.masks_a["face" if row.region == "all" else row.region]
        selected = regional_pixel_support(state, field, regional.polar, row.region, regional.landmark_supported)
        ids = field.simplex_index[selected]
        for scale in MAP_SCALES:
            position = np.clip((values[scale["key"]][ids] - scale["low"]) / (scale["high"] - scale["low"]), 0, 1)
            if scale["key"] == "anisotropy":
                colors = NEUTRAL + position[:, None] * (ORANGE - NEUTRAL)
            else:
                colors = np.where((position <= .5)[:, None],
                    BLUE + (2 * position)[:, None] * (NEUTRAL - BLUE),
                    NEUTRAL + (2 * position - 1)[:, None] * (ORANGE - NEUTRAL))
            image = state.image_a.astype(float) * .35
            missing = region & ~selected
            image[missing] = [65, 70, 80]
            image[missing & hatch] = [195, 168, 125]
            image[selected] = .25 * state.image_a[selected] + .75 * colors
            yield f'regional_{row.region}_{scale["key"]}.png', np.rint(image).astype(np.uint8)


REGIONAL_CSS = """
#regional table{min-width:920px;font-variant-numeric:tabular-nums}
#regional caption{text-align:left;padding:12px 14px;color:var(--muted)}
#regional summary{min-height:44px;padding:10px 0}
#regional .regional-gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;max-width:960px}
#regional svg,#regional img{display:block;width:100%}
#regional .regional-values{max-height:380px;overflow:auto}
#regional .regional-values th{position:sticky;top:0}
#regional .regional-legend{height:12px;margin-top:12px;background:linear-gradient(90deg,#5298ff,#d8e2eb,#ff9c42)}
#regional .regional-legend[data-scale=anisotropy]{background:linear-gradient(90deg,#d8e2eb,#ff9c42)}
#regional .regional-ticks{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}
@media(max-width:760px){#regional .regional-gallery{grid-template-columns:1fr}}
"""
