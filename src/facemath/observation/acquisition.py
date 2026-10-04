"""Measured acquisition contracts over declared source apertures, before inference.

This is a deterministic raster instrument, NOT a face detector, learned image-
quality model, hidden-domain-shift detector, or independent-reference observer.
Every measurement retains its source pixels and aperture. Group extrema use ALL
its enrolled members. Rules are simultaneous predicates, never priority rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import math
import os
import shutil
from .publication import publication_stage

import cv2
import numpy as np

from .contracts import ObservationError, array_digest, canonical_json, frozen_array, object_digest, rgb_image

FORMAT = "facemath.measured_acquisition_policy.v9"
RECIPE = "declared_aperture_RGB_integer299_587_114_div255000_AREA_Sobel_interior.v9"
GROUP_FORMAT = "facemath.measured_acquisition_group.v9"
FEATURE_NAMES = (
    "native_short_edge_px", "native_area_px2", "code_mean", "code_p05", "code_p95",
    "code_range_90", "low_code_fraction", "high_code_fraction", "detail_ratio",
)
FEATURE_UNITS = ("px", "px2", "normalized_code", "normalized_code", "normalized_code",
                 "normalized_code", "fraction", "fraction", "analysis_pixel_gradient_energy/variance")
AGGREGATES = tuple(f"{reducer}.{name}" for reducer in ("min", "max") for name in FEATURE_NAMES)
UNMATCHED, AMBIGUOUS = "__unmatched__", "__ambiguous__"
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ObservationError(f"{field} must be a nonempty trimmed string")
    return value


def _exact(obj: Mapping[str, Any], expected: set[str], field: str) -> None:
    if not isinstance(obj, Mapping) or set(obj) != expected:
        raise ObservationError(f"{field} has missing or unknown fields; expected {sorted(expected)}")


@dataclass(frozen=True)
class Predicate:
    feature: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        if self.feature not in AGGREGATES or self.op not in OPS:
            raise ObservationError("unknown measured feature or comparison operator")
        value = self.threshold
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ObservationError("predicate threshold must be an explicit finite number")
        object.__setattr__(self, "threshold", float(value))

    def evaluate(self, aggregate: np.ndarray) -> bool:
        return bool(OPS[self.op](aggregate.reshape(-1)[AGGREGATES.index(self.feature)], self.threshold))

    def to_dict(self) -> dict:
        return {"feature": self.feature, "op": self.op, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> Predicate:
        _exact(obj, {"feature", "op", "threshold"}, "measurement predicate")
        return cls(**dict(obj))


@dataclass(frozen=True)
class AcquisitionRule:
    category: str
    predicates: tuple[Predicate, ...]
    deployment_permitted: bool = True

    def __post_init__(self) -> None:
        _text(self.category, "measurement category")
        if self.category.startswith("__") or type(self.deployment_permitted) is not bool:
            raise ObservationError("reserved category or nonboolean deployment policy")
        predicates = tuple(self.predicates)
        if not predicates or any(not isinstance(p, Predicate) for p in predicates):
            raise ObservationError("every measured category needs at least one explicit predicate")
        if len({canonical_json(p.to_dict()) for p in predicates}) != len(predicates):
            raise ObservationError("duplicate measurement predicate")
        object.__setattr__(self, "predicates", predicates)

    def to_dict(self) -> dict:
        return {"category": self.category, "predicates": [p.to_dict() for p in self.predicates],
                "deployment_permitted": self.deployment_permitted}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> AcquisitionRule:
        _exact(obj, {"category", "predicates", "deployment_permitted"}, "acquisition rule")
        return cls(obj["category"], tuple(Predicate.from_dict(p) for p in obj["predicates"]), obj["deployment_permitted"])


@dataclass(frozen=True)
class AcquisitionPolicy:
    """Prespecified measurement and routing, never fitted to final outcomes.

    Pixel code values are not radiometrically linearized luminance. Detail is a
    sampling-specific gradient/variance diagnostic; noise can increase it. The
    aperture is declared, never derived from reference or predicted landmarks.
    """
    definition: str
    rules: tuple[AcquisitionRule, ...]
    maximum_group_members: int = 1
    analysis_max_side: int = 128
    low_code: float = 1. / 255.
    high_code: float = 254. / 255.
    variance_floor: float = 1.e-8

    def __post_init__(self) -> None:
        _text(self.definition, "source-acquisition definition")
        rules = tuple(self.rules)
        if not rules or any(not isinstance(r, AcquisitionRule) for r in rules):
            raise ObservationError("measured policy needs concrete rules")
        if len({r.category for r in rules}) != len(rules):
            raise ObservationError("duplicate category rule; use a new explicit policy")
        for name, value, lo, hi in (("maximum_group_members", self.maximum_group_members, 1, 1024),
                                    ("analysis_max_side", self.analysis_max_side, 16, 512)):
            if type(value) is not int or not lo <= value <= hi:
                raise ObservationError(f"{name} outside its supported integer range")
        for name in ("low_code", "high_code", "variance_floor"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ObservationError(f"{name} must be finite numeric")
            object.__setattr__(self, name, float(value))
        if not (0 <= self.low_code < self.high_code <= 1 and 0 < self.variance_floor <= 1):
            raise ObservationError("invalid code thresholds or variance floor")
        object.__setattr__(self, "rules", rules)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(r.category for r in self.rules) + (UNMATCHED, AMBIGUOUS)

    def permits(self, category: str) -> bool:
        return next((r.deployment_permitted for r in self.rules if r.category == category), False)

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def to_dict(self) -> dict:
        return {"format": FORMAT, "recipe": RECIPE, "definition": self.definition,
                "rules": [r.to_dict() for r in self.rules], "maximum_group_members": self.maximum_group_members,
                "analysis_max_side": self.analysis_max_side, "low_code": self.low_code,
                "high_code": self.high_code, "variance_floor": self.variance_floor,
                "group_reducers": ["min", "max"], "feature_order": list(FEATURE_NAMES),
                "routing": "one simultaneous match; zero or multiple matches withhold deployment",
                "aperture": "explicit integer xyxy half-open source rectangle; omission means full stored raster",
                "enrollment": "all declared group members, including prospectively excluded observation cases"}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> AcquisitionPolicy:
        if obj.get("format") != FORMAT or obj.get("recipe") != RECIPE:
            raise ObservationError("unsupported measured acquisition policy")
        out = cls(obj["definition"], tuple(AcquisitionRule.from_dict(r) for r in obj["rules"]),
                  obj["maximum_group_members"], obj["analysis_max_side"], obj["low_code"],
                  obj["high_code"], obj["variance_floor"])
        if canonical_json(out.to_dict()) != canonical_json(dict(obj)):
            raise ObservationError("measurement policy has inconsistent or unknown derived fields")
        return out


@dataclass(frozen=True)
class ApertureSource:
    image: np.ndarray = field(repr=False, compare=False)
    lineage_ids: tuple[str, ...]
    aperture: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        image = rgb_image(self.image)
        h, w = image.shape[:2]
        aperture = (0, 0, w, h) if self.aperture is None else tuple(self.aperture)
        if (len(aperture) != 4 or any(type(v) is not int for v in aperture)
                or not 0 <= aperture[0] < aperture[2] <= w
                or not 0 <= aperture[1] < aperture[3] <= h
                or min(aperture[2]-aperture[0], aperture[3]-aperture[1]) < 4):
            raise ObservationError("aperture must be in-raster integer xyxy, half-open, at least 4x4; no clipping")
        ids = tuple(_text(v, "source lineage") for v in self.lineage_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ObservationError("measured source needs distinct declared lineage IDs")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "aperture", aperture)
        object.__setattr__(self, "lineage_ids", tuple(sorted(ids)))

    @property
    def image_sha256(self) -> str:
        return array_digest(self.image)

    def binding(self) -> dict:
        return {"image_sha256": self.image_sha256, "image_shape": list(self.image.shape[:2]),
                "aperture": list(self.aperture), "lineage_ids": list(self.lineage_ids)}

    def patch(self) -> np.ndarray:
        x0, y0, x1, y1 = self.aperture
        return self.image[y0:y1, x0:x1]


def measure_source(source: ApertureSource, policy: AcquisitionPolicy) -> tuple[np.ndarray, np.ndarray]:
    """Return raw features and the exact analyzed gray patch. No landmark input.

    Percentiles and clipping use native aperture RGB code values. Spatial detail
    alone uses the area-reduced patch and excludes the Sobel boundary pixels.
    No upsampling, EXIF transform, sharpening, automatic crop or denoising occurs.
    """
    patch = source.patch()
    # Integer-weighted numerator is exact for uint8 RGB. Divide only once, so
    # neutral codes 1 and 254 retain their inclusive clipping boundaries.
    # This is a code-value diagnostic, not linear-light luminance.
    gray = (patch.astype(np.float64) @ np.array([299., 587., 114.])) / 255000.
    h, w = gray.shape
    if max(h, w) > policy.analysis_max_side:
        scale = policy.analysis_max_side/max(h, w)
        size = (max(1, int(round(w*scale))), max(1, int(round(h*scale))))
        # An extreme aperture would require upsampling its short dimension.
        if min(size) < 3 or size[0] > w or size[1] > h:
            raise ObservationError("aperture aspect ratio cannot support the analysis grid without upsampling")
        analyzed = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    else:
        analyzed = gray.copy()
    gx = cv2.Sobel(analyzed, cv2.CV_64F, 1, 0, ksize=3, scale=1/8., borderType=cv2.BORDER_REFLECT_101)[1:-1, 1:-1]
    gy = cv2.Sobel(analyzed, cv2.CV_64F, 0, 1, ksize=3, scale=1/8., borderType=cv2.BORDER_REFLECT_101)[1:-1, 1:-1]
    energy = float(np.mean(gx*gx+gy*gy))
    detail = energy/(float(np.var(analyzed[1:-1, 1:-1]))+policy.variance_floor)
    lo, hi = np.quantile(gray, [.05, .95], method="linear")
    features = np.array([min(h, w), h*w, gray.mean(), lo, hi, hi-lo,
                         np.mean(gray <= policy.low_code), np.mean(gray >= policy.high_code), detail])
    return frozen_array(features), frozen_array(analyzed)


@dataclass(frozen=True)
class MeasuredGroup:
    policy: AcquisitionPolicy
    group_id: str
    sources: tuple[ApertureSource, ...] = field(repr=False, compare=False)
    features: np.ndarray = field(init=False, repr=False, compare=False)
    aggregate: np.ndarray = field(init=False, repr=False, compare=False)
    predicate_truth: np.ndarray = field(init=False, repr=False, compare=False)
    predicate_offsets: np.ndarray = field(init=False, repr=False, compare=False)
    rule_matches: np.ndarray = field(init=False, repr=False, compare=False)
    analyzed: tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)
    category: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.group_id, "measured acquisition group")
        if not isinstance(self.policy, AcquisitionPolicy) or any(not isinstance(s, ApertureSource) for s in self.sources):
            raise ObservationError("measured group requires a concrete policy and source-pixel members")
        sources = tuple(sorted(self.sources, key=lambda s: s.image_sha256))
        if not 1 <= len(sources) <= self.policy.maximum_group_members:
            raise ObservationError("measured group size is outside the frozen enrollment contract")
        if len({s.image_sha256 for s in sources}) != len(sources):
            raise ObservationError("duplicate decoded image in measured group")
        measured = [measure_source(s, self.policy) for s in sources]
        features = np.stack([m[0] for m in measured])
        aggregate = np.stack([features.min(0), features.max(0)])
        truth, offsets, matches = [], [0], []
        for rule in self.policy.rules:
            values = [p.evaluate(aggregate) for p in rule.predicates]
            truth.extend(values); offsets.append(len(truth)); matches.append(all(values))
        selected = [r.category for r, match in zip(self.policy.rules, matches) if match]
        category = selected[0] if len(selected) == 1 else AMBIGUOUS if selected else UNMATCHED
        for name, value in (("sources", sources), ("features", frozen_array(features)),
                            ("aggregate", frozen_array(aggregate)), ("predicate_truth", frozen_array(truth, bool)),
                            ("predicate_offsets", frozen_array(offsets, np.int64)), ("rule_matches", frozen_array(matches, bool)),
                            ("analyzed", tuple(m[1] for m in measured)), ("category", category)):
            object.__setattr__(self, name, value)

    @property
    def lineage_ids(self) -> tuple[str, ...]:
        return tuple(sorted({i for s in self.sources for i in s.lineage_ids}))

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def require_source(self, image: np.ndarray) -> None:
        digest = array_digest(rgb_image(image))
        if digest not in {s.image_sha256 for s in self.sources}:
            raise ObservationError("query image is not a member of its measured acquisition group")

    def recompute(self) -> MeasuredGroup:
        out = MeasuredGroup(self.policy, self.group_id, self.sources)
        if out.digest != self.digest:
            raise ObservationError("source measurements changed on recomputation")
        return out

    def to_dict(self) -> dict:
        return {"format": GROUP_FORMAT, "policy_sha256": self.policy.digest, "group_id": self.group_id,
                "lineage_ids": list(self.lineage_ids), "members": [s.binding() for s in self.sources],
                "feature_order": list(FEATURE_NAMES), "feature_units": list(FEATURE_UNITS),
                "features": self.features.tolist(), "aggregate_order": list(AGGREGATES),
                "aggregate": self.aggregate.tolist(), "predicate_offsets": self.predicate_offsets.tolist(),
                "predicate_truth": self.predicate_truth.tolist(), "rule_matches": self.rule_matches.tolist(),
                "category": self.category, "deployment_permitted_by_rule": self.policy.permits(self.category),
                "analyzed_patches": [{"shape": list(a.shape), "sha256": array_digest(a)} for a in self.analyzed],
                "source": "recomputed original pixels in declared apertures; no predictions, references or pair metrics",
                "limit": "not face detection, physiological attributes, blind sharpness accuracy or hidden-shift certification"}

    def arrays(self, prefix: str = "acquisition_") -> dict[str, np.ndarray]:
        result = {prefix+"features": self.features, prefix+"aggregate": self.aggregate,
                  prefix+"predicate_truth": self.predicate_truth, prefix+"predicate_offsets": self.predicate_offsets,
                  prefix+"rule_matches": self.rule_matches,
                  prefix+"apertures": frozen_array([s.aperture for s in self.sources], np.int64)}
        result.update({f"{prefix}analyzed_{i:04d}": a for i, a in enumerate(self.analyzed)})
        return result


def measure_frozen_groups(policy: AcquisitionPolicy, cases: tuple[Mapping[str, Any], ...], image_loader) -> tuple[MeasuredGroup, ...]:
    """Image-loader sees an entry; the measurement instrument gets only RGB/ROI/IDs."""
    grouped: dict[str, list[ApertureSource]] = {}
    for c in cases:
        source = ApertureSource(image_loader(c), tuple(c["lineage_ids"]),
                                tuple(c["acquisition_aperture"]) if c.get("acquisition_aperture") is not None else None)
        grouped.setdefault(c["group_id"], []).append(source)
    return tuple(MeasuredGroup(policy, group, tuple(sources)) for group, sources in sorted(grouped.items()))


def write_group(group: MeasuredGroup, destination: str | Path) -> Path:
    """Create-only source bundle. Raw PNGs make numeric remeasurement possible."""
    from .study import save_rgb, write_json_new, file_inventory
    from .journal import environment
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = publication_stage(destination.parent, ".facemath-acquisition-")
    try:
        group.recompute()
        for i, s in enumerate(group.sources):
            save_rgb(temp/f"source_{i:04d}.png", s.image)
        write_json_new(temp/"policy.json", group.policy.to_dict())
        write_json_new(temp/"measurements.json", group.to_dict())
        write_json_new(temp/"environment.json", environment())
        np.savez_compressed(temp/"arrays.npz", **group.arrays())
        from .acquisition_report import group_html
        (temp/"report.html").write_text(group_html(group), encoding="utf-8")
        manifest = {"format": "facemath.acquisition_group_bundle.v9", "group_sha256": group.digest,
                    "files": file_inventory(temp)}
        manifest["content_sha256"] = object_digest(manifest)
        write_json_new(temp/"manifest.json", manifest)
        load_group(temp)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temp, destination)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination


def load_group(path: str | Path) -> MeasuredGroup:
    """Verify bytes AND remeasure saved rasters, not just trust stored features."""
    from .study import read_json, verify_inventory, load_rgb, local_file
    root = Path(path).resolve()
    manifest = read_json(root/"manifest.json")
    digest = manifest.pop("content_sha256")
    if manifest.get("format") != "facemath.acquisition_group_bundle.v9" or object_digest(manifest) != digest:
        raise ObservationError("invalid acquisition-group manifest")
    verify_inventory(root, manifest["files"], exclude=("manifest.json",))
    policy = AcquisitionPolicy.from_dict(read_json(root/"policy.json"))
    value = read_json(root/"measurements.json")
    if not isinstance(value.get("members"), list) or not 1 <= len(value["members"]) <= policy.maximum_group_members:
        raise ObservationError("member inventory outside frozen policy")
    sources = tuple(ApertureSource(load_rgb(local_file(root, f"source_{i:04d}.png")), tuple(s["lineage_ids"]), tuple(s["aperture"]))
                    for i, s in enumerate(value["members"]))
    result = MeasuredGroup(policy, value["group_id"], sources)
    if canonical_json(result.to_dict()) != canonical_json(value) or result.digest != manifest["group_sha256"]:
        raise ObservationError("recorded acquisition measurements differ from source-pixel recomputation")
    expected = result.arrays()
    with np.load(root/"arrays.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected) or any(not np.array_equal(arrays[k], a) for k, a in expected.items()):
            raise ObservationError("acquisition arrays differ from source-pixel recomputation")
    return result


def group_from_spec(path: str | Path, policy: AcquisitionPolicy) -> MeasuredGroup:
    from .study import read_json, load_rgb, local_file
    p = Path(path).resolve(); spec = read_json(p)
    _exact(spec, {"format", "group_id", "members"}, "source group specification")
    if spec["format"] != "facemath.acquisition_group_spec.v9" or not isinstance(spec["members"], list):
        raise ObservationError("expected acquisition_group_spec.v9 and member list")
    if not 1 <= len(spec["members"]) <= policy.maximum_group_members:
        raise ObservationError("member inventory outside frozen policy")
    sources = []
    for member in spec["members"]:
        if set(member) not in ({"image", "lineage_ids"}, {"image", "lineage_ids", "aperture"}):
            raise ObservationError("member needs only image, lineages and optional aperture")
        sources.append(ApertureSource(load_rgb(local_file(p.parent, member["image"])), tuple(member["lineage_ids"]),
                                      tuple(member["aperture"]) if member.get("aperture") is not None else None))
    return MeasuredGroup(policy, spec["group_id"], tuple(sources))


def policy_from_document(value: Mapping[str, Any]) -> AcquisitionPolicy:
    if value.get("format") == FORMAT:
        return AcquisitionPolicy.from_dict(value)
    from .bias import load_localization_calibration
    from .stratified import BoundStratifiedCalibration
    model = load_localization_calibration(value)
    if isinstance(model, BoundStratifiedCalibration):
        model = model.model
    policy = getattr(getattr(model, "policy", None), "measurement_policy", None)
    if policy is None:
        raise ObservationError("calibration does not contain a measured acquisition policy")
    return policy


def write_group_files(group: MeasuredGroup, root: Path, prefix: str) -> None:
    """Flat, prefixed sidecars governed by the containing pair manifest."""
    from .study import save_rgb, write_json_new
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ObservationError("unsafe acquisition sidecar prefix")
    for i, source in enumerate(group.sources):
        save_rgb(root/f"{prefix}source_{i:04d}.png", source.image)
    write_json_new(root/(prefix+"policy.json"), group.policy.to_dict())
    write_json_new(root/(prefix+"measurements.json"), group.to_dict())
    np.savez_compressed(root/(prefix+"arrays.npz"), **group.arrays())
    from .acquisition_report import group_html
    with (root/(prefix+"report.html")).open("x", encoding="utf-8") as stream:
        stream.write(group_html(group, artifact_prefix=prefix))


def load_group_files(root: Path, prefix: str) -> MeasuredGroup:
    from .study import read_json, load_rgb, local_file
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ObservationError("unsafe acquisition sidecar prefix")
    policy = AcquisitionPolicy.from_dict(read_json(local_file(root, prefix+"policy.json")))
    value = read_json(local_file(root, prefix+"measurements.json"))
    if not 1 <= len(value["members"]) <= policy.maximum_group_members:
        raise ObservationError("sidecar member inventory outside frozen policy")
    sources = tuple(ApertureSource(load_rgb(local_file(root, f"{prefix}source_{i:04d}.png")),
                    tuple(member["lineage_ids"]), tuple(member["aperture"])) for i, member in enumerate(value["members"]))
    result = MeasuredGroup(policy, value["group_id"], sources)
    if canonical_json(result.to_dict()) != canonical_json(value):
        raise ObservationError("pair acquisition evidence differs from original-pixel recomputation")
    expected = result.arrays()
    with np.load(local_file(root,prefix+"arrays.npz"), allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected) or any(not np.array_equal(arrays[k], a) for k,a in expected.items()):
            raise ObservationError("pair acquisition arrays differ from pixel recomputation")
    return result
