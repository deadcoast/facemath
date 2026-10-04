"""Prospective acquisition contracts and self-contained, create-only study locks.

A lock freezes images, references, grouping, slices, eligibility and estimator
choice BEFORE observer execution. A SHA-256 proves consistency, not chronology
or human blinding. Neither this module nor the runner infers subject identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import os
import shutil
import tempfile

import cv2
import numpy as np

from .contracts import Observation, ObservationError, array_digest, canonical_json, object_digest
from .journal import code_fingerprint, numeric_environment
from .reference import REFERENCE_PROTOCOL_SHA256, ReferenceAnnotation, consensus, check_digest, observer_recipe
from .validation import ValidationCase, validate_study

STUDY_FORMAT = "facemath.prospective_study.v5"
REGISTRY_STUDY_FORMAT = "facemath.prospective_study.v16"
MANIFEST_FORMATS = {STUDY_FORMAT: "facemath.study_lock_manifest.v5",
                    REGISTRY_STUDY_FORMAT: "facemath.study_lock_manifest.v16"}


def read_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ObservationError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_json_new(path: str | Path, value: Any) -> None:
    """Exclusive creation; never clobber existing evidence."""
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def local_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or "\\" in name:
        raise ObservationError("study paths must be explicit relative POSIX paths")
    p = root / name
    if any(part in ("..", ".") for part in name.split("/")):
        raise ObservationError("study paths cannot traverse directories")
    if p.is_symlink() or any(parent.is_symlink() for parent in p.parents if parent != root.parent):
        raise ObservationError("symlinked study input is not a frozen source")
    resolved = p.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ObservationError(f"missing or out-of-root study input: {name}")
    return resolved


def load_rgb(path: str | Path) -> np.ndarray:
    # imdecode handles Unicode paths on Windows and ignores EXIF orientation
    # explicitly, preserving the annotation's stored-raster coordinate contract.
    data = np.frombuffer(Path(path).read_bytes(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or min(image.shape[:2]) < 8:
        raise ObservationError(f"not a supported image: {Path(path).name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ObservationError("PNG encoding failed")
    with path.open("xb") as f:
        f.write(data.tobytes())


def file_inventory(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ObservationError("symlinks are not evidence artifacts")
        if p.is_file():
            name = p.relative_to(root).as_posix()
            if name not in exclude:
                data = p.read_bytes()
                out[name] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
    return out


def verify_inventory(root: Path, expected: Mapping[str, Any], *, exclude: tuple[str, ...] = ()) -> None:
    actual = file_inventory(root, exclude=exclude)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise ObservationError(f"artifact inventory mismatch: missing={missing}, extra={extra}, changed={changed}")


@dataclass(frozen=True)
class ObserverDeclaration:
    provider: str
    provider_fingerprint: str
    schema: str
    noise_floor_px: float
    attempted_samples: int
    execution_kind: str
    model_sha256: str | None = None
    runtime_version: str | None = None

    def __post_init__(self) -> None:
        check_digest(self.provider_fingerprint, "observer fingerprint")
        if self.execution_kind not in ("native_mediapipe", "synthetic_fixture"):
            raise ObservationError("declare native_mediapipe or synthetic_fixture execution")
        if not self.provider or not self.schema or (not np.isfinite(self.noise_floor_px) or self.noise_floor_px <= 0):
            raise ObservationError("invalid observer declaration")
        if type(self.attempted_samples) is not int or self.attempted_samples < 1:
            raise ObservationError("attempted_samples must be a positive integer")
        if self.execution_kind == "native_mediapipe":
            check_digest(self.model_sha256, "model SHA-256")
            if (self.provider != "mediapipe" or self.schema != "mediapipe478.v1"
                    or self.attempted_samples not in (1, 7) or not self.runtime_version):
                raise ObservationError("incompatible native MediaPipe declaration")
            from .provider import mediapipe_fingerprint
            if mediapipe_fingerprint(self.model_sha256, self.runtime_version) != self.provider_fingerprint:
                raise ObservationError("native observer fingerprint does not match model/runtime")
        elif "synthetic" not in self.provider:
            raise ObservationError("fixture observer must name its synthetic provenance")

    @property
    def recipe_sha256(self) -> str:
        return object_digest({"provider": self.provider, "provider_fingerprint": self.provider_fingerprint,
            "schema": self.schema, "noise_floor_px": self.noise_floor_px,
            "attempted_samples": self.attempted_samples, "reference_protocol_sha256": REFERENCE_PROTOCOL_SHA256,
            "score": "max_reference8_mahalanobis_to_independent_consensus.v1"})

    def require_observation(self, observation: Observation, image: np.ndarray) -> None:
        observation.require_image(image)
        if observer_recipe(observation) != self.recipe_sha256:
            raise ObservationError("observed output violates the frozen observer recipe")
        if self.execution_kind == "native_mediapipe":
            m = json.loads(observation.metadata_json)
            if m.get("model_sha256") != self.model_sha256 or m.get("runtime_version") != self.runtime_version:
                raise ObservationError("observation model/runtime differs from study lock")

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> ObserverDeclaration:
        return cls(**dict(obj))

    @classmethod
    def mediapipe(cls, model_sha256: str, runtime_version: str, *, noise_floor_px: float = .75,
                  augment: bool = True) -> ObserverDeclaration:
        from .provider import mediapipe_fingerprint
        return cls("mediapipe", mediapipe_fingerprint(model_sha256, runtime_version), "mediapipe478.v1",
                   noise_floor_px, 7 if augment else 1, "native_mediapipe", model_sha256, runtime_version)


@dataclass(frozen=True)
class FrozenStudy:
    root: Path
    lock_json: str

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self.lock_json)

    @property
    def digest(self) -> str:
        return object_digest(self.data)

    @property
    def observer(self) -> ObserverDeclaration:
        return ObserverDeclaration.from_dict(self.data["observer"])

    @property
    def cases(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data["cases"])

    @property
    def included(self) -> tuple[dict[str, Any], ...]:
        return tuple(c for c in self.cases if c["eligibility"] == "included")

    def image(self, case: Mapping[str, Any]) -> np.ndarray:
        image = load_rgb(local_file(self.root, case["image"]))
        if array_digest(image) != case["image_sha256"]:
            raise ObservationError("frozen source pixel mismatch")
        return image

    def references(self, case: Mapping[str, Any]) -> tuple[ReferenceAnnotation, ...]:
        return tuple(ReferenceAnnotation.from_dict(read_json(local_file(self.root, n))) for n in case["references"])


def load_study(path: str | Path) -> FrozenStudy:
    root = Path(path).resolve()
    manifest = read_json(root / "manifest.json")
    expected = manifest.pop("content_sha256")
    if manifest.get("format") not in MANIFEST_FORMATS.values() or object_digest(manifest) != expected:
        raise ObservationError("invalid study-lock manifest")
    verify_inventory(root, manifest["files"], exclude=("manifest.json",))
    data = read_json(root / "study.lock.json")
    if (data.get("format") not in MANIFEST_FORMATS
            or MANIFEST_FORMATS[data["format"]] != manifest["format"]
            or object_digest(data) != manifest["study_sha256"]):
        raise ObservationError("study lock mismatch")
    if data["format"] == REGISTRY_STUDY_FORMAT:
        from .enrollment import require_disjoint
        binding = data.get("enrollment_registry", {})
        if set(binding) != {"path", "content_sha256", "check"} or binding["path"] != "enrollment_registry.json":
            raise ObservationError("registry study lost its frozen registry binding")
        registry = read_json(local_file(root, binding["path"]))
        if registry.get("content_sha256") != binding["content_sha256"]:
            raise ObservationError("frozen registry digest mismatch")
        check = require_disjoint(registry, [(c, tuple(ReferenceAnnotation.from_dict(
            read_json(local_file(root, n))) for n in c["references"])) for c in data["cases"]])
        if check != binding["check"]:
            raise ObservationError("frozen registry enrollment check differs from recomputation")
    elif "enrollment_registry" in data or "enrollment_registry.json" in manifest["files"]:
        raise ObservationError("registry requires the v16 study format")
    ObserverDeclaration.from_dict(data["observer"])
    if data.get("bias_mode") == "development_joint_ocular_stratified":
        from .stratified import RoutingPolicy, validate_routes
        policy = RoutingPolicy.from_dict(data["stratified_error"])
        if policy.strata != tuple(data["strata"]):
            raise ObservationError("loaded routing policy differs from frozen acquisition inventory")
        validate_routes(tuple(data["cases"]), policy)
        if policy.measurement_policy is not None:
            from .acquisition import measure_frozen_groups
            groups = measure_frozen_groups(policy.measurement_policy, tuple(data["cases"]),
                lambda c: load_rgb(local_file(root, c["image"])))
            measured = {"policy_sha256": policy.measurement_policy.digest, "groups": [g.to_dict() for g in groups]}
            if canonical_json(measured) != canonical_json(data.get("measured_acquisition")):
                raise ObservationError("frozen acquisition measurements differ from original-pixel recomputation")
            by_group = {g.group_id: g for g in groups}
            for case in data["cases"]:
                group = by_group[case["group_id"]]
                if case["stratum"] != group.category or case.get("acquisition_group_sha256") != group.digest:
                    raise ObservationError("frozen category differs from measured source group")
        elif "measured_acquisition" in data:
            raise ObservationError("unmeasured study cannot claim measured route evidence")
    elif "stratified_error" in data:
        raise ObservationError("loaded routing policy does not match frozen estimator mode")
    if "admission_policy" in data:
        from .admission import AdmissionPolicy, plan_admission
        policy = AdmissionPolicy.from_dict(data["admission_policy"])
        if data["observer"]["execution_kind"] == "native_mediapipe":
            for case in data["cases"]:
                if case["eligibility"] == "included":
                    refs = [ReferenceAnnotation.from_dict(read_json(local_file(root, n))) for n in case["references"]]
                    if any(r.kind not in ("human_blind", "external_dataset") or r.prediction_exposure != "none_declared" for r in refs):
                        raise ObservationError("native empirical admission requires independent prediction-unexposed labels")
        planned = plan_admission(policy, data["cases"], data["strata"], data["observer"]["execution_kind"])
        if planned != data.get("admission_plan"):
            raise ObservationError("frozen empirical admission plan differs from enrolled evidence")
    from .workspace import verify_workspace_origin
    verify_workspace_origin(data, root)
    if "frozen_model" in data:
        from .transfer import require_transfer
        require_transfer(data["frozen_model"], data, [(c, tuple(
            ReferenceAnnotation.from_dict(read_json(local_file(root, n))) for n in c["references"]))
            for c in data["cases"]])
    return FrozenStudy(root, canonical_json(data))


def freeze_study(spec_path: str | Path, destination: str | Path, *, frozen_model: dict | None = None,
                 enrollment_registry: dict | None = None) -> FrozenStudy:
    """No detector is constructed by this function. Every source is preflighted."""
    spec_path, target = Path(spec_path).resolve(), Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    spec = read_json(spec_path)
    if "frozen_model" in spec:
        raise ObservationError("supply the frozen model capsule explicitly at freeze time")
    if "enrollment_registry" in spec:
        raise ObservationError("supply the enrollment registry explicitly at freeze time")
    if spec.get("format") != STUDY_FORMAT:
        raise ObservationError("expected facemath.prospective_study.v5")
    if not isinstance(spec.get("domain"), str) or not spec["domain"].strip():
        raise ObservationError("declare an acquisition domain")
    declaration = spec["observer"]
    if isinstance(declaration, str):
        declaration = read_json(local_file(spec_path.parent, declaration))
    observer = ObserverDeclaration.from_dict(declaration)
    minimum = spec.get("minimum_raters", 1)
    max_group = spec.get("maximum_cases_per_group", 1)
    min_dev = spec.get("minimum_development_groups", 5)
    for name, value, lo in (("minimum_raters", minimum, 1), ("maximum_cases_per_group", max_group, 1),
                            ("minimum_development_groups", min_dev, 2)):
        if type(value) is not int or value < lo:
            raise ObservationError(f"invalid {name}")
    timeout = spec.get("timeout_seconds", 90.)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout <= 0:
        raise ObservationError("timeout_seconds must be positive and finite")
    alpha = spec.get("alpha", .1)
    from .error_model import corrected_quantile
    corrected_quantile(np.zeros(1), alpha)
    mode = spec.get("bias_mode", "none")
    if mode not in ("none", "development_ocular", "development_joint_ocular", "development_joint_ocular_stratified"):
        raise ObservationError("bias_mode must be frozen as none, development_ocular or development_joint_ocular")
    joint_config = None
    if mode in ("development_joint_ocular", "development_joint_ocular_stratified"):
        from .joint import regularized_scatter
        joint_config = spec.get("joint_error", {"shrinkage": .02, "floor_eye_spans": .001})
        if not isinstance(joint_config, dict) or set(joint_config) != {"shrinkage", "floor_eye_spans"}:
            raise ObservationError("joint_error requires exactly shrinkage and floor_eye_spans")
        regularized_scatter(np.zeros((1,16,16)), joint_config["shrinkage"], joint_config["floor_eye_spans"])
    elif "joint_error" in spec:
        raise ObservationError("joint_error parameters require development_joint_ocular mode")
    strata = spec.get("strata", ["all"])
    if not isinstance(strata, list) or not strata or len(set(strata)) != len(strata) or any(not isinstance(s, str) or not s for s in strata):
        raise ObservationError("declare unique nonempty strata before acquisition")
    routing_policy = None
    if mode == "development_joint_ocular_stratified":
        from .stratified import RoutingPolicy, validate_routes
        route_spec = spec.get("stratified_error")
        if not isinstance(route_spec, dict) or set(route_spec) not in (
                {"definition", "minimum_calibration_groups"}, {"definition", "minimum_calibration_groups", "measurement_policy"}):
            raise ObservationError("stratified_error needs a pre-observation definition and minimum_calibration_groups")
        routing_policy = RoutingPolicy(tuple(strata), **route_spec)
        if (routing_policy.measurement_policy is not None
                and routing_policy.measurement_policy.maximum_group_members != max_group):
            raise ObservationError("measured enrollment size differs from the frozen study group maximum")
    elif "stratified_error" in spec:
        raise ObservationError("stratified_error requires development_joint_ocular_stratified mode")
    entries = spec.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ObservationError("study needs an enrolled case inventory")
    if routing_policy is not None and routing_policy.measurement_policy is None:
        validate_routes(tuple(entries), routing_policy)
    loaded, cases = [], []
    ids, images, tokens, group_sizes = set(), set(), {}, {}
    for original_entry in entries:
        e = dict(original_entry)
        if e.get("observation") or e.get("failure"):
            raise ObservationError("prospective locks do not import predictions or select by observed failures")
        case_id, group, split = e.get("case_id"), e.get("group_id"), e.get("split")
        if any(not isinstance(x, str) or not x.strip() for x in (case_id, group)) or split not in ("development", "calibration", "evaluation"):
            raise ObservationError("cases need IDs and explicit splits")
        if case_id in ids:
            raise ObservationError("duplicate case ID")
        ids.add(case_id)
        eligibility = e.get("eligibility", "included")
        if eligibility not in ("included", "excluded"):
            raise ObservationError("eligibility is included or prospectively excluded")
        reason = e.get("exclusion_reason", "")
        if (eligibility == "excluded") != bool(reason):
            raise ObservationError("only prospectively excluded cases require an exclusion reason")
        measured_routing = routing_policy is not None and routing_policy.measurement_policy is not None
        if not measured_routing and e.get("stratum", "all") not in strata:
            raise ObservationError("unregistered acquisition stratum")
        if not measured_routing and "acquisition_aperture" in e:
            raise ObservationError("source apertures require a frozen measured acquisition policy")
        lineages = e.get("lineage_ids", [])
        if not isinstance(lineages, list) or any(not isinstance(x, str) or not x for x in lineages) or len(set(lineages)) != len(lineages):
            raise ObservationError("lineage IDs must be unique nonempty strings")
        image_file = local_file(spec_path.parent, e["image"])
        image = load_rgb(image_file); digest = array_digest(image)
        if digest in images:
            raise ObservationError("duplicate decoded source image; do not count re-encodings independently")
        images.add(digest)
        for token in ["group:"+group] + ["lineage:"+s for s in lineages]:
            previous = tokens.get(token)
            if previous and previous != (group, split):
                raise ObservationError("source group/lineage split leakage")
            tokens[token] = (group, split)
        group_sizes[group] = group_sizes.get(group, 0) + 1
        if group_sizes[group] > max_group:
            raise ObservationError("enrolled group exceeds the prespecified maximum size")
        refs = tuple(ReferenceAnnotation.from_dict(read_json(local_file(spec_path.parent, p))) for p in e.get("references", []))
        if any(r.image_sha256 != digest or r.image_shape != image.shape[:2] for r in refs):
            raise ObservationError("reference does not match the enrolled source raster")
        if eligibility == "included":
            if spec.get("admission_policy") and observer.execution_kind == "native_mediapipe":
                if any(r.kind not in ("human_blind", "external_dataset") or r.prediction_exposure != "none_declared" for r in refs):
                    raise ObservationError("native empirical admission requires declared independent prediction-unexposed reference labels")
            consensus(refs, minimum).eye_span_px
            if observer.execution_kind == "native_mediapipe" and any(r.kind == "synthetic_fixture" for r in refs):
                # Native smoke tests on synthetic inputs can be run separately,
                # but must not silently become a human validation study.
                raise ObservationError("native validation needs independently annotated nonsynthetic references")
            if observer.execution_kind == "synthetic_fixture" and any(r.kind != "synthetic_fixture" for r in refs):
                raise ObservationError("fixture study cannot acquire human evidence")
            cases.append(ValidationCase(case_id, group, split, refs, None, "prospective_not_attempted", tuple(lineages)))
        loaded.append((e, image, refs, sha256(image_file.read_bytes()).hexdigest()))
    measured_groups = None
    if routing_policy is not None and routing_policy.measurement_policy is not None:
        from .acquisition import measure_frozen_groups
        by_id = {e["case_id"]: image for e, image, refs, encoded in loaded}
        measured_groups = measure_frozen_groups(routing_policy.measurement_policy,
            tuple(e for e, image, refs, encoded in loaded), lambda c: by_id[c["case_id"]])
        by_group = {g.group_id: g for g in measured_groups}
        for e, image, refs, encoded in loaded:
            group = by_group[e["group_id"]]
            if "stratum" in e and e["stratum"] != group.category:
                raise ObservationError("declared study category conflicts with measured source pixels")
            e["stratum"] = group.category
            member = next(x for x in group.sources if x.image_sha256 == array_digest(image))
            e["acquisition_aperture"] = list(member.aperture)
            e["acquisition_group_sha256"] = group.digest
        validate_routes(tuple(e for e, image, refs, encoded in loaded), routing_policy)
    validate_study(tuple(cases))
    if mode in ("development_ocular", "development_joint_ocular", "development_joint_ocular_stratified") and len({c.group_id for c in cases if c.split == "development"}) < min_dev:
        raise ObservationError("too few prospectively enrolled development groups for selected bias mode")
    if frozen_model is not None:
        from .transfer import require_transfer
        require_transfer(frozen_model, {**spec, "observer": observer.to_dict()}, [(
            {**e, "eligibility": e.get("eligibility", "included"), "lineage_ids": e.get("lineage_ids", []),
             "image_sha256": array_digest(image), "original_encoded_sha256": encoded}, refs)
            for e, image, refs, encoded in loaded])
    elif not any(c.split == "calibration" for c in cases) or not any(c.split == "evaluation" for c in cases):
        raise ObservationError("prospective qualification requires calibration and evaluation groups")
    if enrollment_registry is not None:
        from .enrollment import require_disjoint
        registry_check = require_disjoint(enrollment_registry, [(
            {**e, "eligibility": e.get("eligibility", "included"), "lineage_ids": e.get("lineage_ids", []),
             "image_sha256": array_digest(image), "image_shape": list(image.shape[:2]),
             "original_encoded_sha256": encoded}, refs) for e, image, refs, encoded in loaded])
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".facemath-study-", dir=target.parent))
    try:
        (temp / "inputs").mkdir()
        frozen_cases = []
        for i, (e, image, refs, encoded_sha) in enumerate(loaded):
            stem = f"inputs/case_{i:05d}"
            image_name = stem + ".png"; save_rgb(temp / image_name, image)
            reference_names = []
            for j, ref in enumerate(refs):
                name = f"{stem}_reference_{j:02d}.json"
                write_json_new(temp / name, ref.to_dict()); reference_names.append(name)
            frozen_cases.append({"case_id": e["case_id"], "group_id": e["group_id"], "split": e["split"],
                "lineage_ids": e.get("lineage_ids", []), "stratum": e.get("stratum", "all"),
                "eligibility": e.get("eligibility", "included"), "exclusion_reason": e.get("exclusion_reason", ""),
                "image": image_name, "image_sha256": array_digest(image), "image_shape": list(image.shape[:2]),
                "original_encoded_sha256": encoded_sha, "references": reference_names})
            if measured_groups is not None:
                frozen_cases[-1].update(acquisition_aperture=e["acquisition_aperture"],
                                        acquisition_group_sha256=e["acquisition_group_sha256"])
        lock = {"format": STUDY_FORMAT, "domain": spec["domain"], "observer": observer.to_dict(),
                "observer_recipe_sha256": observer.recipe_sha256, "minimum_raters": minimum,
                "maximum_cases_per_group": max_group, "minimum_development_groups": min_dev,
                "alpha": alpha, "bias_mode": mode, "strata": strata, "cases": frozen_cases,
                "timeout_seconds": float(timeout), "native_case_isolation": "fresh_process_per_source",
                "image_coordinate_convention": "decoded_stored_raster_ignore_EXIF_orientation",
                "engine_code_sha256": code_fingerprint(), "protocol_sha256": REFERENCE_PROTOCOL_SHA256,
                "source_manifest_sha256": sha256(spec_path.read_bytes()).hexdigest(),
                "declarations": {"chronology_and_blinding": "curator-declared, not externally attested",
                                 "model_receives": "source raster only; not reference coordinates, split or stratum"}}
        if "workspace_origin" in spec:
            lock["workspace_origin"] = spec["workspace_origin"]
        if enrollment_registry is not None:
            lock["format"] = REGISTRY_STUDY_FORMAT
            write_json_new(temp / "enrollment_registry.json", enrollment_registry)
            lock["enrollment_registry"] = {"path": "enrollment_registry.json",
                "content_sha256": enrollment_registry["content_sha256"], "check": registry_check}
        if frozen_model is not None:
            lock["frozen_model"] = frozen_model
        if joint_config is not None:
            lock["joint_error"] = joint_config
        if routing_policy is not None:
            lock["stratified_error"] = routing_policy.to_dict()
        if measured_groups is not None:
            lock["measured_acquisition"] = {"policy_sha256": routing_policy.measurement_policy.digest,
                                            "groups": [g.to_dict() for g in measured_groups]}
        if "admission_policy" in spec:
            from .admission import AdmissionPolicy, plan_admission
            admission = AdmissionPolicy.from_dict(spec["admission_policy"])
            if routing_policy is None or routing_policy.measurement_policy is None:
                raise ObservationError("empirical admission requires measured whole-group acquisition routing")
            lock["admission_policy"] = admission.to_dict()
            lock["admission_plan"] = plan_admission(admission, frozen_cases, strata, observer.execution_kind)
        write_json_new(temp / "study.lock.json", lock)
        manifest = {"format": MANIFEST_FORMATS[lock["format"]], "study_sha256": object_digest(lock),
                    "files": file_inventory(temp)}
        manifest["content_sha256"] = object_digest(manifest)
        write_json_new(temp / "manifest.json", manifest)
        load_study(temp)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temp, target)
        return load_study(target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
