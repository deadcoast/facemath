"""Portable frozen joint models and evaluation-only transfer; never fits a model."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .contracts import ObservationError, object_digest
from .joint import JointCalibration, joint_scored_rows
from .reference import ReferenceAnnotation, REFERENCE_PROTOCOL_SHA256, check_digest
from .study import ObserverDeclaration, read_json, write_json_new

CAPSULE_FORMAT = "facemath.frozen_model_capsule.v14"
REGISTRY_CAPSULE_FORMAT = "facemath.frozen_model_capsule.v16"
REPORT_FORMAT = "facemath.model_transfer.v14"
SOURCE_HASHES = ("source_encoded_sha256", "source_raster_sha256")


def _bound_json(capsule: dict, text: str, name: str) -> dict:
    data = text.encode("utf-8")
    pin = capsule["donor_manifest"]["files"][name]
    if len(data) != pin["bytes"] or sha256(data).hexdigest() != pin["sha256"]:
        raise ObservationError("capsule donor artifact binding mismatch: " + name)
    return json.loads(text)


def load_capsule(capsule: dict) -> JointCalibration:
    """Validate the full fitted state and donor enrollment, including exclusions."""
    unsigned = {k: v for k, v in capsule.items() if k != "content_sha256"}
    if capsule.get("format") not in (CAPSULE_FORMAT, REGISTRY_CAPSULE_FORMAT) or object_digest(unsigned) != capsule.get("content_sha256"):
        raise ObservationError("frozen model capsule digest mismatch")
    manifest = dict(capsule["donor_manifest"])
    digest = manifest.pop("content_sha256")
    if manifest.get("format") != "facemath.qualification_bundle.v5" or object_digest(manifest) != digest:
        raise ObservationError("capsule donor manifest mismatch")
    model = JointCalibration.from_dict(_bound_json(capsule, capsule["model_json"], "localization_calibration.json"))
    study = _bound_json(capsule, capsule["study_json"], "acquisition/study/study.lock.json")
    if (object_digest(study) != manifest["study_sha256"] or study.get("frozen_model") is not None
            or study["bias_mode"] != "development_joint_ocular"
            or study["protocol_sha256"] != REFERENCE_PROTOCOL_SHA256
            or study["domain"] != model.residual.domain
            or study["alpha"] != model.alpha or study["minimum_raters"] != model.residual.minimum_raters
            or ObserverDeclaration.from_dict(study["observer"]).recipe_sha256 != model.bias.observer_recipe_sha256):
        raise ObservationError("capsule model differs from donor study protocol")
    expected_refs = {n for c in study["cases"] for n in c["references"]}
    if set(capsule["references_json"]) != expected_refs:
        raise ObservationError("capsule lost donor reference enrollment")
    refs = {}
    for name, value in capsule["references_json"].items():
        refs[name] = ReferenceAnnotation.from_dict(_bound_json(capsule, value, "acquisition/study/" + name))
    for case in study["cases"]:
        for name in case["references"]:
            ref = refs[name]
            if ref.image_sha256 != case["image_sha256"] or list(ref.image_shape) != case["image_shape"]:
                raise ObservationError("capsule donor reference/image mismatch")
            if case["eligibility"] == "included" and ref.kind != model.reference_kind:
                raise ObservationError("capsule donor reference kind mismatch")
    if "enrollment_registry" in study:
        from .enrollment import require_disjoint
        if capsule["format"] != REGISTRY_CAPSULE_FORMAT or study["format"] != "facemath.prospective_study.v16":
            raise ObservationError("registry donor requires a v16 model capsule")
        registry = _bound_json(capsule, capsule["registry_json"], "acquisition/study/enrollment_registry.json")
        check = require_disjoint(registry, [(c, tuple(refs[n] for n in c["references"])) for c in study["cases"]])
        if (study["enrollment_registry"]["content_sha256"] != registry["content_sha256"]
                or study["enrollment_registry"]["check"] != check):
            raise ObservationError("capsule donor registry binding mismatch")
    elif capsule["format"] == REGISTRY_CAPSULE_FORMAT or "registry_json" in capsule or study["format"] != "facemath.prospective_study.v5":
        raise ObservationError("capsule registry format differs from donor study")
    dev = [c for c in study["cases"] if c["split"] == "development" and c["eligibility"] == "included"]
    cal = [c for c in study["cases"] if c["split"] == "calibration" and c["eligibility"] == "included"]
    for cases, images, groups, lineages, references in (
        (dev, model.bias.development_images, model.bias.development_groups,
         model.bias.development_lineages, model.bias.development_reference_digests),
        (cal, model.residual.calibration_images, dict(model.residual.group_scores),
         model.residual.calibration_lineages, model.residual.reference_digests),
    ):
        if (set(images) != {c["image_sha256"] for c in cases}
                or set(groups) != {c["group_id"] for c in cases}
                or set(lineages) != {s for c in cases for s in c["lineage_ids"]}
                or set(references) != {refs[n].digest for c in cases for n in c["references"]}):
            raise ObservationError("capsule fitted model lost donor enrollment")
    return model


def export_model(qualification: str | Path, destination: str | Path) -> dict:
    from .study_io import verify_qualification_bundle
    root, target = Path(qualification).resolve(), Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    verify_qualification_bundle(root)
    study = read_json(root / "acquisition/study/study.lock.json")
    # Decode bytes directly: text-mode newline translation would change pins.
    capsule = {"format": CAPSULE_FORMAT,
        "model_json": (root / "localization_calibration.json").read_bytes().decode("utf-8"),
        "study_json": (root / "acquisition/study/study.lock.json").read_bytes().decode("utf-8"),
        "references_json": {n: (root / "acquisition/study" / n).read_bytes().decode("utf-8")
                            for c in study["cases"] for n in c["references"]},
        "donor_manifest": read_json(root / "manifest.json"),
        "limit": "verified donor byte consistency and complete fitted state; not external authenticity, new inference or empirical admission"}
    if "enrollment_registry" in study:
        capsule["format"] = REGISTRY_CAPSULE_FORMAT
        capsule["registry_json"] = (root / "acquisition/study/enrollment_registry.json").read_bytes().decode("utf-8")
    capsule["content_sha256"] = object_digest(capsule)
    model = load_capsule(capsule)
    write_json_new(target, capsule)
    return {"capsule": str(target), "calibration_sha256": model.digest,
            "capsule_sha256": capsule["content_sha256"], "donor_enrolled_cases": len(study["cases"])}


def _image_tokens(case: dict, references: tuple[ReferenceAnnotation, ...]) -> set[str]:
    # Compare all hashes in a single namespace to catch original/crop aliases.
    values = {case["image_sha256"], case["original_encoded_sha256"]}
    for ref in references:
        metadata = json.loads(ref.metadata_json)
        for key in SOURCE_HASHES:
            if key in metadata:
                check_digest(metadata[key], key)
                values.add(metadata[key])
    return values


def require_transfer(capsule: dict, spec: dict, cases_and_references: list | tuple) -> JointCalibration:
    model = load_capsule(capsule)
    if (spec.get("bias_mode", "none") != "none" or "joint_error" in spec or "admission_policy" in spec
            or "stratified_error" in spec or spec.get("alpha", .1) != model.alpha
            or spec.get("minimum_raters", 1) != model.residual.minimum_raters
            or ObserverDeclaration.from_dict(spec["observer"]).recipe_sha256 != model.bias.observer_recipe_sha256):
        raise ObservationError("transfer must preserve observer/rater/alpha contracts and cannot fit or admit")
    donor = json.loads(capsule["study_json"])
    images, groups, lineages = set(), set(), set()
    for case in donor["cases"]:
        refs = tuple(ReferenceAnnotation.from_dict(json.loads(capsule["references_json"][n])) for n in case["references"])
        images.update(_image_tokens(case, refs)); groups.add(case["group_id"]); lineages.update(case["lineage_ids"])
    if not cases_and_references or not any(c["eligibility"] == "included" for c, refs in cases_and_references):
        raise ObservationError("transfer requires eligible evaluation cases")
    if "registry_json" in capsule:
        from .enrollment import require_disjoint
        require_disjoint(json.loads(capsule["registry_json"]), cases_and_references)
    for case, refs in cases_and_references:
        if case["split"] != "evaluation":
            raise ObservationError("transfer accepts evaluation cases ONLY")
        if (images & _image_tokens(case, refs) or case["group_id"] in groups
                or lineages & set(case["lineage_ids"])):
            raise ObservationError("transfer overlaps full donor enrollment (image, source, group or lineage)")
        if any(r.kind != model.reference_kind for r in refs):
            raise ObservationError("transfer reference kind differs from donor model")
    return model


def evaluate_transfer(study: Any, raw_report: dict, status: dict) -> dict:
    from .error_model import evaluate_coverage
    from .qualification import rows_from_report
    from .utility import joint_bound_utility
    data = study.data
    model = require_transfer(data["frozen_model"], data, [(c, study.references(c)) for c in study.cases])
    rows = rows_from_report(raw_report)
    if {r.case_id for r in rows} != {c["case_id"] for c in study.included}:
        raise ObservationError("transfer rows differ from frozen eligible inventory")
    lookup = {c["case_id"]: c for c in study.included}
    for row in rows:
        case = lookup[row.case_id]
        if (row.split != "evaluation" or row.image_sha256 != case["image_sha256"]
                or row.group_id != case["group_id"] or tuple(case["lineage_ids"]) != row.lineage_ids):
            raise ObservationError("transfer row differs from frozen case")
    scored = joint_scored_rows(rows, model.bias, model.scatter)
    coverage = evaluate_coverage(model.residual, scored)
    # The donor nominal value describes its calibration design, not a target guarantee.
    coverage["donor_nominal_coverage"] = coverage.pop("nominal_coverage")
    coverage["target_coverage_guaranteed"] = False
    report = {"format": REPORT_FORMAT, "study_sha256": study.digest,
        "capsule_sha256": data["frozen_model"]["content_sha256"], "calibration_sha256": model.digest,
        "bias_sha256": model.bias.digest, "scatter_sha256": model.scatter.digest,
        "radius": model.quantile, "donor_domain": model.residual.domain, "target_domain": data["domain"],
        "domain_names_match": model.residual.domain == data["domain"],
        "fit_performed": False, "raw_coordinates_changed": False, "empirical_admission": False,
        "coverage": coverage, "raw_localization": raw_report["splits"]["evaluation"],
        "outcomes": status["outcomes"], "enrolled_cases": len(study.cases), "eligible_cases": len(rows),
        "prospectively_excluded": len(study.cases)-len(rows),
        "bound_utility": joint_bound_utility(rows, model, study.cases),
        "cases": [{"case_id": r.case_id, "group_id": r.group_id, "score": r.simultaneous_score,
                   "failure": r.failure} for r in scored],
        "limit": "diagnostic transfer to frozen target; domain names, source disjointness and observed coverage do not establish exchangeability, person independence, anatomical truth or target query admission"}
    report["content_sha256"] = object_digest(report)
    return report
