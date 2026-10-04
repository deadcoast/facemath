"""Portable, prediction-free preparation for the existing prospective runner.

Sources and study settings are immutable. Annotation corrections are explicit
append-only revisions, not overwritten labels. Freeze selects each rater's
latest revision and preserves the complete revision history in the study lock.
Rater IDs express curator declarations, not proof of independent humans.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from .publication import publication_stage
from typing import Any
import os
import shutil

from .annotation_ui import create_annotation_workspace
from .contracts import ObservationError, array_digest, object_digest, canonical_json
from .reference import ReferenceAnnotation, consensus
from .study import (ObserverDeclaration, freeze_study, load_rgb, local_file, save_rgb, read_json,
                    write_json_new, file_inventory)
from .study_runner import writer_lock, utc_now

FORMAT = "facemath.reference_workspace.v10"
SPEC_FORMAT = "facemath.reference_workspace_spec.v10"


def _load(root: Path) -> dict[str, Any]:
    data = read_json(root / "workspace.json")
    if data.get("format") != FORMAT:
        raise ObservationError("unknown reference workspace format")
    digest = data.pop("content_sha256")
    if object_digest(data) != digest:
        raise ObservationError("workspace content digest mismatch")
    expected = data["files"]
    actual = file_inventory(root, exclude=("workspace.json", ".workspace-writer.lock"))
    immutable = {k: v for k, v in actual.items() if not k.startswith("labels/")}
    if immutable != expected:
        raise ObservationError("workspace source/workbench/settings inventory changed")
    data["content_sha256"] = digest
    return data


def _revision_state(root: Path, data: dict) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    receipts, selected, previous = [], {}, "0" * 64
    files = sorted((root / "labels").glob("*.receipt.json"))
    expected_files = set()
    by_id = {c["case_id"]: c for c in data["cases"]}
    for i, path in enumerate(files):
        if path.name != f"revision_{i:06d}.receipt.json":
            raise ObservationError("annotation receipt sequence has a gap or renamed entry")
        receipt = read_json(path)
        digest = receipt.pop("content_sha256")
        if (object_digest(receipt) != digest or receipt.get("sequence") != i
                or receipt.get("previous_sha256") != previous
                or receipt.get("workspace_sha256") != data["content_sha256"]):
            raise ObservationError("annotation receipt chain mismatch")
        case = by_id.get(receipt.get("case_id"))
        if case is None or receipt.get("rater_id") not in data["rater_ids"]:
            raise ObservationError("annotation receipt references an unknown case/rater")
        label_name = f"labels/revision_{i:06d}.reference.json"
        if receipt["reference"] != label_name:
            raise ObservationError("annotation receipt references an unexpected file")
        annotation = ReferenceAnnotation.from_dict(read_json(local_file(root, label_name)))
        if (annotation.digest != receipt["reference_sha256"] or annotation.image_sha256 != case["image_sha256"]
                or list(annotation.image_shape) != case["image_shape"] or annotation.rater_id != receipt["rater_id"]):
            raise ObservationError("annotation revision binding mismatch")
        if data["study"]["observer"]["execution_kind"] == "native_mediapipe":
            if annotation.kind not in ("human_blind", "external_dataset") or annotation.prediction_exposure != "none_declared":
                raise ObservationError("native reference workspace requires declared prediction-unexposed nonsynthetic labels")
        elif annotation.kind != "synthetic_fixture":
            raise ObservationError("synthetic workspace cannot import human-reference evidence")
        prior = selected.get((receipt["case_id"], receipt["rater_id"]))
        if receipt.get("supersedes_reference_sha256") != (prior["reference_sha256"] if prior else None):
            raise ObservationError("annotation supersession does not match the previous selected label")
        receipt["content_sha256"] = digest
        selected[(receipt["case_id"], receipt["rater_id"])] = receipt
        receipts.append(receipt); previous = digest
        expected_files.update((path.relative_to(root).as_posix(), label_name))
    actual = {p.relative_to(root).as_posix() for p in (root / "labels").rglob("*") if p.is_file()}
    if actual != expected_files:
        raise ObservationError("annotation intake has uncommitted or untracked files; preserve and repair before freezing")
    return receipts, selected


def prepare_workspace(spec_path: str | Path, destination: str | Path) -> Path:
    spec_path, target = Path(spec_path).resolve(), Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    spec = read_json(spec_path)
    if set(spec) != {"format", "study", "rater_ids", "sources", "source_description"} or spec["format"] != SPEC_FORMAT:
        raise ObservationError("workspace spec needs exactly format, study, rater_ids, sources and source_description")
    raters = spec["rater_ids"]
    if (not isinstance(raters, list) or not raters or any(not isinstance(r, str) or not r.strip() for r in raters)
            or len(set(raters)) != len(raters)):
        raise ObservationError("declare distinct nonempty rater IDs")
    settings = dict(spec["study"])
    if "cases" in settings or "workspace_origin" in settings:
        raise ObservationError("workspace study settings cannot contain predictions/cases or forged origin")
    if settings.get("format") != "facemath.prospective_study.v5":
        raise ObservationError("workspace uses the existing prospective study settings")
    declaration = settings.get("observer")
    if isinstance(declaration, str):
        declaration = read_json(local_file(spec_path.parent, declaration))
    observer = ObserverDeclaration.from_dict(declaration)
    settings["observer"] = observer.to_dict()
    if len(raters) < settings.get("minimum_raters", 1):
        raise ObservationError("too few declared raters for study minimum")
    if not isinstance(spec["source_description"], str) or not spec["source_description"].strip():
        raise ObservationError("source_description is required")
    sources = spec["sources"]
    if not isinstance(sources, list) or not sources:
        raise ObservationError("workspace requires an explicit source inventory")
    loaded, ids, pixels, dependencies, group_sizes = [], set(), set(), {}, {}
    permitted = {"case_id", "group_id", "split", "image", "lineage_ids", "eligibility", "exclusion_reason", "acquisition_aperture", "stratum"}
    for source in sources:
        if not isinstance(source, dict) or not set(source) <= permitted:
            raise ObservationError("workspace sources contain unknown fields; predictions and labels are not accepted")
        item = dict(source)
        for name in ("case_id", "group_id"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ObservationError("every source needs nonempty case and group IDs")
        if item.get("split") not in ("development", "calibration", "evaluation"):
            raise ObservationError("declare a source split before observation")
        if item["case_id"] in ids:
            raise ObservationError("duplicate workspace case ID")
        ids.add(item["case_id"])
        lineages = item.get("lineage_ids", [])
        if not isinstance(lineages, list) or any(not isinstance(v, str) or not v for v in lineages) or len(set(lineages)) != len(lineages):
            raise ObservationError("source lineage IDs must be unique strings")
        item["lineage_ids"] = lineages
        item.setdefault("eligibility", "included"); item.setdefault("exclusion_reason", "")
        if item["eligibility"] not in ("included", "excluded") or ((item["eligibility"] == "excluded") != bool(item["exclusion_reason"])):
            raise ObservationError("invalid prospective source exclusion")
        path = local_file(spec_path.parent, item["image"])
        image = load_rgb(path); digest = array_digest(image)
        if digest in pixels:
            raise ObservationError("duplicate source pixels cannot be enrolled twice")
        pixels.add(digest)
        for token in ["group:" + item["group_id"], *("lineage:" + v for v in lineages)]:
            assignment = (item["group_id"], item["split"])
            if token in dependencies and dependencies[token] != assignment:
                raise ObservationError("workspace group/lineage crosses splits or groups")
            dependencies[token] = assignment
        count = group_sizes.get(item["group_id"], 0) + 1
        group_sizes[item["group_id"]] = count
        if count > settings.get("maximum_cases_per_group", 1):
            raise ObservationError("workspace group exceeds frozen maximum")
        if "acquisition_aperture" in item:
            from .acquisition import ApertureSource
            ApertureSource(image, tuple(lineages), tuple(item["acquisition_aperture"]))
        loaded.append((item, image, sha256(path.read_bytes()).hexdigest()))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = publication_stage(target.parent, ".facemath-workspace-")
    try:
        (temp / "sources").mkdir(); (temp / "workbenches").mkdir(); (temp / "labels").mkdir()
        cases = []
        for i, (item, image, encoded) in enumerate(loaded):
            name = f"sources/source_{i:05d}.png"
            save_rgb(temp / name, image)
            case = {**item, "image": name, "image_sha256": array_digest(image),
                    "image_shape": list(image.shape[:2]), "original_encoded_sha256": encoded,
                    "workbenches": {}}
            if item["eligibility"] == "included":
                for j, rater in enumerate(raters):
                    html = f"workbenches/source_{i:05d}_rater_{j:03d}.html"
                    create_annotation_workspace(image, temp / html,
                        source=f"{spec['source_description']} | case {item['case_id']} | assigned rater {rater}")
                    case["workbenches"][rater] = html
            cases.append(case)
        payload = {"format": FORMAT, "study": settings, "rater_ids": raters,
            "source_description": spec["source_description"], "cases": cases,
            "files": file_inventory(temp), "predictions_shown": False,
            "created_at": utc_now(), "revision_rule": "latest explicit import per case/rater; every prior label remains retained",
            "limit": "IDs do not prove independent humans; hashes do not attest real capture or blindness"}
        payload["content_sha256"] = object_digest(payload)
        write_json_new(temp / "workspace.json", payload)
        _load(temp)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temp, target)
        return target
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def import_reference(root: str | Path, reference_path: str | Path, case_id: str) -> dict[str, Any]:
    root = Path(root).resolve()
    with writer_lock(root / ".workspace-writer.lock"):
        data = _load(root); receipts, selected = _revision_state(root, data)
        matches = [c for c in data["cases"] if c["case_id"] == case_id]
        if not matches or matches[0]["eligibility"] != "included":
            raise ObservationError("annotation intake needs an included enrolled case")
        case = matches[0]
        ref = ReferenceAnnotation.from_dict(read_json(reference_path))
        if ref.rater_id not in data["rater_ids"]:
            raise ObservationError("annotation rater is outside the assigned inventory")
        if ref.image_sha256 != case["image_sha256"] or list(ref.image_shape) != case["image_shape"]:
            raise ObservationError("annotation belongs to another source raster")
        native = data["study"]["observer"]["execution_kind"] == "native_mediapipe"
        if native and (ref.kind not in ("human_blind", "external_dataset") or ref.prediction_exposure != "none_declared"):
            raise ObservationError("native labels must be nonsynthetic and declared unexposed to predictions")
        if not native and ref.kind != "synthetic_fixture":
            raise ObservationError("synthetic workspace requires synthetic reference provenance")
        previous = selected.get((case_id, ref.rater_id))
        if previous is not None and previous["reference_sha256"] == ref.digest:
            return {"imported": False, "idempotent": True, "receipt": previous}
        i = len(receipts)
        name = f"labels/revision_{i:06d}.reference.json"
        receipt = {"format": "facemath.annotation_intake.v10", "sequence": i,
            "workspace_sha256": data["content_sha256"], "case_id": case_id, "rater_id": ref.rater_id,
            "reference": name, "reference_sha256": ref.digest, "imported_at": utc_now(),
            "previous_sha256": receipts[-1]["content_sha256"] if receipts else "0" * 64,
            "supersedes_reference_sha256": previous["reference_sha256"] if previous else None}
        receipt["content_sha256"] = object_digest(receipt)
        # A crash between these publications leaves an explicit uncommitted
        # orphan that status refuses; it can never become a selected label.
        write_json_new(root / name, ref.to_dict())
        write_json_new(root / f"labels/revision_{i:06d}.receipt.json", receipt)
        _revision_state(root, data)
        return {"imported": True, "idempotent": False, "receipt": receipt}


def workspace_status(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    with writer_lock(root / ".workspace-writer.lock"):
        data = _load(root); receipts, selected = _revision_state(root, data)
        rows, blockers = [], []
        minimum = data["study"].get("minimum_raters", 1)
        for case in data["cases"]:
            refs = [ReferenceAnnotation.from_dict(read_json(root / selected[(case["case_id"], r)]["reference"]))
                    for r in data["rater_ids"] if (case["case_id"], r) in selected]
            reasons = []
            if case["eligibility"] == "included":
                if len(refs) < minimum:
                    reasons.append(f"{len(refs)}/{minimum} minimum rater labels available")
                else:
                    try:
                        consensus(tuple(refs), minimum).eye_span_px
                    except ObservationError as exc:
                        reasons.append(str(exc))
            blockers.extend(f"{case['case_id']}: {r}" for r in reasons)
            rows.append({"case_id": case["case_id"], "eligibility": case["eligibility"],
                "group_id": case["group_id"], "split": case["split"], "labels": len(refs),
                "ready": not reasons, "reasons": reasons, "workbenches": case["workbenches"]})
        return {"format": "facemath.workspace_readiness.v10", "workspace_sha256": data["content_sha256"],
                "ready_to_freeze_labels": not blockers, "blockers": blockers, "cases": rows,
                "retained_annotation_revisions": len(receipts), "selected_labels": len(selected),
                "native_inference_executed": False, "source_pixels_verified": True,
                "next_step": "workspace freeze" if not blockers else "annotate and workspace import missing references",
                "limit": "label readiness only; freezer validates full protocol and runtime preflight occurs separately"}


def freeze_workspace(root: str | Path, destination: str | Path):
    root = Path(root).resolve()
    # One lock covers intake selection and all original pixel copying.
    with writer_lock(root / ".workspace-writer.lock"):
        data = _load(root); receipts, selected = _revision_state(root, data)
        with TemporaryDirectory(prefix="facemath-study-inputs-") as directory:
            stage = Path(directory)
            shutil.copytree(root / "sources", stage / "sources")
            shutil.copytree(root / "labels", stage / "labels")
            entries = []
            for case in data["cases"]:
                entry = {k: v for k, v in case.items() if k in ("case_id", "group_id", "split", "image", "lineage_ids", "eligibility", "exclusion_reason", "acquisition_aperture", "stratum")}
                entry["references"] = [selected[(case["case_id"], r)]["reference"] for r in data["rater_ids"] if (case["case_id"], r) in selected]
                entries.append(entry)
            spec = {**data["study"], "cases": entries,
                "workspace_origin": {"format": "facemath.workspace_origin.v10",
                    "workspace_sha256": data["content_sha256"], "annotation_receipts": receipts,
                    "reference_revisions": [read_json(root / r["reference"]) for r in receipts],
                    "selected_reference_sha256": sorted(r["reference_sha256"] for r in selected.values()),
                    "source_inventory": [{k: c[k] for k in ("case_id", "image_sha256", "original_encoded_sha256")} for c in data["cases"]]}}
            write_json_new(stage / "study.json", spec)
            return freeze_study(stage / "study.json", destination)


def verify_workspace_origin(data: dict, root: Path) -> None:
    """Recheck retained label history against the actual selected frozen labels.

    This is semantic consistency checking, not a claim that a hash chain proves
    human independence or that a curator did not recreate an entire history.
    """
    origin = data.get("workspace_origin")
    if origin is None:
        return
    if not isinstance(origin, dict) or origin.get("format") != "facemath.workspace_origin.v10":
        raise ObservationError("unknown workspace-origin contract")
    receipts = origin.get("annotation_receipts", [])
    revisions = origin.get("reference_revisions", [])
    if len(receipts) != len(revisions):
        raise ObservationError("workspace origin lost reference revisions")
    cases = {c["case_id"]: c for c in data["cases"]}
    selected, previous = {}, "0"*64
    for i, (original, record) in enumerate(zip(receipts, revisions)):
        receipt = dict(original)
        digest = receipt.pop("content_sha256", None)
        if (digest != object_digest(receipt) or receipt.get("sequence") != i
                or receipt.get("previous_sha256") != previous
                or receipt.get("workspace_sha256") != origin["workspace_sha256"]
                or receipt.get("reference") != f"labels/revision_{i:06d}.reference.json"):
            raise ObservationError("frozen annotation history chain mismatch")
        reference = ReferenceAnnotation.from_dict(record)
        case = cases.get(receipt.get("case_id"))
        if (case is None or case["eligibility"] != "included"
                or reference.image_sha256 != case["image_sha256"]
                or list(reference.image_shape) != case["image_shape"]
                or receipt.get("rater_id") != reference.rater_id
                or receipt.get("reference_sha256") != reference.digest):
            raise ObservationError("frozen annotation revision binding mismatch")
        key = (case["case_id"], reference.rater_id)
        if receipt.get("supersedes_reference_sha256") != selected.get(key):
            raise ObservationError("frozen annotation supersession mismatch")
        selected[key] = reference.digest
        previous = digest
    if sorted(selected.values()) != origin.get("selected_reference_sha256"):
        raise ObservationError("workspace selected-label inventory mismatch")
    actual = {}
    for case in cases.values():
        for name in case["references"]:
            ref = ReferenceAnnotation.from_dict(read_json(local_file(root, name)))
            actual[(case["case_id"], ref.rater_id)] = ref.digest
    if actual != selected:
        raise ObservationError("frozen references differ from latest retained annotation revisions")
    inventory = origin.get("source_inventory", [])
    if len(inventory) != len(cases) or len({c["case_id"] for c in inventory}) != len(cases):
        raise ObservationError("workspace source inventory mismatch")
    for source in inventory:
        case = cases.get(source["case_id"])
        if case is None or source["image_sha256"] != case["image_sha256"]:
            raise ObservationError("workspace source pixels differ from frozen study")
