"""Pinned published-reference data -> the existing frozen native study pipeline.

Network access is explicit in fetch_dataset only. Preparation never infers points,
selects detector successes, resizes pixels, or claims independent human labels.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
import json
import math
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import ObservationError, array_digest, canonical_json, object_digest
from .journal import Journal, environment, verify_journal
from .reference import IBUG68_INDICES, ReferenceAnnotation, check_digest, consensus
from .study import (ObserverDeclaration, file_inventory, freeze_study, load_rgb,
                    local_file, read_json, save_rgb, verify_inventory, write_json_new)

RECIPE_FORMAT = "facemath.dlib_dataset_recipe.v11"
DATASET_FORMAT = "facemath.dataset_bundle.v11"
PIPELINE_FORMAT = "facemath.native_pipeline_bundle.v11"


def _name(value: str) -> str:
    # Flat assets are deliberate: portable paths, no archive or URL path inference.
    if (not isinstance(value, str) or not value or value in (".", "..")
            or any(c in value for c in '/\\:\0<>|?*"') or any(ord(c) < 32 for c in value) or value[-1] in ' .'
            or value.split('.')[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}):
        raise ObservationError("dataset asset must have a portable basename")
    return value


def _recipe(path: Path) -> dict:
    recipe = read_json(path)
    from .source_cohort import RECIPE_FORMATS as PTS_FORMATS, verify_recipe
    pts = recipe.get("format") in PTS_FORMATS
    if pts:
        verify_recipe(recipe)
    elif recipe.get("format") != RECIPE_FORMAT or recipe.get("label_kind") != "model_generated":
        raise ObservationError("this adapter requires an explicit model_generated dlib recipe")
    if type(recipe.get("coordinate_origin")) is not int or recipe["coordinate_origin"] not in (0, 1):
        raise ObservationError("declare coordinate origin 0 or 1")
    margin = recipe.get("box_margin")
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ObservationError("box margin must be finite and between zero and one")
    for key in ("name", "source", "label_source", "image_terms", "grouping_limit", "runtime_version"):
        if not isinstance(recipe.get(key), str) or not recipe[key].strip():
            raise ObservationError(f"dataset must declare {key}")
    assets = recipe.get("assets", [])
    names = [_name(a["path"]) for a in assets]
    if (not assets or len({n.casefold() for n in names}) != len(names)
            or {n.casefold() for n in names} & {"recipe.json", "fetch.json", "fetch.failure.json"}):
        raise ObservationError("dataset requires unique pinned assets")
    for asset in assets:
        check_digest(asset["sha256"], "dataset asset")
        if type(asset.get("bytes")) is not int or not 0 < asset["bytes"] <= 128 * 1024**2:
            raise ObservationError("dataset asset size must be between 1 and 128 MiB")
        url = urlsplit(asset["url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ObservationError("dataset asset URL must be public HTTPS without credentials")
    if sum(a["bytes"] for a in assets) > 512 * 1024**2:
        raise ObservationError("dataset recipe exceeds the 512 MiB intake limit")
    annotations = recipe.get("annotations", [])
    if not annotations or len(set(annotations)) != len(annotations) or not set(annotations) <= set(names):
        raise ObservationError("declare unique pinned annotation assets")
    images = recipe.get("images", [])
    if not images or len({i["path"] for i in images}) != len(images):
        raise ObservationError("declare unique image enrollment")
    if not {i["path"] for i in images} <= set(names) or recipe.get("model_asset") not in names:
        raise ObservationError("enrolled images and native model must be pinned assets")
    groups, lineages = {}, {}
    for image in images:
        group, split = image.get("group_id"), image.get("split")
        if not isinstance(group, str) or not group.strip() or split not in ("development", "calibration", "evaluation"):
            raise ObservationError("source images require groups and explicit splits")
        if group in groups and groups[group] != split:
            raise ObservationError("source group split leakage")
        groups[group] = split
        ids = image.get("lineage_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ObservationError("source images require unique nonempty lineage IDs")
        for token in ids:
            if token in lineages and lineages[token] != (group, split):
                raise ObservationError("source lineage split leakage")
            lineages[token] = (group, split)
    protocol = recipe.get("protocol", {})
    allowed = {"domain", "minimum_raters", "maximum_cases_per_group", "minimum_development_groups",
               "timeout_seconds", "alpha", "bias_mode", "joint_error", "strata"}
    if not isinstance(protocol, dict) or set(protocol) - allowed:
        raise ObservationError("unsupported dataset protocol; machine references cannot request empirical admission")
    return recipe


def _check_asset(path: Path, asset: dict) -> None:
    if path.stat().st_size != asset["bytes"] or sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
        raise ObservationError(f"dataset asset pin mismatch: {asset['path']}")


def fetch_dataset(recipe_path: str | Path, destination: str | Path) -> Path:
    """Explicit opt-in download; original failures and partial bytes are retained."""
    recipe_path, root = Path(recipe_path).resolve(), Path(destination).resolve()
    recipe = _recipe(recipe_path)
    if recipe['format'] != RECIPE_FORMAT:
        raise ObservationError("archive-derived PTS assets require dataset cohort-fetch and cohort intake")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(recipe_path, root / "recipe.json")
    receipts = []
    try:
        for asset in recipe["assets"]:
            with urlopen(asset["url"], timeout=90) as response:
                if urlsplit(response.url).scheme != "https":
                    raise ObservationError("dataset download redirected away from HTTPS")
                data = response.read(asset["bytes"] + 1)
                record = {"path": asset["path"], "url": asset["url"], "final_url": response.url,
                          "bytes": len(data), "sha256": sha256(data).hexdigest()}
            with (root / asset["path"]).open("xb") as stream:
                stream.write(data)
            _check_asset(root / asset["path"], asset)
            receipts.append(record)
    except Exception as exc:
        write_json_new(root / "fetch.failure.json", {"completed": False, "verified_downloads": receipts,
                                                    "exception": type(exc).__name__, "detail": str(exc)})
        raise
    write_json_new(root / "fetch.json", {"completed": True, "assets": receipts})
    return root


def _source_boxes(recipe: dict, assets: Path) -> dict:
    """Adapt label containers only; both use the same pixel/crop conversion below."""
    if recipe['format'] != RECIPE_FORMAT:
        from .reference import parse_ibug68
        result = {}
        for entry in recipe['images']:
            path = local_file(assets, entry['annotation'])
            if path.stat().st_size > 64 * 1024:
                raise ObservationError("PTS annotation exceeds 64 KiB")
            points = parse_ibug68(path.read_text(encoding='utf-8'))
            lo = np.floor(points.min(axis=0)); hi = np.ceil(points.max(axis=0))
            box = ET.Element('box', left=str(int(lo[0])), top=str(int(lo[1])),
                             width=str(int(hi[0]-lo[0]+1)), height=str(int(hi[1]-lo[1]+1)))
            for i, (x, y) in enumerate(points):
                ET.SubElement(box, 'part', name=f'{i:02}', x=str(float(x)), y=str(float(y)))
            result[entry['path']] = (entry['annotation'], [box])
        return result
    source_boxes = {}
    for name in recipe["annotations"]:
        data = (assets / name).read_bytes()
        if len(data) > 8 * 1024**2 or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
            raise ObservationError("unsupported annotation XML declarations or size")
        try:
            xml = ET.fromstring(data)
        except ET.ParseError as exc:
            raise ObservationError("invalid annotation XML") from exc
        if xml.tag != "dataset":
            raise ObservationError("expected dlib dataset XML")
        for image in xml.findall("./images/image"):
            key = _name(image.attrib["file"])
            if key in source_boxes or not image.findall("box"):
                raise ObservationError("duplicate image or empty annotation inventory")
            source_boxes[key] = (name, image.findall("box"))
    if set(source_boxes) != {i["path"] for i in recipe["images"]}:
        raise ObservationError("XML and declared source-image enrollment must match exactly")
    return source_boxes


def _plan(recipe: dict, assets: Path) -> tuple[list, dict]:
    pins = {a["path"]: a for a in recipe["assets"]}
    for asset in pins.values():
        _check_asset(local_file(assets, asset["path"]), asset)
    source_boxes = _source_boxes(recipe, assets)
    cases, records, seen = [], [], set()
    for entry in recipe["images"]:
        source = load_rgb(assets / entry["path"])
        digest = array_digest(source)
        if digest in seen:
            raise ObservationError("duplicate decoded source image")
        seen.add(digest)
        annotation, boxes = source_boxes[entry["path"]]
        for box_index, box in enumerate(boxes):
            parts = box.findall("part")
            keys = [p.attrib["name"] for p in parts]
            if len(keys) != 68 or set(keys) != {f"{i:02}" for i in range(68)}:
                raise ObservationError("each annotated box requires exactly 68 unique parts 00..67")
            lookup = {p.attrib["name"]: (float(p.attrib["x"]), float(p.attrib["y"])) for p in parts}
            points = np.array([lookup[f"{i:02}"] for i in IBUG68_INDICES]) - recipe["coordinate_origin"]
            if not np.isfinite(points).all():
                raise ObservationError("annotation coordinates must be finite")
            left, top, width, height = (int(box.attrib[k]) for k in ("left", "top", "width", "height"))
            if width <= 0 or height <= 0:
                raise ObservationError("annotation box dimensions must be positive")
            left -= recipe["coordinate_origin"]; top -= recipe["coordinate_origin"]
            margin = recipe["box_margin"]
            h, w = source.shape[:2]
            x0, y0 = max(0, math.floor(left - margin * width)), max(0, math.floor(top - margin * height))
            x1, y1 = min(w, math.ceil(left + (1 + margin) * width)), min(h, math.ceil(top + (1 + margin) * height))
            if x1 - x0 < 8 or y1 - y0 < 8:
                raise ObservationError("annotated crop must contain at least 8x8 stored pixels")
            crop = np.ascontiguousarray(source[y0:y1, x0:x1])
            xy = points - [x0, y0]
            visible = (xy[:, 0] >= 0) & (xy[:, 0] <= x1-x0-1) & (xy[:, 1] >= 0) & (xy[:, 1] <= y1-y0-1)
            xy[~visible] = 0
            metadata = {"source_schema": "ibug68", "source_indices_zero_based": list(IBUG68_INDICES),
                "source_origin": recipe["coordinate_origin"], "source_encoded_sha256": pins[entry["path"]]["sha256"],
                "source_raster_sha256": digest, "annotation_sha256": pins[annotation]["sha256"],
                "box_index": box_index, "crop_xyxy_exclusive": [x0, y0, x1, y1],
                "source_to_crop": [[1, 0, -x0], [0, 1, -y0], [0, 0, 1]],
                "visibility": "no_occlusion_labels; in_crop_assumed_observable", "mapping_requires_review": True,
                "label_source": recipe["label_source"], "grouping_limit": recipe["grouping_limit"],
                "image_terms": recipe["image_terms"]}
            if recipe['label_kind'] == 'external_dataset':
                metadata.update(source_format='original_pts', crop_basis='inclusive_integer_68_point_extrema',
                                cohort_archive_sha256=recipe['cohort']['plan']['archive']['sha256'])
            rater = 'published_teacher_model' if recipe['label_kind'] == 'model_generated' else 'published_dataset_annotation'
            ref = ReferenceAnnotation(array_digest(crop), crop.shape[:2], xy, visible,
                rater, recipe['label_kind'], recipe["source"], "unknown", canonical_json(metadata))
            reason = ""
            try:
                consensus((ref,)).eye_span_px
            except ObservationError as exc:
                reason = "prospective_reference_support: " + str(exc)
            case_id = f"case_{len(cases):05d}"
            case = {"case_id": case_id, "group_id": entry["group_id"], "split": entry["split"],
                "lineage_ids": entry["lineage_ids"], "stratum": "all", "eligibility": "excluded" if reason else "included",
                "exclusion_reason": reason, "image": f"inputs/{case_id}.png", "references": [f"inputs/{case_id}.reference.json"]}
            cases.append((case, crop, ref))
            records.append({"case_id": case_id, "source": entry["path"], "annotation": annotation,
                            "source_box_index": box_index, "image_sha256": ref.image_sha256,
                            "reference_sha256": ref.digest, **metadata})
    observer = ObserverDeclaration.mediapipe(pins[recipe["model_asset"]]["sha256"], recipe["runtime_version"])
    spec = {**recipe["protocol"], "format": "facemath.prospective_study.v5", "observer": observer.to_dict(),
            "cases": [case for case, _, _ in cases]}
    return cases, {"study_spec": spec, "conversions": records}


def _seal(root: Path, kind: str) -> None:
    manifest = {"format": kind, "files": file_inventory(root)}
    manifest["content_sha256"] = object_digest(manifest)
    write_json_new(root / "manifest.json", manifest)


def _verify_seal(root: Path, kind: str) -> None:
    if root.is_symlink():
        raise ObservationError("bundle root cannot be a symlink")
    manifest = read_json(root / "manifest.json")
    digest = manifest.pop("content_sha256")
    if manifest.get("format") != kind or object_digest(manifest) != digest:
        raise ObservationError("dataset/pipeline manifest mismatch")
    verify_inventory(root, manifest["files"], exclude=("manifest.json",))


def prepare_dataset(recipe_path: str | Path, destination: str | Path) -> Path:
    recipe_path, root = Path(recipe_path).resolve(), Path(destination).resolve()
    recipe = _recipe(recipe_path)
    cases, plan = _plan(recipe, recipe_path.parent)
    root.mkdir(parents=True, exist_ok=False)
    (root / "assets").mkdir(); (root / "inputs").mkdir()
    shutil.copyfile(recipe_path, root / "recipe.json")
    for asset in recipe["assets"]:
        shutil.copyfile(local_file(recipe_path.parent, asset["path"]), root / "assets" / asset["path"])
    for case, crop, ref in cases:
        save_rgb(root / case["image"], crop)
        write_json_new(root / case["references"][0], ref.to_dict())
    write_json_new(root / "study.spec.json", plan["study_spec"])
    write_json_new(root / "conversions.json", plan["conversions"])
    _seal(root, DATASET_FORMAT)
    verify_dataset(root)
    return root


def verify_dataset(path: str | Path) -> dict:
    root = Path(path)
    _verify_seal(root, DATASET_FORMAT)
    recipe = _recipe(root / "recipe.json")
    cases, plan = _plan(recipe, root / "assets")
    if (read_json(root / "study.spec.json") != plan["study_spec"]
            or read_json(root / "conversions.json") != plan["conversions"]):
        raise ObservationError("dataset conversion or split plan differs from pinned sources")
    for case, crop, ref in cases:
        if (not np.array_equal(load_rgb(root / case["image"]), crop)
                or read_json(root / case["references"][0]) != ref.to_dict()):
            raise ObservationError("dataset crop or reference differs from pinned source coordinates")
    return {"verified": True, "kind": "machine_reference_dataset" if recipe['label_kind'] == 'model_generated' else 'published_reference_dataset', "source_images": len(recipe["images"]),
            "cases": len(cases), "eligible": sum(c["eligibility"] == "included" for c, _, _ in cases),
            "reference_kind": recipe['label_kind'], "conversion_recomputed": True,
            "human_accuracy_validated": False}


def run_pipeline(dataset: str | Path, destination: str | Path, *, model: str | Path | None = None,
                 registry: str | Path | None = None) -> dict:
    from .enrollment import read_registry
    from .study_runner import NativeExecutor, run_session
    from .study_io import finalize_session, replay_qualification_bundle
    source, root = Path(dataset).resolve(), Path(destination).resolve()
    verify_dataset(source)
    if root == source or root.is_relative_to(source):
        raise ObservationError("pipeline output must be outside the immutable dataset")
    root.mkdir(parents=True, exist_ok=False)
    journal = Journal()
    try:
        with journal.step("pipeline.retain_dataset", {}) as step:
            shutil.copytree(source, root / "dataset")
            step.update(verify_dataset(root / "dataset"))
        with journal.step("pipeline.freeze_before_inference", {}) as step:
            study = freeze_study(root / "dataset/study.spec.json", root / "study",
                                 enrollment_registry=read_registry(registry) if registry is not None else None)
            step.update(study_sha256=study.digest)
        recipe = _recipe(root / "dataset/recipe.json")
        model_path = Path(model) if model else root / "dataset/assets" / recipe["model_asset"]
        with journal.step("pipeline.native_acquisition", {"automatic_retries": 0}) as step:
            status = run_session(study.root, root / "session", NativeExecutor(model_path,
                                 timeout_seconds=study.data["timeout_seconds"]))
            step.update(outcomes=status["outcomes"], complete=status["complete"])
        with journal.step("pipeline.fit_calibrate_evaluate", {}) as step:
            finalize_session(root / "session", root / "qualification")
            report = read_json(root / "qualification/qualification.json")
            step.update(raw_coordinates_changed=report["raw_coordinates_changed"])
        with journal.step("pipeline.exact_numerical_replay", {}) as step:
            replay = replay_qualification_bundle(root / "qualification", root / "replay")
            step.update(replay_verified=replay["replay_verified"], native_reexecuted=False)
        result = {"completed": True, "reference_kind": recipe['label_kind'], "human_accuracy_validated": False,
            "outcomes": status["outcomes"], "enrolled": status["enrolled"], "eligible": status["eligible"],
            "selected_error_model": report["selected_error_model"], "replay_verified": replay["replay_verified"],
            "environment": environment(), "grouping_limit": recipe["grouping_limit"]}
        write_json_new(root / "pipeline.json", result)
        journal.record("execution.complete", {"replay_verified": True})
        (root / "events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
        _seal(root, PIPELINE_FORMAT)
        verify_pipeline(root)
        return {**result, "pipeline": str(root)}
    except Exception as exc:
        journal.record("execution.failed", {"exception": type(exc).__name__, "detail": str(exc)})
        (root / "failure.events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
        write_json_new(root / "failure.json", {"completed": False, "exception": type(exc).__name__, "detail": str(exc)})
        exc.failure_path = str(root)
        raise


def verify_pipeline(path: str | Path) -> dict:
    from .study_io import verify_qualification_bundle
    from .study import load_study
    from .study_runner import verify_session
    root = Path(path)
    _verify_seal(root, PIPELINE_FORMAT)
    dataset = verify_dataset(root / "dataset")
    verify_qualification_bundle(root / "qualification")
    verify_qualification_bundle(root / "replay")
    study = load_study(root / "study")
    session = verify_session(root / "session")
    bound = [session["study_sha256"], load_study(root / "qualification/acquisition/study").digest,
             load_study(root / "replay/acquisition/study").digest]
    if (any(digest != study.digest for digest in bound)
            or study.data["source_manifest_sha256"] != sha256((root / "dataset/study.spec.json").read_bytes()).hexdigest()):
        raise ObservationError("pipeline stages are not bound to the retained dataset/study")
    for name in ("qualification.json", "localization_calibration.json"):
        if read_json(root / "qualification" / name) != read_json(root / "replay" / name):
            raise ObservationError("pipeline qualification replay differs")
    with np.load(root / "qualification/arrays.npz", allow_pickle=False) as old, np.load(root / "replay/arrays.npz", allow_pickle=False) as new:
        if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
            raise ObservationError("pipeline numerical replay arrays differ")
    result = read_json(root / "pipeline.json")
    report = read_json(root / "qualification/qualification.json")
    if (result.get("completed") is not True or result.get("human_accuracy_validated") is not False
            or result.get("reference_kind") != dataset['reference_kind'] or result.get("replay_verified") is not True
            or result.get("selected_error_model") != report["selected_error_model"]
            or any(result.get(key) != session[key] for key in ("outcomes", "enrolled", "eligible"))):
        raise ObservationError("published-reference pipeline has inconsistent completion/provenance flags")
    verify_journal([json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()])
    return {"verified": True, "kind": "native_machine_reference_pipeline" if dataset['reference_kind'] == 'model_generated' else 'native_published_reference_pipeline', "dataset": dataset,
            "native_reexecuted": False, "human_accuracy_validated": False}
