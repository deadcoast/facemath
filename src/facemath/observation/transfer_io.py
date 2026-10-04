"""Create-only transfer evidence using saved observations and a frozen model."""
from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import shutil
from .publication import publication_stage

import numpy as np

from .contracts import ObservationError, array_digest, object_digest
from .journal import Journal, code_fingerprint, environment, verify_journal
from .numerics import require_numeric_environment
from .study import file_inventory, load_rgb, load_study, read_json, verify_inventory, write_json_new
from .study_runner import verify_session, writer_lock
from .transfer import evaluate_transfer, load_capsule, REPORT_FORMAT
from .validation_io import run_validation_study, verify_validation_bundle

BUNDLE_FORMAT = "facemath.transfer_bundle.v14"
PIPELINE_FORMAT = "facemath.transfer_pipeline.v14"


def finalize_transfer(session_path: str | Path, destination: str | Path) -> Path:
    source, target = Path(session_path).resolve(), Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    if target == source or target.is_relative_to(source):
        raise ObservationError("transfer output must be outside the acquisition session")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = publication_stage(target.parent, ".facemath-transfer-")
    journal = Journal()
    try:
        with writer_lock(source / ".runner.lock"):
            with journal.step("transfer.verify_frozen_acquisition", {}) as step:
                status = verify_session(source)
                study = load_study(source / "study")
                session = read_json(source / "session.json")
                if not status["complete"] or not status["contract_valid"] or "frozen_model" not in study.data:
                    raise ObservationError("transfer needs a complete valid evaluation-only acquisition")
                if session["engine_code_sha256"] != code_fingerprint():
                    raise ObservationError("transfer finalization requires the acquisition's code")
                require_numeric_environment(session["numeric_environment"], "transfer finalization")
                step.update(study_sha256=study.digest, outcomes=status["outcomes"])
            shutil.copytree(source, temp / "acquisition", ignore=lambda directory, names:
                            {".runner.lock"} if Path(directory) == source else set())
            (temp / "acquisition/.runner.lock").write_bytes(b"0")
        study = load_study(temp / "acquisition/study")
        receipts = {r["case_id"]: r for r in status["receipts"]}
        entries = []
        for i, c in enumerate(study.cases):
            if c["eligibility"] != "included":
                continue
            entry = {k: c[k] for k in ("case_id", "group_id", "split", "lineage_ids")}
            entry.update(image="acquisition/study/" + c["image"],
                         references=["acquisition/study/" + n for n in c["references"]])
            receipt = receipts[c["case_id"]]
            if receipt["status"] == "observed":
                entry["observation"] = f"acquisition/cases/case_{i:05d}/observation.json"
            else:
                entry["failure"] = receipt["status"] + ": " + receipt["detail"]
            entries.append(entry)
        write_json_new(temp / "validation_inputs.json", {"format": "facemath.validation_study.v4",
            "domain": study.data["domain"], "minimum_raters": study.data["minimum_raters"], "cases": entries})
        with journal.step("transfer.reuse_raw_evaluator_without_calibration", {"fit_performed": False}) as step:
            raw = run_validation_study(temp / "validation_inputs.json", temp / "raw_validation", alpha=study.data["alpha"])
            step.update(raw_report_sha256=read_json(raw / "validation.json")["content_sha256"])
        with journal.step("transfer.evaluate_unchanged_model", {"fit_performed": False}) as step:
            report = evaluate_transfer(study, read_json(raw / "validation.json"), status)
            step.update(calibration_sha256=report["calibration_sha256"], raw_coordinates_changed=False)
        write_json_new(temp / "transfer.json", report)
        (temp / "localization_calibration.json").write_bytes(study.data["frozen_model"]["model_json"].encode("utf-8"))
        write_json_new(temp / "environment.json", environment())
        journal.record("execution.complete", {"report_sha256": report["content_sha256"]})
        (temp / "events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
        manifest = {"format": BUNDLE_FORMAT, "study_sha256": study.digest, "files": file_inventory(temp),
                    "limit": "saved-model research transfer; no refit, inference here or empirical admission"}
        manifest["content_sha256"] = object_digest(manifest)
        write_json_new(temp / "manifest.json", manifest)
        verify_transfer_bundle(temp)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temp, target)
        return target
    except Exception as exc:
        from .journal import retain_failed_run
        retain_failed_run(temp, target, journal, exc)
        raise


def verify_transfer_bundle(path: str | Path) -> dict:
    root = Path(path).resolve()
    manifest = read_json(root / "manifest.json")
    digest = manifest.pop("content_sha256")
    if manifest.get("format") != BUNDLE_FORMAT or object_digest(manifest) != digest:
        raise ObservationError("transfer manifest mismatch")
    verify_inventory(root, manifest["files"], exclude=("manifest.json",))
    study = load_study(root / "acquisition/study")
    status = verify_session(root / "acquisition")
    if ("frozen_model" not in study.data or not status["complete"] or not status["contract_valid"]
            or study.digest != manifest["study_sha256"]):
        raise ObservationError("transfer acquisition binding mismatch")
    model = load_capsule(study.data["frozen_model"])
    if (root / "localization_calibration.json").read_bytes() != study.data["frozen_model"]["model_json"].encode("utf-8"):
        raise ObservationError("transfer altered the frozen model bytes")
    report = read_json(root / "transfer.json")
    checksum = report.pop("content_sha256")
    if (report.get("format") != REPORT_FORMAT or object_digest(report) != checksum
            or report["study_sha256"] != study.digest or report["calibration_sha256"] != model.digest
            or report["capsule_sha256"] != study.data["frozen_model"]["content_sha256"]
            or report["bias_sha256"] != model.bias.digest or report["scatter_sha256"] != model.scatter.digest
            or report["radius"] != model.quantile or report["target_domain"] != study.data["domain"]
            or report["donor_domain"] != model.residual.domain
            or any(report[k] is not False for k in ("fit_performed", "raw_coordinates_changed", "empirical_admission"))
            or report["outcomes"] != status["outcomes"] or report["enrolled_cases"] != len(study.cases)
            or report["eligible_cases"] != len(study.included)):
        raise ObservationError("transfer report/model binding mismatch")
    raw = root / "raw_validation"
    verify_validation_bundle(raw)
    if (raw / "localization_calibration.json").exists() or (raw / "coverage.json").exists():
        raise ObservationError("transfer raw evaluator must not fit calibration")
    lock = read_json(raw / "study.lock.json")
    if (len(lock["cases"]) != len(study.included) or any(c["split"] != "evaluation" for c in lock["cases"])
            or {c["case_id"] for c in lock["cases"]} != {c["case_id"] for c in study.included}
            or lock["domain"] != study.data["domain"] or lock["minimum_raters"] != study.data["minimum_raters"]):
        raise ObservationError("transfer raw evaluation enrollment mismatch")
    by_id = {c["case_id"]: (i, c) for i, c in enumerate(study.cases)}
    for c in lock["cases"]:
        i, original = by_id[c["case_id"]]
        if original["eligibility"] != "included" or any(c[k] != original[k] for k in ("group_id", "lineage_ids")):
            raise ObservationError("transfer raw case binding mismatch")
        if (array_digest(load_rgb(raw / c["image"])) != original["image_sha256"]
                or [read_json(raw / n) for n in c["references"]] != [r.to_dict() for r in study.references(original)]):
            raise ObservationError("transfer changed source pixels or references")
        receipt = read_json(root / f"acquisition/cases/case_{i:05d}/receipt.json")
        if receipt["status"] == "observed":
            if "observation" not in c or "failure" in c:
                raise ObservationError("transfer lost an observed attempt")
            if read_json(raw / c["observation"]) != read_json(root / f"acquisition/cases/case_{i:05d}/observation.json"):
                raise ObservationError("transfer changed an observation")
        elif c.get("failure") != receipt["status"] + ": " + receipt["detail"] or "observation" in c:
            raise ObservationError("transfer changed a failed attempt")
    verify_journal([json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()])
    return {"verified": True, "kind": "frozen_model_transfer", "files": len(manifest["files"]),
            "calibration_sha256": model.digest, "study_sha256": study.digest,
            "fit_performed": False, "empirical_admission": False, "live_inference_reexecuted": False}


def replay_transfer_bundle(path: str | Path, destination: str | Path) -> dict:
    root = Path(path).resolve()
    verify_transfer_bundle(root)
    env = read_json(root / "environment.json")
    if env["engine_code_sha256"] != code_fingerprint():
        raise ObservationError("transfer replay requires matching source")
    require_numeric_environment(env["numeric_environment"], "transfer replay")
    target = finalize_transfer(root / "acquisition", destination)
    for name in ("transfer.json", "raw_validation/validation.json"):
        if read_json(root / name) != read_json(target / name):
            raise ObservationError("transfer numerical replay differs: " + name)
    with np.load(root / "raw_validation/arrays.npz", allow_pickle=False) as old, np.load(target / "raw_validation/arrays.npz", allow_pickle=False) as new:
        if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
            raise ObservationError("transfer raw numerical array replay differs")
    return {"replay_verified": True, "bundle": str(target), "comparison": "exact transfer/raw reports, frozen model bytes and raw arrays",
            "fit_performed": False, "live_inference_reexecuted": False}


def run_transfer_pipeline(dataset: str | Path, capsule_path: str | Path, destination: str | Path,
                          *, model: str | Path | None = None, registry: str | Path | None = None) -> dict:
    from .enrollment import read_registry
    from .dataset import verify_dataset, _seal
    from .study import freeze_study
    from .study_runner import NativeExecutor, run_session
    source, root = Path(dataset).resolve(), Path(destination).resolve()
    verify_dataset(source)
    capsule = read_json(Path(capsule_path))
    load_capsule(capsule)
    if root == source or root.is_relative_to(source):
        raise ObservationError("transfer pipeline output must be outside the immutable dataset")
    root.mkdir(parents=True, exist_ok=False)
    journal = Journal()
    try:
        with journal.step("transfer_pipeline.freeze_before_inference", {}) as step:
            shutil.copytree(source, root / "dataset")
            write_json_new(root / "frozen_model.json", capsule)
            study = freeze_study(root / "dataset/study.spec.json", root / "study", frozen_model=capsule,
                                 enrollment_registry=read_registry(registry) if registry is not None else None)
            step.update(study_sha256=study.digest, capsule_sha256=capsule["content_sha256"])
        recipe = read_json(root / "dataset/recipe.json")
        model_path = Path(model) if model else root / "dataset/assets" / recipe["model_asset"]
        with journal.step("transfer_pipeline.native_acquisition", {"automatic_retries": 0}) as step:
            status = run_session(study.root, root / "session", NativeExecutor(model_path,
                                 timeout_seconds=study.data["timeout_seconds"]))
            step.update(outcomes=status["outcomes"], complete=status["complete"])
        with journal.step("transfer_pipeline.evaluate_fixed_model", {"fit_performed": False}) as step:
            finalize_transfer(root / "session", root / "evaluation")
            report = read_json(root / "evaluation/transfer.json")
            step.update(calibration_sha256=report["calibration_sha256"])
        with journal.step("transfer_pipeline.exact_replay", {}) as step:
            result = replay_transfer_bundle(root / "evaluation", root / "replay")
            step.update(replay_verified=result["replay_verified"])
        result = {"completed": True, "fit_performed": False, "empirical_admission": False,
            "outcomes": status["outcomes"], "enrolled": status["enrolled"], "eligible": status["eligible"],
            "calibration_sha256": report["calibration_sha256"], "replay_verified": True,
            "reference_kind": recipe["label_kind"], "grouping_limit": recipe["grouping_limit"], "environment": environment()}
        write_json_new(root / "pipeline.json", result)
        journal.record("execution.complete", {"replay_verified": True})
        (root / "events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
        _seal(root, PIPELINE_FORMAT)
        verify_transfer_pipeline(root)
        return {**result, "pipeline": str(root)}
    except Exception as exc:
        journal.record("execution.failed", {"exception": type(exc).__name__, "detail": str(exc)})
        (root / "failure.events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
        write_json_new(root / "failure.json", {"completed": False, "exception": type(exc).__name__, "detail": str(exc)})
        exc.failure_path = str(root)
        raise


def verify_transfer_pipeline(path: str | Path) -> dict:
    from .dataset import verify_dataset, _verify_seal
    root = Path(path)
    _verify_seal(root, PIPELINE_FORMAT)
    dataset = verify_dataset(root / "dataset")
    study = load_study(root / "study")
    status = verify_session(root / "session")
    capsule = read_json(root / "frozen_model.json")
    model = load_capsule(capsule)
    for name in ("evaluation", "replay"):
        result = verify_transfer_bundle(root / name)
        copied = verify_session(root / name / "acquisition")
        if result["study_sha256"] != study.digest or copied["receipts"] != status["receipts"]:
            raise ObservationError("transfer pipeline study mismatch")
    if (capsule != study.data["frozen_model"] or status["study_sha256"] != study.digest
            or study.data["source_manifest_sha256"] != sha256((root / "dataset/study.spec.json").read_bytes()).hexdigest()
            or read_json(root / "evaluation/transfer.json") != read_json(root / "replay/transfer.json")
            or read_json(root / "evaluation/raw_validation/validation.json") != read_json(root / "replay/raw_validation/validation.json")):
        raise ObservationError("transfer pipeline replay or input binding mismatch")
    with np.load(root / "evaluation/raw_validation/arrays.npz", allow_pickle=False) as old, np.load(root / "replay/raw_validation/arrays.npz", allow_pickle=False) as new:
        if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
            raise ObservationError("transfer pipeline replay arrays differ")
    result = read_json(root / "pipeline.json")
    if (result.get("completed") is not True or result.get("replay_verified") is not True
            or result.get("fit_performed") is not False or result.get("empirical_admission") is not False
            or result["calibration_sha256"] != model.digest or result["reference_kind"] != dataset["reference_kind"]
            or any(result.get(k) != status[k] for k in ("outcomes", "enrolled", "eligible"))):
        raise ObservationError("transfer pipeline completion flags differ from evidence")
    verify_journal([json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()])
    return {"verified": True, "kind": "native_frozen_model_transfer", "calibration_sha256": model.digest,
            "dataset": dataset, "fit_performed": False, "empirical_admission": False}
