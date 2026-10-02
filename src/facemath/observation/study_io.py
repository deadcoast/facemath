"""Acquisition → existing validation engine → qualified error model → pair API.

This is the v5 integration boundary. The v4 source-coordinate evaluator remains
the sole raw error implementation. New qualification consumes its retained rows,
not a second reimplementation with subtly different denominators or geometry.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import shutil
import tempfile

import numpy as np

from .contracts import ObservationError, array_digest, object_digest
from .journal import Journal, code_fingerprint, environment, verify_journal
from .numerics import require_numeric_environment
from .qualification import qualify
from .study import file_inventory, load_study, read_json, verify_inventory, write_json_new
from .study_report import render_study_report
from .study_runner import verify_session, writer_lock
from .validation_io import run_validation_study, verify_validation_bundle


def finalize_session(session_path: str | Path, destination: str | Path) -> Path:
    source, target = Path(session_path).resolve(), Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    if target == source or target.is_relative_to(source):
        raise ObservationError("qualification output must be outside the acquisition session")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".facemath-qualify-", dir=target.parent))
    journal = Journal()
    try:
        with writer_lock(source / ".runner.lock"):
            with journal.step("study.verify_completed_acquisition", {}) as step:
                status = verify_session(source)
                if not status["complete"] or not status["contract_valid"]:
                    raise ObservationError("cannot qualify incomplete or contract-failed acquisition")
                study = load_study(source / "study")
                spec = study.data
                session = read_json(source / "session.json")
                if session["engine_code_sha256"] != code_fingerprint():
                    raise ObservationError("qualification requires the acquisition's code")
                require_numeric_environment(session["numeric_environment"], "qualification")
                step.update(study_sha256=study.digest, committed=status["committed"],
                            execution_kind=status["execution_kind"], no_inference_performed_here=True)
            # The Windows byte-range lock cannot be read by copytree. It is
            # process coordination, not observation evidence. Keep the source
            # locked throughout copying, and retain a canonical sentinel so
            # later replay cannot add a file to the sealed inventory.
            shutil.copytree(source, temp / "acquisition", ignore=lambda directory, names:
                            {".runner.lock"} if Path(directory) == source else set())
            (temp / "acquisition" / ".runner.lock").write_bytes(b"0")
        study = load_study(temp / "acquisition" / "study")
        receipts = {r["case_id"]: r for r in status["receipts"]}
        entries = []
        for i, c in enumerate(study.cases):
            if c["eligibility"] != "included":
                continue
            receipt = receipts[c["case_id"]]
            entry = {k: c[k] for k in ("case_id", "group_id", "split", "lineage_ids")}
            entry.update(image="acquisition/study/"+c["image"],
                         references=["acquisition/study/"+n for n in c["references"]])
            if receipt["status"] == "observed":
                entry["observation"] = f"acquisition/cases/case_{i:05d}/observation.json"
            else:
                entry["failure"] = receipt["status"]+": "+receipt["detail"]
            entries.append(entry)
        import_spec = {"format": "facemath.validation_study.v4", "domain": spec["domain"],
                       "minimum_raters": spec["minimum_raters"], "cases": entries}
        write_json_new(temp / "validation_inputs.json", import_spec)
        with journal.step("study.reuse_source_coordinate_evaluator", {"predictions_are_frozen": True}) as step:
            validation = run_validation_study(temp / "validation_inputs.json", temp / "raw_validation",
                                              alpha=spec["alpha"])
            raw = read_json(validation / "validation.json")
            step.update(raw_report_sha256=raw["content_sha256"], attempted_cases=len(raw["cases"]))
        with journal.step("study.frozen_bias_and_heldout_qualification", {"bias_mode": spec["bias_mode"]}) as step:
            report, arrays, calibration = qualify(study, raw, status)
            step.update(report_sha256=report["content_sha256"], calibration_sha256=calibration.digest,
                        finite_calibration=(any(r.quantile is not None for r in calibration.radii) if hasattr(calibration, "radii") else calibration.quantile is not None),
                        radius_mode="per_acquisition_category" if hasattr(calibration, "radii") else "pooled",
                        raw_coordinates_changed=False)
        write_json_new(temp / "qualification.json", report)
        write_json_new(temp / "localization_calibration.json", calibration.to_dict())
        np.savez_compressed(temp / "arrays.npz", **arrays)
        write_json_new(temp / "environment.json", environment())
        from .readout import admission_panel
        html = render_study_report(report).replace("</header>", "</header>" + admission_panel(report.get("empirical_admission")), 1)
        (temp / "report.html").write_text(html, encoding="utf-8")
        journal.record("execution.complete", {"qualification_sha256": report["content_sha256"]})
        (temp / "events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
        manifest = {"format": "facemath.qualification_bundle.v5", "study_sha256": study.digest,
            "files": file_inventory(temp),
            "arrays": {k: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": array_digest(a)}
                       for k, a in arrays.items()},
            "limit": "internal consistency, not externally attested execution/chronology or certified deployment accuracy"}
        manifest["content_sha256"] = object_digest(manifest)
        write_json_new(temp / "manifest.json", manifest)
        verify_qualification_bundle(temp)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temp, target)
        return target
    except Exception as exc:
        from .journal import retain_failed_run
        retain_failed_run(temp, target, journal, exc)
        raise


def verify_qualification_bundle(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest = read_json(root / "manifest.json")
    expected = manifest.pop("content_sha256")
    if manifest.get("format") != "facemath.qualification_bundle.v5" or object_digest(manifest) != expected:
        raise ObservationError("qualification manifest mismatch")
    verify_inventory(root, manifest["files"], exclude=("manifest.json",))
    with np.load(root / "arrays.npz", allow_pickle=False) as data:
        if set(data.files) != set(manifest["arrays"]):
            raise ObservationError("qualification array inventory mismatch")
        for name, spec in manifest["arrays"].items():
            array = data[name]
            if list(array.shape) != spec["shape"] or array.dtype.str != spec["dtype"] or array_digest(array) != spec["sha256"]:
                raise ObservationError(f"qualification array mismatch: {name}")
    report = read_json(root / "qualification.json")
    checksum = report.pop("content_sha256")
    if object_digest(report) != checksum or report["study_sha256"] != manifest["study_sha256"]:
        raise ObservationError("qualification report binding mismatch")
    status = verify_session(root / "acquisition")
    if not status["complete"] or not status["contract_valid"] or status["study_sha256"] != manifest["study_sha256"]:
        raise ObservationError("qualification source acquisition is not valid and complete")
    verify_validation_bundle(root / "raw_validation")
    from .bias import load_localization_calibration
    calibration = load_localization_calibration(read_json(root / "localization_calibration.json"))
    if calibration.digest != report["selected_error_model"]["calibration_sha256"]:
        raise ObservationError("qualification/calibration binding mismatch")
    verify_journal([json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()])
    return {"verified": True, "kind": "observer_qualification", "files": len(manifest["files"]),
            "arrays": len(manifest["arrays"]), "study_sha256": manifest["study_sha256"],
            "manifest_content_sha256": expected, "live_inference_reexecuted": False}


def replay_qualification_bundle(path: str | Path, destination: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    verify_qualification_bundle(root)
    env = read_json(root / "environment.json")
    if env["engine_code_sha256"] != code_fingerprint():
        raise ObservationError("qualification replay requires matching source")
    require_numeric_environment(env["numeric_environment"], "qualification replay")
    target = finalize_session(root / "acquisition", destination)
    for name in ("qualification.json", "localization_calibration.json"):
        if read_json(root / name) != read_json(target / name):
            raise ObservationError(f"qualification numerical replay differs: {name}")
    with np.load(root / "arrays.npz", allow_pickle=False) as old, np.load(target / "arrays.npz", allow_pickle=False) as new:
        if set(old.files) != set(new.files) or any(not np.array_equal(old[k], new[k]) for k in old.files):
            raise ObservationError("qualification numerical array replay differs")
    return {"replay_verified": True, "bundle": str(target), "comparison": "exact qualification/calibration JSON and numerical array equality",
            "live_inference_reexecuted": False}
