"""An executable native smoke check, not a fabricated accuracy study.

Uses the existing pinned NativeExecutor/worker and keeps their actual process
artifacts. Preflight failures are NOT detection attempts. No implicit installs,
model downloads, or network-isolation assertions are made.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import importlib.metadata

from .contracts import ObservationError, ProviderUnavailable, array_digest, object_digest
from .journal import Journal, environment
from .study import ObserverDeclaration, file_inventory, load_rgb, save_rgb, write_json_new, read_json, verify_inventory
from .study_runner import NativeExecutor


def native_smoke(image_path: str | Path, model_path: str | Path, destination: str | Path,
                 *, timeout_seconds: float = 90., expected_sha256: str | None = None) -> dict:
    root = Path(destination)
    if root.exists():
        raise FileExistsError(root)
    image = load_rgb(image_path)
    executor = NativeExecutor(model_path, timeout_seconds=timeout_seconds)
    root.mkdir(parents=True)
    save_rgb(root / "source.png", image)
    journal = Journal()
    model = Path(model_path)
    report = {"format": "facemath.native_smoke.v10", "source_sha256": array_digest(image),
        "status": "not_attempted", "observer_attempted": False, "native_observation_produced": False,
        "human_accuracy_validated": False, "blockers": [], "environment": environment(),
        "meaning": "execution smoke check only; independent labels and group-disjoint study still required",
        "network_isolation": "not supplied; third-party runtime may emit utilization metrics"}
    with journal.step("native_smoke.preflight", {"downloads_allowed": False, "timeout_seconds": timeout_seconds}) as step:
        try:
            version = importlib.metadata.version("mediapipe")
        except importlib.metadata.PackageNotFoundError:
            version = None
            report["blockers"].append("MediaPipe is not installed in this interpreter")
        model_hash = sha256(model.read_bytes()).hexdigest() if model.is_file() else None
        if model_hash is None:
            report["blockers"].append("explicit local model file is missing")
        if expected_sha256 is not None and model_hash != expected_sha256:
            report["blockers"].append("model digest differs from requested pinned digest")
        report.update(runtime_version=version, model_sha256=model_hash)
        if not report["blockers"]:
            declaration = ObserverDeclaration.mediapipe(model_hash, version)
            try:
                step.update(executor.preflight(declaration))
            except ProviderUnavailable as exc:
                report["blockers"].append(str(exc))
        step.update(blockers=report["blockers"], inference_executed=False)
    if not report["blockers"]:
        directory = root / "attempt"; directory.mkdir()
        with journal.step("native_smoke.execute_existing_worker", {"receives_references": False}) as step:
            report["observer_attempted"] = True
            outcome = executor.execute(root / "source.png", declaration, directory)
            report.update(status=outcome.status, detail=outcome.detail,
                          native_observation_produced=outcome.status == "observed")
            if outcome.observation is not None:
                declaration.require_observation(outcome.observation, image)
                write_json_new(root / "observation.json", outcome.observation.to_dict())
                report.update(landmarks=len(outcome.observation.xy), samples=len(outcome.observation.samples),
                              observation_sha256=outcome.observation.digest)
            step.update(status=outcome.status, observation_produced=outcome.status == "observed")
    journal.record("execution.complete", {"status": report["status"], "accuracy_claim": False})
    (root / "events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
    write_json_new(root / "smoke.json", report)
    manifest = {"format": "facemath.native_smoke_bundle.v10", "files": file_inventory(root)}
    manifest["content_sha256"] = object_digest(manifest)
    write_json_new(root / "manifest.json", manifest)
    return report


def verify_smoke(root: str | Path) -> dict:
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    digest = manifest.pop("content_sha256")
    if manifest["format"] != "facemath.native_smoke_bundle.v10" or object_digest(manifest) != digest:
        raise ObservationError("native smoke manifest mismatch")
    verify_inventory(root, manifest["files"], exclude=("manifest.json",))
    report = read_json(root / "smoke.json")
    if array_digest(load_rgb(root / "source.png")) != report["source_sha256"]:
        raise ObservationError("native smoke source mismatch")
    from .journal import verify_journal
    import json
    verify_journal([json.loads(s) for s in (root / "events.jsonl").read_text().splitlines()])
    if report["human_accuracy_validated"] is not False:
        raise ObservationError("native smoke cannot assert reference accuracy validation")
    attempted = report["observer_attempted"]
    if type(attempted) is not bool or type(report["native_observation_produced"]) is not bool:
        raise ObservationError("native smoke execution flags must be booleans")
    allowed = {"observed","no_face","multiple_faces","timeout","interrupted","runtime_error","contract_error"}
    if (not attempted and (report["status"] != "not_attempted" or not report["blockers"] or (root/"attempt").exists())
            or attempted and (report["status"] not in allowed or report["blockers"])):
        raise ObservationError("native smoke outcome contradicts preflight/attempt accounting")
    if report["native_observation_produced"] != (attempted and report["status"] == "observed"):
        raise ObservationError("native smoke observation flag contradicts terminal outcome")
    if (root/"observation.json").is_file() != report["native_observation_produced"]:
        raise ObservationError("native smoke observation artifact contradicts terminal outcome")
    if report["native_observation_produced"]:
        from .contracts import Observation
        observation = Observation.from_dict(read_json(root / "observation.json"))
        observation.require_image(load_rgb(root / "source.png"))
        if observation.digest != report["observation_sha256"]:
            raise ObservationError("native smoke observation mismatch")
    return {"verified": True, "kind": "native_smoke", "status": report["status"],
            "observer_reexecuted": False, "accuracy_validated": False}
