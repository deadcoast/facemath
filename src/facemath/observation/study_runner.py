"""Resumable single-attempt acquisition on a frozen study.

Receipts are committed per case. Resume never re-runs an already started case:
interrupted attempts become explicit unavailable observations. Contract failures
stop the session and prohibit scientific finalization. The model worker receives
an image and a model declaration, never labels, split names or strata.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Protocol
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from .contracts import Observation, ObservationError, ProviderUnavailable, object_digest
from .journal import code_fingerprint, numeric_environment
from .provider import DetectionMiss
from .study import (
    FrozenStudy, ObserverDeclaration, file_inventory, load_study, read_json,
    write_json_new, verify_inventory, load_rgb,
)

TERMINAL = {"observed", "no_face", "multiple_faces", "timeout", "interrupted", "runtime_error", "contract_error"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def writer_lock(path: Path) -> Iterator[None]:
    """OS-released interprocess lock; stale lock FILES do not imply active locks."""
    if path.is_symlink():
        raise ObservationError("session lock cannot be a symlink")
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ObservationError("another writer owns this acquisition session") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ObservationError("another writer owns this acquisition session") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class AttemptOutcome:
    status: str
    observation: Observation | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in TERMINAL or ((self.status == "observed") != (self.observation is not None)):
            raise ObservationError("inconsistent observation attempt result")
        if self.status != "observed" and not self.detail:
            raise ObservationError("unsuccessful attempts require a reason")


class ObserverExecutor(Protocol):
    def preflight(self, declaration: ObserverDeclaration) -> dict[str, Any]: ...
    def execute(self, image_path: Path, declaration: ObserverDeclaration,
                directory: Path) -> AttemptOutcome: ...


class NativeExecutor:
    """One bounded native Python process per image; stdout/stderr are retained.

    This is process isolation, NOT a network sandbox. Upstream MediaPipe's
    published metrics behavior must be assessed in the deployment environment.
    """
    def __init__(self, model: str | Path, *, timeout_seconds: float = 90.) -> None:
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ObservationError("positive finite native timeout required")
        self.model = Path(model).resolve()
        self.timeout = float(timeout_seconds)

    def preflight(self, declaration: ObserverDeclaration) -> dict[str, Any]:
        if declaration.execution_kind != "native_mediapipe":
            raise ProviderUnavailable("native execution cannot impersonate a synthetic study")
        if not self.model.is_file():
            raise ProviderUnavailable("explicit local MediaPipe model is missing")
        actual = sha256(self.model.read_bytes()).hexdigest()
        if actual != declaration.model_sha256:
            raise ProviderUnavailable("local model digest differs from the frozen declaration")
        try:
            version = importlib.metadata.version("mediapipe")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProviderUnavailable("MediaPipe is not installed in this interpreter") from exc
        if version != declaration.runtime_version:
            raise ProviderUnavailable("MediaPipe runtime differs from the frozen declaration")
        return {"ready": True, "kind": "native_mediapipe", "model_sha256": actual,
                "runtime_version": version, "python_executable": sys.executable,
                "timeout_seconds": self.timeout, "inference_executed": False,
                "network_isolation": "not supplied by this process wrapper"}

    def execute(self, image_path: Path, declaration: ObserverDeclaration, directory: Path) -> AttemptOutcome:
        request = {"format": "facemath.worker_request.v5", "image": str(image_path.resolve()),
                   "model": str(self.model), "observer": declaration.to_dict(),
                   "engine_code_sha256": code_fingerprint(), "numeric_environment": numeric_environment()}
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        write_json_new(request_path, request)
        command = [sys.executable, "-m", "facemath.observation.worker", str(request_path), str(response_path)]
        write_json_new(directory / "invocation.json", {"argv": command, "shell": False,
            "timeout_seconds": self.timeout, "started_at": utc_now()})
        with (directory / "stdout.txt").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
            try:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired:
                write_json_new(directory / "process.json", {"timed_out": True, "returncode": None, "completed_at": utc_now()})
                return AttemptOutcome("timeout", detail="native worker exceeded frozen session timeout; not automatically retried")
        write_json_new(directory / "process.json", {"timed_out": False, "returncode": result.returncode, "completed_at": utc_now()})
        if not response_path.exists():
            return AttemptOutcome("runtime_error", detail=f"native worker exit={result.returncode}; no response artifact")
        try:
            response = read_json(response_path)
            if response.get("format") != "facemath.worker_response.v5":
                raise ObservationError("invalid native response format")
            obs = Observation.from_dict(response["observation"]) if response.get("observation") else None
            if result.returncode != 0 and response["status"] == "observed":
                raise ObservationError("worker reported success after nonzero exit")
            return AttemptOutcome(response["status"], obs, response.get("detail", ""))
        except (KeyError, TypeError, ValueError) as exc:
            return AttemptOutcome("contract_error", detail=f"invalid worker response: {type(exc).__name__}: {exc}")


def _commit_receipt(directory: Path, start: dict[str, Any], outcome: AttemptOutcome,
                    elapsed_seconds: float | None) -> dict[str, Any]:
    if outcome.observation is not None:
        write_json_new(directory / "observation.json", outcome.observation.to_dict())
    receipt = {"format": "facemath.attempt_receipt.v5", "case_id": start["case_id"],
        "study_sha256": start["study_sha256"], "start_sha256": object_digest(start),
        "execution_kind": start["execution_kind"], "status": outcome.status, "detail": outcome.detail,
        "completed_at": utc_now(), "elapsed_seconds": elapsed_seconds,
        "observation_sha256": outcome.observation.digest if outcome.observation else None,
        "files": file_inventory(directory), "automatic_retries": 0}
    receipt["content_sha256"] = object_digest(receipt)
    # A partial receipt may not be interpreted as a completed attempt. Rename a
    # fully flushed file under the single-writer lock; never overwrite a receipt.
    temp = directory / ".receipt.pending"
    write_json_new(temp, receipt)
    if (directory / "receipt.json").exists():
        raise FileExistsError(directory / "receipt.json")
    os.rename(temp, directory / "receipt.json")
    return receipt


def verify_receipt(directory: Path, *, study: FrozenStudy, case: MappingLike) -> dict[str, Any]:
    if directory.is_symlink():
        raise ObservationError("case directory cannot be a symlink")
    receipt = read_json(directory / "receipt.json")
    expected = receipt.pop("content_sha256")
    if receipt.get("format") != "facemath.attempt_receipt.v5" or object_digest(receipt) != expected:
        raise ObservationError("attempt receipt hash mismatch")
    if receipt["study_sha256"] != study.digest or receipt["case_id"] != case["case_id"]:
        raise ObservationError("receipt belongs to a different study/case")
    if receipt["status"] not in TERMINAL or receipt["execution_kind"] != study.observer.execution_kind:
        raise ObservationError("receipt violates frozen execution kind")
    verify_inventory(directory, receipt["files"], exclude=("receipt.json",))
    start = read_json(directory / "start.json")
    if (object_digest(start) != receipt["start_sha256"] or start["image_sha256"] != case["image_sha256"]
            or start["study_sha256"] != study.digest or start["case_id"] != case["case_id"]):
        raise ObservationError("attempt-start binding mismatch")
    obs_path = directory / "observation.json"
    if receipt["status"] == "observed":
        obs = Observation.from_dict(read_json(obs_path))
        study.observer.require_observation(obs, study.image(case))
        if obs.digest != receipt["observation_sha256"]:
            raise ObservationError("receipt observation mismatch")
    elif obs_path.exists() or receipt["observation_sha256"] is not None:
        raise ObservationError("failed attempt cannot contain a successful observation")
    receipt["content_sha256"] = expected
    return receipt


# A type alias keeps native worker requests independent from the richer study case.
MappingLike = dict[str, Any]


def verify_session(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    study = load_study(root / "study")
    session = read_json(root / "session.json")
    checksum = session.pop("content_sha256")
    if (session.get("format") != "facemath.acquisition_session.v5" or object_digest(session) != checksum
            or session["study_sha256"] != study.digest):
        raise ObservationError("acquisition session binding mismatch")
    allowed = {"study", "session.json", "cases", ".runner.lock", "preflight"}
    if set(p.name for p in root.iterdir()) - allowed:
        raise ObservationError("unlisted acquisition session artifact")
    expected_names = {f"case_{i:05d}": c for i, c in enumerate(study.cases) if c["eligibility"] == "included"}
    if any(p.name not in expected_names for p in (root / "cases").iterdir()):
        raise ObservationError("case directory is not in frozen enrollment")
    receipts, pending, interrupted = [], [], []
    for name, case in expected_names.items():
        directory = root / "cases" / name
        if not directory.exists():
            pending.append(case["case_id"])
        elif (directory / "receipt.json").exists():
            receipts.append(verify_receipt(directory, study=study, case=case))
        else:
            if directory.is_symlink():
                raise ObservationError("symlinked interrupted attempt")
            # A crash may occur after mkdir but before the start record is flushed.
            interrupted.append(case["case_id"])
    counts = {k: sum(r["status"] == k for r in receipts) for k in sorted(TERMINAL)}
    return {"verified": True, "study_sha256": study.digest, "execution_kind": study.observer.execution_kind,
        "enrolled": len(study.cases), "eligible": len(study.included),
        "prospectively_excluded": len(study.cases)-len(study.included), "committed": len(receipts),
        "pending": pending, "interrupted_uncommitted": interrupted, "outcomes": counts,
        "complete": not pending and not interrupted, "contract_valid": counts["contract_error"] == 0,
        "receipts": receipts, "limit": "byte and contract consistency; not external attestation of execution"}


def run_session(study_path: str | Path, destination: str | Path, executor: ObserverExecutor, *,
                maximum_new_cases: int | None = None) -> dict[str, Any]:
    """Idempotent resume of an existing lock-bound session; one attempt per case."""
    if Path(destination).is_symlink():
        raise ObservationError("session destination cannot be a symlink")
    study, root = load_study(study_path), Path(destination).resolve()
    if maximum_new_cases is not None and (type(maximum_new_cases) is not int or maximum_new_cases < 1):
        raise ObservationError("maximum_new_cases must be a positive integer")
    if study.data["engine_code_sha256"] != code_fingerprint():
        raise ObservationError("study code differs; freeze a new study rather than mutating its protocol")
    if not root.exists():
        root.mkdir(parents=True)
        # Initial copying is guarded by the same writer lock as collection.
    if root.is_symlink():
        raise ObservationError("session destination cannot be a symlink")
    with writer_lock(root / ".runner.lock"):
        if not (root / "session.json").exists():
            if set(p.name for p in root.iterdir()) != {".runner.lock"}:
                raise ObservationError("incomplete session initialization; preserve it and use a new output path")
            shutil.copytree(study.root, root / "study")
            (root / "cases").mkdir(); (root / "preflight").mkdir()
            session = {"format": "facemath.acquisition_session.v5", "study_sha256": study.digest,
                "engine_code_sha256": code_fingerprint(), "numeric_environment": numeric_environment(),
                "created_at": utc_now(), "attempt_policy": "one_observer_attempt_per_enrolled_case_no_hidden_retry"}
            session["content_sha256"] = object_digest(session)
            write_json_new(root / "session.json", session)
        status = verify_session(root)
        if status["study_sha256"] != study.digest:
            raise ObservationError("resume study differs from frozen session")
        session = read_json(root / "session.json")
        if session["engine_code_sha256"] != code_fingerprint() or session["numeric_environment"] != numeric_environment():
            raise ObservationError("resume requires the same code and numerical runtime")
        if not status["contract_valid"]:
            raise ObservationError("contract-failed session cannot resume or qualify")
        if status["complete"]:
            return status
        preflight_path = root / "preflight" / f"check_{len(list((root/'preflight').iterdir())):05d}.json"
        try:
            ready = executor.preflight(study.observer)
            if study.observer.execution_kind == "native_mediapipe" and ready.get("timeout_seconds") != study.data["timeout_seconds"]:
                raise ObservationError("native timeout differs from frozen acquisition protocol")
            if ready.get("ready") is not True or ready.get("kind") != study.observer.execution_kind:
                raise ProviderUnavailable("executor preflight did not establish the frozen execution kind")
            write_json_new(preflight_path, {"at": utc_now(), "ready": True, "result": ready})
        except Exception as exc:
            write_json_new(preflight_path, {"at": utc_now(), "ready": False,
                           "error_type": type(exc).__name__, "error": str(exc), "inference_executed": False})
            raise
        acquired = 0
        for i, case in enumerate(study.cases):
            if case["eligibility"] != "included":
                continue
            directory = root / "cases" / f"case_{i:05d}"
            if (directory / "receipt.json").exists():
                continue
            if directory.exists():
                # No evidence of a completed result: preserve all artifacts and
                # count an interrupted output. Never retry or guess a result.
                start_path = directory / "start.json"
                if start_path.exists():
                    start = read_json(start_path)
                else:
                    start = {"case_id": case["case_id"], "study_sha256": study.digest,
                        "image_sha256": case["image_sha256"], "execution_kind": study.observer.execution_kind,
                        "started_at": None, "missing_start_record": True}
                    write_json_new(start_path, start)
                if (directory / ".receipt.pending").exists():
                    # The pending receipt is retained as a partial commit artifact;
                    # no successful observation is promoted without verification.
                    os.rename(directory / ".receipt.pending", directory / "abandoned_receipt.json")
                if (directory / "observation.json").exists():
                    os.rename(directory / "observation.json", directory / "uncommitted_observation.json")
                _commit_receipt(directory, start, AttemptOutcome("interrupted", detail="prior attempt did not commit; not automatically retried"), None)
                continue
            if maximum_new_cases is not None and acquired >= maximum_new_cases:
                break
            directory.mkdir()
            start = {"case_id": case["case_id"], "study_sha256": study.digest,
                "image_sha256": case["image_sha256"], "execution_kind": study.observer.execution_kind,
                "started_at": utc_now(), "pid": os.getpid(), "engine_code_sha256": code_fingerprint()}
            write_json_new(directory / "start.json", start)
            begin = time.perf_counter()
            try:
                image_path = root / "study" / case["image"]
                outcome = executor.execute(image_path, study.observer, directory)
                if not isinstance(outcome, AttemptOutcome):
                    raise ObservationError("executor returned an invalid outcome type")
                if outcome.observation is not None:
                    study.observer.require_observation(outcome.observation, study.image(case))
            except DetectionMiss as exc:
                outcome = AttemptOutcome(exc.kind, detail=str(exc))
            except ObservationError as exc:
                outcome = AttemptOutcome("contract_error", detail=str(exc))
            except Exception as exc:
                (directory / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
                outcome = AttemptOutcome("runtime_error", detail=f"{type(exc).__name__}: {exc}")
            _commit_receipt(directory, start, outcome, time.perf_counter()-begin)
            acquired += 1
            if outcome.status == "contract_error":
                raise ObservationError("provider contract failed; receipt retained and remaining cases not attempted")
        return verify_session(root)
