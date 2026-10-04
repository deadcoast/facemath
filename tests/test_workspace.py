"""Application lifecycle and request boundaries; native inference is exercised by browser QA."""
import base64
import json
from pathlib import Path
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from facemath.cli import build_parser
from facemath.observation.contracts import ObservationError
from facemath.workspace import Workspace, WorkspaceServer


@pytest.fixture
def work(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"explicit test fixture, not a native model")
    instance = Workspace(tmp_path / "comparisons", model)
    yield instance
    instance.close()


def image(name="portrait.png"):
    ok, raw = cv2.imencode(".png", np.full((24, 32, 3), 100, np.uint8))
    assert ok
    return {"name": name, "data": base64.b64encode(raw.tobytes()).decode()}


class Pending:
    def __init__(self):
        self.event = threading.Event()
        self.returncode = None

    def wait(self, timeout=None):
        if not self.event.wait(timeout):
            raise subprocess.TimeoutExpired("fixture", timeout)
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.event.set()

    kill = terminate


def test_cli_preserves_annotation_workspace_and_adds_app():
    parser = build_parser()
    assert parser.parse_args(["app", "--model", "model.task"]).command == "app"
    assert "workspace" in parser._subparsers._group_actions[0].choices


@pytest.mark.parametrize("item", [None, {}, {"name": "a", "data": "bad base64"},
                                      {"name": "a", "data": base64.b64encode(b"not an image").decode()}])
def test_bad_inputs_do_not_create_runs(work, item):
    with pytest.raises((ValueError, TypeError)):
        work.submit({"a": item, "b": image()})
    assert work.history() == []


@pytest.mark.parametrize("tolerance", [True, 0, -1, 2, float("nan"), "0.05"])
def test_invalid_tolerance_does_not_launch(work, tolerance):
    with pytest.raises(ValueError):
        work.submit({"a": image(), "b": image(), "tolerance": tolerance})
    assert work.history() == []


def test_worker_receives_full_existing_instrument_and_retains_input_bytes(work, monkeypatch):
    process = Pending()
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return process
    monkeypatch.setattr(subprocess, "Popen", launch)
    record = work.submit({"a": image("../../a.png"), "b": image()})
    command = calls[0]
    assert command[1:4] == ["-m", "facemath", "analyze"]
    assert not any(x in command for x in ("--no-flow", "--no-edges", "--no-intrinsic", "--draws"))
    assert command[command.index("--model-sha256") + 1] == work.model_hash
    directory = Path(record["directory"])
    assert (directory / "input_a.image").read_bytes() == base64.b64decode(image()["data"])
    assert record["inputs"][0]["name"] == "a.png"
    assert cv2.imread(str(directory / "preview_a.png")).shape == (24, 32, 3)
    with pytest.raises(BlockingIOError):
        work.submit({"a": image(), "b": image()})
    work.cancel(record["id"])
    work.thread.join(3)
    assert work.status(record["id"])["status"] == "cancelled"
    assert work.status(record["id"])["returncode"] == -15
    assert (directory / "input_a.image").exists()
    assert not (directory / "comparison.zip").exists()


def test_failed_launch_can_retry_without_replacing_run(work, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("explicit failed launch")
    monkeypatch.setattr(subprocess, "Popen", fail)
    first = work.submit({"a": image(), "b": image()})
    second = work.submit({"a": image(), "b": image()})
    assert first["status"] == second["status"] == "failed"
    assert first["id"] != second["id"]
    assert len(work.history()) == 2
    assert "explicit failed launch" in first["message"]


def test_nonzero_worker_does_not_publish_empty_results(work, monkeypatch):
    process = Pending()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: process)
    record = work.submit({"a": image(), "b": image()})
    (Path(record["directory"]) / "stderr.log").write_text("FACEMATH: no face detected\n")
    process.returncode = 2
    process.event.set()
    work.thread.join(3)
    result = work.status(record["id"])
    assert result["status"] == "failed" and result["returncode"] == 2
    assert result["message"] == "no face detected"


def test_second_app_cannot_interrupt_active_workspace(work):
    with pytest.raises(ObservationError, match="already open"):
        Workspace(work.root, work.model)


def test_restart_retains_interrupted_run(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"fixture")
    root = tmp_path / "comparisons"
    work = Workspace(root, model)
    run = root / ("a" * 32)
    run.mkdir()
    work._save(run, {"id": run.name, "status": "running", "created": "2026-10-04", "inputs": []})
    work.close()
    resumed = Workspace(root, model)
    try:
        assert resumed.status(run.name)["status"] == "interrupted"
    finally:
        resumed.close()


def test_http_capability_origin_limits_and_path_boundaries(work):
    server = WorkspaceServer(work)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(server.url) as response:
            assert b"Compare two portraits" in response.read()
        with urlopen(server.url + "api/jobs") as response:
            assert json.load(response) == []
        for target, headers, expected in [
            (server.origin + "/api/jobs", {}, 404),
            (server.url + "api/jobs", {"Origin": "https://foreign.invalid"}, 403),
            (server.url + "api/jobs", {"Host": "foreign.invalid"}, 403),
            (server.url + "runs/" + "a" * 32 + "/../../model.task", {}, 404),
        ]:
            with pytest.raises(HTTPError) as caught:
                urlopen(Request(target, headers=headers))
            assert caught.value.code == expected
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(server.url + "api/jobs", data=b"{}", headers={"Content-Type": "text/plain"}))
        assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
