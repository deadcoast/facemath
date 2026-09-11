"""Executable acquisition/measurement/artifact path; no detector is impersonated."""

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest
from test_wave4_reference import case

from facemath.observation.annotation_ui import create_annotation_workspace
from facemath.observation.contracts import ObservationError
from facemath.observation.lab import phantom_observation, phantom_points, render_phantom
from facemath.observation.validation_io import (
    replay_validation_bundle,
    run_validation_study,
    verify_validation_bundle,
)


def study(root, count=11, acquisition=False):
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(count):
        c = case(i, split="calibration" if i < 9 else "evaluation", delta=(0.1 * i, 0))
        image = render_phantom(phantom_points(), seed=i + 500)
        ip = f"{i}.png"
        rp = f"{i}.reference.json"
        op = f"{i}.observation.json"
        cv2.imwrite(str(root / ip), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        (root / rp).write_text(json.dumps(c.references[0].to_dict()))
        (root / op).write_text(json.dumps(c.observation.to_dict()))
        e = dict(
            case_id=c.case_id,
            group_id=c.group_id,
            split=c.split,
            image=ip,
            references=[rp],
        )
        if not acquisition:
            e["observation"] = op
        entries.append(e)
    spec = dict(
        format="facemath.validation_study.v4",
        domain="synthetic_io_controls",
        minimum_raters=1,
        cases=entries,
    )
    path = root / "study.json"
    path.write_text(json.dumps(spec))
    return path


def test_study_exports_every_input_raw_array_and_actual_events(tmp_path):
    source = study(tmp_path / "inputs")
    target = run_validation_study(source, tmp_path / "run")
    verified = verify_validation_bundle(target)
    assert verified["verified"] and verified["arrays"] == 11
    with np.load(target / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["reference_xy"].shape == (11, 8, 2)
        np.testing.assert_allclose(arrays["error_norm_px"][5], 0.5)
    report = json.loads((target / "validation.json").read_text())
    assert report["alignment"].startswith("NONE") and len(report["cases"]) == 11
    events = [json.loads(x) for x in (target / "events.jsonl").read_text().splitlines()]
    text = json.dumps(events)
    assert "provider_invocation" not in text
    assert text.count("imported_observation_no_new_inference") == 11
    assert replay_validation_bundle(target, tmp_path / "replay")["replay_verified"]


@pytest.mark.parametrize(
    "asset",
    [
        "arrays.npz",
        "validation.json",
        "case_0000_reference_00.json",
        "events.jsonl",
        "environment.json",
    ],
)
def test_validation_rejects_byte_corruption(tmp_path, asset):
    run = run_validation_study(study(tmp_path / "inputs"), tmp_path / "run")
    with (run / asset).open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ObservationError, match="mismatch"):
        verify_validation_bundle(run)


def test_validation_rejects_extra_claimed_artifact(tmp_path):
    run = run_validation_study(study(tmp_path / "inputs"), tmp_path / "run")
    (run / "pretend_truth.json").write_text("{}")
    with pytest.raises(ObservationError, match="unlisted"):
        verify_validation_bundle(run)


def test_validation_create_only_and_failed_attempts_preserved(tmp_path):
    source = study(tmp_path / "inputs")
    spec = json.loads(source.read_text())
    e = spec["cases"][-1]
    del e["observation"]
    e["failure"] = "scripted detection failure, not real model inference"
    source.write_text(json.dumps(spec))
    run = run_validation_study(source, tmp_path / "run")
    report = json.loads((run / "validation.json").read_text())
    assert report["splits"]["evaluation"]["attempted_cases"] == 2
    assert report["splits"]["evaluation"]["failed_cases"] == 1
    assert report["cases"][-1]["nme"] is None
    with pytest.raises(FileExistsError):
        run_validation_study(source, run)


@pytest.mark.parametrize(
    "fault",
    ["traversal", "absolute", "mismatched_image", "split_leakage", "two_origins"],
)
def test_bad_study_is_rejected_without_published_bundle(tmp_path, fault):
    source = study(tmp_path / "inputs")
    spec = json.loads(source.read_text())
    e = spec["cases"][0]
    if fault == "traversal":
        e["image"] = "../inputs/../outside.png"
    if fault == "absolute":
        e["image"] = str((source.parent / "0.png").absolute())
    if fault == "mismatched_image":
        e["image"] = "1.png"
    if fault == "split_leakage":
        spec["cases"][-1]["group_id"] = e["group_id"]
    if fault == "two_origins":
        e["failure"] = "bad duplicate origin"
    source.write_text(json.dumps(spec))
    out = tmp_path / "not-published"
    with pytest.raises(ObservationError):
        run_validation_study(source, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".facemath-validation-*"))


def test_runtime_fixture_failures_are_real_invocations_but_not_live_accuracy(tmp_path):
    class ScriptedProvider:
        def __init__(self):
            self.calls = 0

        def observe(self, image):
            self.calls += 1
            if self.calls == 11:
                raise ObservationError("scripted nondetection")
            return phantom_observation(image, phantom_points())

    source = study(tmp_path / "inputs", acquisition=True)
    provider = ScriptedProvider()
    run = run_validation_study(source, tmp_path / "run", provider=provider)
    assert provider.calls == 11
    log = (run / "events.jsonl").read_text()
    events = [json.loads(x) for x in log.splitlines()]
    assert sum(e["operation"] == "provider.observe.begin" for e in events) == 11
    assert sum(e["operation"] == "provider.observe.end" for e in events) == 10
    assert sum(e["operation"] == "provider.observe.error" for e in events) == 1
    assert "scripted nondetection" in log
    r = json.loads((run / "validation.json").read_text())
    assert r["splits"]["evaluation"]["failed_cases"] == 1
    assert replay_validation_bundle(run, tmp_path / "replay")["replay_verified"]


def test_provider_source_misbinding_aborts_instead_of_recording_nondetection(tmp_path):
    class WrongImageProvider:
        def observe(self, image):
            return phantom_observation(
                render_phantom(phantom_points(), seed=909090), phantom_points()
            )

    source = study(tmp_path / "inputs", acquisition=True)
    with pytest.raises(ObservationError):
        run_validation_study(source, tmp_path / "run", provider=WrongImageProvider())
    assert not (tmp_path / "run").exists()


def test_reference_workspace_starts_blank_and_is_script_safe(tmp_path):
    image = render_phantom(phantom_points())
    path = create_annotation_workspace(
        image,
        tmp_path / "annotation.html",
        source="</script><script>alert(1)</script>",
        kind="synthetic_fixture",
    )
    html = path.read_text()
    assert "connect-src 'none'" in html
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html
    assert "points=Array(8).fill(null)" in html
    assert "synthetic_fixture" in html
    with pytest.raises(FileExistsError):
        create_annotation_workspace(image, path, source="different")


def test_validation_cli_and_replay_dispatch(tmp_path):
    source = study(tmp_path / "inputs")
    run = tmp_path / "run"
    cmd = [sys.executable, "-m", "facemath"]
    for args in [
        ["validate-observer", str(source), "--out", str(run)],
        ["verify", str(run)],
        ["replay", str(run), "--out", str(tmp_path / "replay")],
    ]:
        p = subprocess.run(cmd + args, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        data = json.loads(p.stdout)
        assert data.get("verified") or data.get("replay_verified")


def test_doctor_never_claims_inference(tmp_path):
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "facemath",
            "doctor",
            "--model",
            str(tmp_path / "missing.task"),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data["inference_executed"] is False and data["model_present"] is False
