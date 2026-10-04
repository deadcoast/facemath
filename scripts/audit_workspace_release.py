"""Collect the executed application workflow, then summarize the completed release."""
from __future__ import annotations
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from facemath.observation.journal import code_fingerprint


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--app", type=Path)
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    feature = root / "evidence/workspace-20261004"
    if args.collect:
        browser = read(args.browser / "browser.json")
        assert browser["passed"]
        destination = feature / "final"
        destination.mkdir()
        shutil.copytree(args.browser, destination / "browser", ignore=shutil.ignore_patterns("comparison.zip"))
        for key, expected in (("run_id", "complete"), ("no_face_run", "failed"), ("cancelled_run", "cancelled")):
            source = args.app / browser[key]
            record = read(source / "status.json")
            assert record["status"] == expected
            shutil.copytree(source, destination / browser[key], ignore=shutil.ignore_patterns("runtime-cache"))
        complete = destination / browser["run_id"]
        environment = read(complete / "result/environment.json")
        assert environment["engine_code_sha256"] == code_fingerprint()
        assert digest(args.browser / "comparison.zip") == digest(complete / "comparison.zip")
        write(destination / "workflow.json", {"code_sha256": code_fingerprint(),
            "browser": browser, "native_returncode": read(complete / "status.json")["returncode"],
            "download_sha256": digest(complete / "comparison.zip"),
            "scope": "actual native browser workflow; no new cohort or anatomical validation"})
        runtime = feature / "runtime"
        runtime.mkdir()
        for name in ("face_landmarker.task", "portrait.png"):
            shutil.copyfile(root / "evidence/edge-comparison-20261004/runtime" / name, runtime / name)
        print(json.dumps({"collected": str(destination), "code_sha256": code_fingerprint()}))
        return
    release = args.release.resolve()
    validation = read(release / "validation.json")
    assert validation["completed"] and all(x["returncode"] == 0 for x in validation["commands"])
    tests = {}
    milestones = {}
    for tier, relative in (("source", "source.xml"), ("wheel", "wheel-check/pytest.xml"), ("archive", "archive.xml")):
        suite = ET.parse(release / relative).getroot().find("testsuite")
        counts = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
        assert not counts["failures"] and not counts["errors"]
        counts["passed"] = counts["tests"] - counts["skipped"]
        counts["skip_reasons"] = sorted({x.attrib.get("message", "") for x in suite.iter("skipped")})
        live = [case for case in suite.iter("testcase") if case.attrib.get("name") == "test_live_mediapipe_provider"]
        counts["live_observer_smoke_passed"] = bool(live) and all(case.find("skipped") is None for case in live)
        assert counts["live_observer_smoke_passed"]
        tests[tier] = counts
        lab = release / ({"source": "source-lab", "wheel": "wheel-check/milestone", "archive": "archive-lab"}[tier]) / "laboratory.json"
        record = read(lab)
        pair = record["admitted_pair"]
        assert pair["raw_arrays_unchanged"] and pair["qualification_snapshot_replayed"]
        milestones[tier] = {"observed_arrays_unchanged": pair["raw_arrays_checked"],
                            "qualification_snapshot_replayed": True, "scope": "synthetic qualification control"}
    post = root / "evidence/workspace-release-20261004"
    post.mkdir()
    for source in release.iterdir():
        if source.is_file() and source.suffix in (".json", ".log", ".xml"):
            shutil.copyfile(source, post / source.name)
    shutil.copytree(release / "wheel-check", post / "wheel-check",
                    ignore=shutil.ignore_patterns("replayed", "pair-replayed", "milestone"))
    archive = root / "packages/facemath-0.22.0-local-application-20261004.zip"
    if archive.exists():
        raise FileExistsError(archive)
    shutil.copyfile(release / "facemath-0.22.0.zip", archive)
    workflow = read(feature / "final/workflow.json")
    assert workflow["code_sha256"] == code_fingerprint()
    result = {"version": "0.22.0", "code_sha256": code_fingerprint(), "tests": tests, "milestones": milestones,
        "archive": {"path": archive.relative_to(root).as_posix(), "sha256": digest(archive),
                    "files": validation["archive"]["files"]},
        "application": workflow, "installed_bytes": read(release / "wheel-check/installed-byte-check.json"),
        "limits": ["Workspace exposes observed comparisons; research calibration and qualification remain CLI options.",
                   "This supplies a working local workflow, not independent anatomical validation."],
        "evidence": post.relative_to(root).as_posix()}
    write(root / "APP_WORKSPACE.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
