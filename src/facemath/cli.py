"""FACEMATH CLI: frozen acquisition, qualified observation, and one geometric state."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import fields
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from . import __version__

from .observation import (
    FacialInstrument,
    InstrumentConfig,
    MediaPipeProvider,
    Observation,
    ObservationError,
    ProviderUnavailable,
)
from .observation.artifacts import verify_bundle, write_bundle
from .observation.capabilities import inventory
from .observation.cohort import Cohort, cohort_from_bundles
from .observation.journal import code_fingerprint, numeric_environment
from .observation.lab import (
    laboratory,
    phantom_observation,
    phantom_points,
    render_phantom,
)
from .observation.provider import annotated_observation, locate_faces
from .observation.sequence_guard import inspect_video


from .observation.study import load_rgb, read_json


def write_json_new(path: str | Path, data: dict) -> None:
    """Create a new file. Never silently replace observation or run evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, allow_nan=False)
    with target.open("x", encoding="utf-8") as output:
        output.write(text + "\n")


def _configuration(args: argparse.Namespace) -> InstrumentConfig:
    return InstrumentConfig(
        canvas_size=args.size,
        eye_span_px=args.size * .3125,
        uncertainty_draws=args.draws,
        seed=args.seed,
        tolerance_ipd=args.tolerance,
        enable_flow=not args.no_flow,
        enable_edges=not args.no_edges,
        enable_intrinsic=not args.no_intrinsic,
        intrinsic_length_tolerance=args.relation_length_tolerance,
        intrinsic_angle_tolerance_rad=np.deg2rad(args.relation_angle_tolerance_deg),
        intrinsic_area_tolerance=args.relation_area_tolerance,
        intrinsic_asymmetry_tolerance=args.relation_asymmetry_tolerance,
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=256, help="canonical canvas size, 64..1024")
    parser.add_argument("--draws", type=int, default=96, help="conditional sensitivity draws, 4..4096")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=.05, help="landmark tolerance in registered eye spans")
    parser.add_argument("--no-flow", action="store_true", help="disable independent optical-flow diagnostic")
    parser.add_argument("--no-edges", action="store_true", help="disable landmark-guided image edge diagnostics")
    parser.add_argument("--no-intrinsic", action="store_true", help="disable reference8 intrinsic relations")
    parser.add_argument("--relation-length-tolerance", type=float, default=.05,
                        help="difference tolerance for lengths in each source's own reference eye spans")
    parser.add_argument("--relation-angle-tolerance-deg", type=float, default=5., help="angular difference tolerance; stored in radians")
    parser.add_argument("--relation-area-tolerance", type=float, default=.05, help="signed double-area difference in squared eye spans")
    parser.add_argument("--relation-asymmetry-tolerance", type=float, default=.05, help="signed bilateral contrast difference tolerance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facemath",
        description="One observed anatomy. One coordinate gauge. Raw, replayable evidence.",
    )
    parser.add_argument("--version", action="version", version=f"facemath {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Wave 3 integrated pair instrument")
    p.add_argument("image_a")
    p.add_argument("image_b")
    p.add_argument("--observation-a")
    p.add_argument("--observation-b")
    p.add_argument("--acquisition-a", help="original-pixel measured group bundle for source A")
    p.add_argument("--acquisition-b", help="original-pixel measured group bundle for source B")
    p.add_argument("--model", help="local MediaPipe .task model when observations are not supplied")
    p.add_argument("--model-sha256")
    p.add_argument("--lineage-a", action="append", default=[], help="declared source A acquisition lineage; repeatable")
    p.add_argument("--lineage-b", action="append", default=[], help="declared source B acquisition lineage; repeatable")
    p.add_argument("--qualification", help="complete held-out study bundle; numerically replayed before empirical admission")
    p.add_argument("--localization-calibration", help="independent reference error calibration JSON")
    p.add_argument("--query-groups", nargs=2, metavar=("GROUP_A", "GROUP_B"))
    p.add_argument("--query-strata", nargs=2, metavar=("CATEGORY_A", "CATEGORY_B"),
                   help="prespecified acquisition categories for a stratified error model; never inferred from similarity")
    p.add_argument("--domain", default="", help="calibration acquisition domain, not inferred")
    p.add_argument("--out", required=True, help="new evidence bundle directory")
    _add_config(p)

    p = sub.add_parser("observe", help="local MediaPipe observation with augmentation repeatability")
    p.add_argument("image")
    p.add_argument("--model", required=True)
    p.add_argument("--model-sha256")
    p.add_argument("--out", required=True)
    p.add_argument("--sigma", type=float, default=.75)

    p = sub.add_parser("import-annotation", help="bind explicit semantic21 JSON to source pixels")
    p.add_argument("image")
    p.add_argument("annotation")
    p.add_argument("--annotation-id", required=True)
    p.add_argument("--sigma", required=True, type=float)
    p.add_argument("--out", required=True)

    p = sub.add_parser("locate", help="offline coarse face ROI proposals; not anatomical landmarks")
    p.add_argument("image")
    p.add_argument("--out", required=True)

    p = sub.add_parser("inspect-video", help="time-aware editing/mix diagnostics; never a similarity score")
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))

    p = sub.add_parser("demo", help="known-geometry synthetic bundle, no detector models needed")
    p.add_argument("--out", required=True)
    _add_config(p)

    p = sub.add_parser("laboratory", help="execute quantitative falsification controls")
    p.add_argument("--out", required=True)

    p = sub.add_parser("verify", help="verify byte hashes, array contracts, and execution journal")
    p.add_argument("bundle")

    p = sub.add_parser("replay", help="recompute from stored source rasters and observations")
    p.add_argument("bundle")
    p.add_argument("--out", required=True)

    p = sub.add_parser("runtime-profile", help="record numerical kernels and check exact-replay prerequisites")
    p.add_argument("--compare", help="retained environment.json, session.json, or runtime profile; exit 2 if incompatible")
    p.add_argument("--out", help="new JSON file; never overwrite retained runtime evidence")

    p = sub.add_parser("capabilities", help="audit original mappings without claiming mathematical equivalence")
    p.add_argument("--out")

    p = sub.add_parser("cohort-create", help="ingest verified, compatible run bundles as an explicit reference")
    p.add_argument("bundles", nargs="+")
    p.add_argument("--name", required=True)
    p.add_argument("--kind", required=True, choices=("synthetic", "observational"))
    p.add_argument("--metrics", nargs="+", default=["geometry.all.rms"])
    p.add_argument("--subjects", help="optional JSON mapping image SHA-256 to declared subject ID")
    p.add_argument("--out", required=True)

    p = sub.add_parser("calibrate", help="one measured metric against an exact-recipe reference cohort")
    p.add_argument("bundle")
    p.add_argument("--cohort", required=True)
    p.add_argument("--metric", default="geometry.all.rms")
    p.add_argument("--out", required=True)
    p.add_argument("--higher-is-closer", action="store_true")
    p.add_argument("--query-subjects", nargs="*", default=[])
    p = sub.add_parser("annotate", help="create offline prediction-blind eight-point reference workbench")
    p.add_argument("image")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True, help="new standalone HTML file")

    p = sub.add_parser("import-ibug-reference", help="import reviewed eight-point subset of a 68-point PTS annotation")
    p.add_argument("image")
    p.add_argument("pts")
    p.add_argument("--origin", required=True, choices=(0, 1), type=int)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("validate-observer", help="run source-coordinate evaluation with independent reference labels")
    p.add_argument("study", help="local study manifest with development/calibration/evaluation groups")
    p.add_argument("--model", help="local model, needed only when saved observations are absent")
    p.add_argument("--model-sha256")
    p.add_argument("--alpha", type=float, default=.1)
    p.add_argument("--out", required=True)

    p = sub.add_parser("validation-lab", help="execute synthetic stable-bias localization controls; not human accuracy")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=24091)

    p = sub.add_parser("observer-pin", help="declare a local model/runtime recipe; performs no inference")
    p.add_argument("--model", required=True)
    p.add_argument("--runtime-version", help="explicit planned MediaPipe version; defaults to installed version")
    p.add_argument("--sigma", type=float, default=.75)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--out", required=True)

    p = sub.add_parser("study-freeze", help="snapshot enrolled images, references and protocol before inference")
    p.add_argument("study", help="prospective study JSON; no imported predictions")
    p.add_argument("--frozen-model", help="exported model capsule for evaluation-only transfer")
    p.add_argument("--registry", help="immutable prior-enrollment registry; refuse any source overlap")
    p.add_argument("--out", required=True)

    p = sub.add_parser("study-run", help="acquire or resume one native observation attempt per included source")
    p.add_argument("study", help="frozen study directory")
    p.add_argument("--model", required=True, help="exact local model matching the frozen SHA-256")
    p.add_argument("--max-cases", type=int, help="bound new attempts this invocation without changing the study")
    p.add_argument("--out", required=True, help="new or matching resumable acquisition session")

    p = sub.add_parser("study-status", help="verify a session and report all acquisition outcomes")
    p.add_argument("session")
    p.add_argument("--raw", action="store_true", help="include every full receipt in console JSON")

    p = sub.add_parser("study-finalize", help="qualify a completed session; no model is rerun")
    p.add_argument("session")
    p.add_argument("--out", required=True)

    p = sub.add_parser("model-export", help="export complete fitted joint state and full donor enrollment")
    p.add_argument("qualification")
    p.add_argument("--out", required=True)
    p = sub.add_parser("enrollment", help="export full enrollment, build immutable registries and check source reuse")
    e = p.add_subparsers(dest="enrollment_command", required=True)
    q = e.add_parser("export"); q.add_argument("study"); q.add_argument("--out", required=True)
    q = e.add_parser("build"); q.add_argument("--receipt", action="append", required=True)
    q.add_argument("--scope", required=True); q.add_argument("--prior"); q.add_argument("--out", required=True)
    q = e.add_parser("check"); q.add_argument("registry"); q.add_argument("study"); q.add_argument("--out", required=True)
    p = sub.add_parser("transfer-finalize", help="evaluate a frozen-model acquisition without fitting")
    p.add_argument("session")
    p.add_argument("--out", required=True)

    p = sub.add_parser("study-lab", help="execute controlled prospective bias/domain-shift study; NOT human validation")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=25091)

    p = sub.add_parser("intrinsic-lab", help="execute uncertain-normalization and refitted-shape controls; synthetic only")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=26091)

    p = sub.add_parser("joint-lab", help="execute correlated-reference-error study and shape controls; synthetic only")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=27091)

    p = sub.add_parser("stratified-lab", help="execute acquisition-category calibration and mixed-source geometry controls; synthetic only")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=28091)

    p = sub.add_parser("acquisition-group", help="measure a declared raw source group under frozen pixel rules; no detector")
    p.add_argument("spec", help="group JSON with original images, apertures and complete declared lineages")
    p.add_argument("--policy", required=True, help="measurement policy or measured calibration JSON")
    p.add_argument("--out", required=True)

    p = sub.add_parser("measured-lab", help="execute pixel-routed acquisition and integrated geometry controls; synthetic only")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=29091)

    p = sub.add_parser("doctor", help="report runtime/model availability; never claims inference was run")
    p.add_argument("--model")
    p.add_argument("--out")
    p = sub.add_parser("overview", help="compact typed readout of a verified pair or study")
    p.add_argument("bundle")
    p.add_argument("--raw", action="store_true")

    p = sub.add_parser("observer-smoke", help="execute the actual native worker or retain an honest preflight failure")
    p.add_argument("image")
    p.add_argument("--model", required=True)
    p.add_argument("--model-sha256")
    p.add_argument("--timeout", type=float, default=90.)
    p.add_argument("--out", required=True)

    p = sub.add_parser("dataset", help="fetch pinned assets, prepare published-reference data, execute a native pipeline")
    d = p.add_subparsers(dest="dataset_command", required=True)
    q = d.add_parser("fetch"); q.add_argument("recipe"); q.add_argument("--out", required=True)
    q = d.add_parser("prepare"); q.add_argument("recipe"); q.add_argument("--out", required=True)
    q = d.add_parser("run"); q.add_argument("dataset"); q.add_argument("--model"); q.add_argument("--out", required=True)
    q.add_argument("--frozen-model", help="exported joint model capsule; target must have evaluation cases only")
    q.add_argument("--registry", help="immutable prior-enrollment registry bound before inference")
    q = d.add_parser("cohort-fetch"); q.add_argument("plan"); q.add_argument("--out", required=True)
    q = d.add_parser("cohort"); q.add_argument("plan"); q.add_argument("--archive", required=True)
    q.add_argument("--model", required=True); q.add_argument("--out", required=True)

    p = sub.add_parser("workspace", help="prepare portraits, import independent label revisions, freeze a study")
    w = p.add_subparsers(dest="workspace_command", required=True)
    q = w.add_parser("prepare"); q.add_argument("spec"); q.add_argument("--out", required=True)
    q = w.add_parser("import"); q.add_argument("workspace"); q.add_argument("reference"); q.add_argument("--case-id", required=True)
    q = w.add_parser("status"); q.add_argument("workspace")
    q = w.add_parser("freeze"); q.add_argument("workspace"); q.add_argument("--out", required=True)

    p = sub.add_parser("milestone-lab", help="Wave10 end-to-end consolidation and empirical acceptance controls")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=31091)
    return parser


def _analyze(args: argparse.Namespace) -> dict:
    image_a, image_b = load_rgb(args.image_a), load_rgb(args.image_b)
    obs_a = Observation.from_dict(read_json(args.observation_a)) if args.observation_a else None
    obs_b = Observation.from_dict(read_json(args.observation_b)) if args.observation_b else None
    config = _configuration(args)
    acquisition = None
    if args.acquisition_a or args.acquisition_b:
        if not (args.acquisition_a and args.acquisition_b):
            raise ObservationError("supply both measured acquisition groups")
        from .observation.acquisition import load_group
        acquisition = (load_group(args.acquisition_a), load_group(args.acquisition_b))
    calibration = None
    qualification = None
    if getattr(args, "qualification", None):
        if args.localization_calibration:
            raise ObservationError("choose a complete qualification OR a research calibration, not both")
        from .observation.deployment import ValidatedQualification
        qualification = ValidatedQualification.from_directory(args.qualification)
        calibration = qualification.calibration
        if not args.domain:
            args.domain = qualification.report["domain"]
    if args.localization_calibration:
        from .observation.bias import load_localization_calibration
        calibration = load_localization_calibration(read_json(args.localization_calibration))
        if (not args.query_groups and acquisition is None) or not args.domain:
            raise ObservationError("calibrated analysis requires --domain and query groups (or complete measured source groups)")
    extra = dict(localization_calibration=calibration, query_groups=tuple(args.query_groups or ("", "")), domain=args.domain,
                 query_lineages=(tuple(args.lineage_a), tuple(args.lineage_b)),
                 query_strata=tuple(args.query_strata or ("", "")), acquisition_groups=acquisition, qualification_evidence=qualification)
    if obs_a is not None and obs_b is not None:
        result = FacialInstrument(config, **extra).run(image_a, image_b, obs_a, obs_b)
    else:
        if not args.model:
            raise ObservationError("supply both observation files or --model; no fabricated anatomical fallback")
        # Acquisition-policy refusal must precede native model construction too.
        # The instrument invokes observe only after validating both pixel groups.
        with ExitStack() as resources:
            class DeferredProvider:
                def __init__(self):
                    self.instance = None

                def observe(self, image):
                    if self.instance is None:
                        self.instance = resources.enter_context(MediaPipeProvider(args.model, args.model_sha256))
                    return self.instance.observe(image)

            result = FacialInstrument(config, DeferredProvider(), **extra).run(image_a, image_b, obs_a, obs_b)
    output = write_bundle(result, args.out, image_a, image_b)
    return {"bundle": str(output), "report": str(output / "report.html"), **verify_bundle(output)}


def _inspect_video(args: argparse.Namespace) -> dict:
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    roi = tuple(args.roi) if args.roi else None
    data, frames = inspect_video(args.video, args.frames, roi)
    target.mkdir(parents=True)
    write_json_new(target / "video_inspection.json", data)
    cells = []
    for image, record in zip(frames, data["sampling_records"]):
        scale = min(256 / image.shape[1], 260 / image.shape[0])
        thumb = cv2.resize(image, None, fx=scale, fy=scale)
        cell = np.full((288, 276, 3), 245, np.uint8)
        cell[10:10 + thumb.shape[0], 10:10 + thumb.shape[1]] = thumb
        label = f"frame {record['frame_index']} | {record['timestamp_cfr_seconds']:.3f}s"
        cv2.putText(cell, label, (10, 281), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1, cv2.LINE_AA)
        cells.append(cell)
    while len(cells) % 4:
        cells.append(np.full_like(cells[0], 245))
    sheet = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])
    encoded = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(target / "contact.png"), encoded):
        raise ObservationError("failed to write video inspection contact sheet")
    return {"directory": str(target), "frames": len(frames), "facial_similarity_evidence": False}


def _replay(args: argparse.Namespace) -> dict:
    root = Path(args.bundle)
    if read_json(root / "manifest.json").get("format") in ("facemath.transfer_pipeline.v14", "facemath.transfer_bundle.v14"):
        from .observation.transfer_io import replay_transfer_bundle, verify_transfer_pipeline
        if read_json(root / "manifest.json")["format"] == "facemath.transfer_pipeline.v14":
            verify_transfer_pipeline(root)
            root = root / "evaluation"
        return replay_transfer_bundle(root, args.out)
    if read_json(root / "manifest.json").get("format") == "facemath.native_pipeline_bundle.v11":
        from .observation.dataset import verify_pipeline
        from .observation.study_io import replay_qualification_bundle
        verify_pipeline(root)
        return replay_qualification_bundle(root / "qualification", args.out)
    if read_json(root / "manifest.json").get("format") == "facemath.acquisition_group_bundle.v9":
        from .observation.acquisition import load_group, write_group
        group = load_group(root)
        target = write_group(group, args.out)
        return {"replay_verified": True, "new_bundle": str(target), "group_sha256": group.digest,
                "scope": "exact source-pixel remeasurement, no detector invocation"}
    if read_json(root / "manifest.json").get("format") == "facemath.qualification_bundle.v5":
        from .observation.study_io import replay_qualification_bundle
        return replay_qualification_bundle(root, args.out)
    if read_json(root / "manifest.json").get("format") == "facemath.validation_bundle.v4":
        from .observation.validation_io import replay_validation_bundle
        return replay_validation_bundle(root, args.out)
    verify_bundle(root)
    env = read_json(root / "environment.json")
    if env["engine_code_sha256"] != code_fingerprint():
        raise ObservationError("source implementation differs; perform a new analysis, not an exact replay")
    from .observation.numerics import require_numeric_environment
    require_numeric_environment(env["numeric_environment"], "pair replay")
    evidence = read_json(root / "evidence.json")
    observations = read_json(root / "observations.json")
    allowed = {f.name for f in fields(InstrumentConfig)}
    config = InstrumentConfig(**{k: v for k, v in evidence["configuration"].items() if k in allowed})
    image_a, image_b = load_rgb(root / "source_a.png"), load_rgb(root / "source_b.png")
    calibration = None
    envelope = evidence.get("localization_envelope")
    if envelope:
        from .observation.bias import load_localization_calibration
        calibration = load_localization_calibration(read_json(root / "localization_calibration.json"))
        if calibration.digest != envelope["calibration_sha256"]:
            raise ObservationError("localization calibration binding mismatch")
    acquisition = None
    if evidence.get("measured_acquisition") is not None:
        from .observation.acquisition import load_group_files
        acquisition = tuple(load_group_files(root, f"acquisition_{n}_") for n in ("a", "b"))
    qualification = None
    if evidence.get("empirical_admission", {}).get("format") == "facemath.validated_qualification_link.v10":
        from .observation.deployment import load_snapshot
        qualification = load_snapshot(root / "qualification_snapshot.zip")
    result = FacialInstrument(config, localization_calibration=calibration, acquisition_groups=acquisition,
        qualification_evidence=qualification,
        query_groups=tuple(envelope["query_groups"]) if envelope else ("", ""),
        query_lineages=tuple(tuple(x) for x in envelope.get("query_lineages", ((), ()))) if envelope else ((), ()),
        domain=envelope["domain"] if envelope else "").run(
        image_a, image_b,
        Observation.from_dict(observations["a"]),
        Observation.from_dict(observations["b"]),
    )
    from .observation.contracts import object_digest
    if object_digest(result.evidence().get("empirical_admission")) != object_digest(evidence.get("empirical_admission")):
        raise ObservationError("replayed empirical admission differs")
    if object_digest(result.evidence().get("measured_acquisition")) != object_digest(evidence.get("measured_acquisition")):
        raise ObservationError("replayed source-pixel acquisition record differs")
    if object_digest(result.evidence().get("intrinsic_profile")) != object_digest(evidence.get("intrinsic_profile")):
        raise ObservationError("replayed intrinsic profile differs")
    if result.state.recipe_sha256 != evidence["recipe_sha256"]:
        raise ObservationError("replayed measurement recipe differs")
    previous_metrics = {m["name"]: m for m in evidence["measurement_profile"]}
    if set(previous_metrics) != {m.name for m in result.profile.measurements}:
        raise ObservationError("replayed measurement inventory differs")
    for measurement in result.profile.measurements:
        expected = previous_metrics[measurement.name]["value"]
        actual = measurement.value
        if expected is None or actual is None:
            if expected != actual:
                raise ObservationError(f"replay support mismatch: {measurement.name}")
        elif not np.isclose(actual, expected, atol=1e-10, rtol=1e-8):
            raise ObservationError(f"replay measurement mismatch: {measurement.name}")
    arrays = result.arrays()
    with np.load(root / "arrays.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ObservationError("replayed raw-state inventory differs")
        for name, array in arrays.items():
            if saved[name].shape != array.shape or not np.allclose(saved[name], array, atol=1e-9, rtol=1e-7):
                raise ObservationError(f"replay raw-state mismatch: {name}")
    output = write_bundle(result, args.out, image_a, image_b)
    return {
        "replay_verified": True,
        "source_bundle": str(root),
        "new_bundle": str(output),
        "numerical_tolerance": "arrays atol=1e-9, rtol=1e-7; measurements atol=1e-10, rtol=1e-8",
    }


def _calibrate(args: argparse.Namespace) -> dict:
    root = Path(args.bundle)
    verify_bundle(root)
    evidence, observations = read_json(root / "evidence.json"), read_json(root / "observations.json")
    cohort = Cohort.from_dict(read_json(args.cohort))
    value = next((m["value"] for m in evidence["measurement_profile"] if m["name"] == args.metric), None)
    if value is None:
        raise ObservationError("metric missing or withheld in query evidence")
    data = cohort.calibrate(
        args.metric, value, evidence["recipe_sha256"],
        (observations["a"]["image_sha256"], observations["b"]["image_sha256"]),
        lower_is_closer=not args.higher_is_closer,
        query_subjects=tuple(args.query_subjects),
    )
    write_json_new(args.out, data)
    return data


def _session_summary(status: dict, directory: str | Path) -> dict:
    """Keep normal console output bounded without removing any disk evidence."""
    summary = {k: v for k, v in status.items() if k != "receipts"}
    summary["receipt_directory"] = str(Path(directory) / "cases")
    summary["full_receipts_omitted_from_console"] = len(status.get("receipts", ()))
    return summary


def execute(args: argparse.Namespace) -> dict:
    if args.command == "enrollment":
        from .observation.enrollment import export_enrollment, build_registry, load_registry, check_enrollment, read_registry
        from .observation.study import load_study
        if args.enrollment_command == "export":
            return export_enrollment(args.study, args.out)
        if args.enrollment_command == "build":
            registry = build_registry([read_json(p) for p in args.receipt], args.scope,
                                      prior=read_registry(args.prior) if args.prior is not None else None)
            write_json_new(args.out, registry)
            rows = load_registry(registry)
            return {"registry": args.out, "content_sha256": registry["content_sha256"],
                    "scope": registry["scope"], "registered_studies": len({r["study_sha256"] for r in rows}),
                    "registered_cases": len(rows), "inference_executed": False}
        study = load_study(args.study)
        result = check_enrollment(read_registry(args.registry), [(c, study.references(c)) for c in study.cases])
        result.update(target_study_sha256=study.digest, inference_executed=False)
        write_json_new(args.out, result)
        return result
    if args.command == "model-export":
        from .observation.transfer import export_model
        return export_model(args.qualification, args.out)
    if args.command == "transfer-finalize":
        from .observation.transfer_io import finalize_transfer, verify_transfer_bundle
        target = finalize_transfer(args.session, args.out)
        return {"bundle": str(target), **verify_transfer_bundle(target)}
    if args.command == "runtime-profile":
        from .observation.numerics import compare_numeric_environments
        profile = numeric_environment()
        data = {"format": "facemath.runtime_profile.v12", "numeric_environment": profile,
                "engine_code_sha256": code_fingerprint(), "inference_executed": False,
                "exact_replay_verified": False}
        if args.compare:
            previous = read_json(args.compare)
            data["comparison"] = compare_numeric_environments(previous.get("numeric_environment", previous), profile)
        if args.out:
            write_json_new(args.out, data)
        return data
    if args.command == "dataset":
        from .observation.dataset import fetch_dataset, prepare_dataset, run_pipeline, verify_dataset
        if args.dataset_command == 'cohort-fetch':
            from .observation.source_cohort import fetch_cohort
            return {'assets': str(fetch_cohort(args.plan, args.out))}
        if args.dataset_command == 'cohort':
            from .observation.source_cohort import prepare_cohort
            root = prepare_cohort(args.plan, args.archive, args.model, args.out)
            return {'dataset': str(root), **verify_dataset(root)}
        if args.dataset_command == "fetch":
            root = fetch_dataset(args.recipe, args.out)
            return {"assets": str(root), "pins_verified": True, "inference_executed": False}
        if args.dataset_command == "prepare":
            root = prepare_dataset(args.recipe, args.out)
            return {"dataset": str(root), **verify_dataset(root)}
        if args.frozen_model:
            from .observation.transfer_io import run_transfer_pipeline
            return run_transfer_pipeline(args.dataset, args.frozen_model, args.out, model=args.model, registry=args.registry)
        return run_pipeline(args.dataset, args.out, model=args.model, registry=args.registry)
    if args.command == "overview":
        from .observation.readout import inspect_readout
        return inspect_readout(args.bundle)
    if args.command == "observer-smoke":
        from .observation.native_check import native_smoke
        return native_smoke(args.image, args.model, args.out, timeout_seconds=args.timeout, expected_sha256=args.model_sha256)
    if args.command == "workspace":
        from .observation.workspace import prepare_workspace, import_reference, workspace_status, freeze_workspace
        if args.workspace_command == "prepare":
            root = prepare_workspace(args.spec, args.out)
            return {"workspace": str(root), **workspace_status(root)}
        if args.workspace_command == "import":
            return import_reference(args.workspace, args.reference, args.case_id)
        if args.workspace_command == "status":
            return workspace_status(args.workspace)
        study = freeze_workspace(args.workspace, args.out)
        return {"study": str(study.root), "study_sha256": study.digest, "inference_executed": False}
    if args.command == "milestone-lab":
        from .observation.milestone_lab import run_milestone_laboratory
        return run_milestone_laboratory(args.out, seed=args.seed)
    if args.command == "acquisition-group":
        from .observation.acquisition import policy_from_document, group_from_spec, write_group
        group = group_from_spec(args.spec, policy_from_document(read_json(args.policy)))
        out = write_group(group, args.out)
        return {"bundle": str(out), "group_sha256": group.digest, "derived_category": group.category,
                "deployment_permitted_by_rule": group.policy.permits(group.category), "inference_executed": False}
    if args.command == "measured-lab":
        from .observation.acquisition_lab import run_measured_laboratory
        return run_measured_laboratory(args.out, seed=args.seed)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "observe":
        with MediaPipeProvider(args.model, args.model_sha256, args.sigma) as provider:
            observation = provider.observe(load_rgb(args.image))
        write_json_new(args.out, observation.to_dict())
        return {"observation": args.out, "landmarks": len(observation.xy), "samples": len(observation.samples)}
    if args.command == "import-annotation":
        from .observation.anatomy import SEMANTIC_21
        obj = read_json(args.annotation)
        observation = annotated_observation(
            load_rgb(args.image), np.asarray(obj["xy"]), annotation_id=args.annotation_id,
            sigma_px=args.sigma, names=tuple(obj.get("names", SEMANTIC_21)),
            source_notes=tuple(obj.get("notes", [])),
        )
        write_json_new(args.out, observation.to_dict())
        return {"observation": args.out, "provider": observation.provider}
    if args.command == "locate":
        data = {"kind": "coarse_ROI_proposals_not_landmarks", "regions": locate_faces(load_rgb(args.image))}
        write_json_new(args.out, data)
        return data
    if args.command == "inspect-video":
        return _inspect_video(args)
    if args.command == "demo":
        points_a = phantom_points()
        points_b = points_a.copy()
        points_b[19, 0] -= 8
        points_b[20, 0] += 8
        image_a, image_b = render_phantom(points_a), render_phantom(points_b)
        result = FacialInstrument(_configuration(args)).run(
            image_a, image_b,
            phantom_observation(image_a, points_a),
            phantom_observation(image_b, points_b),
        )
        output = write_bundle(result, args.out, image_a, image_b)
        return {"bundle": str(output), "kind": "synthetic_known_geometry_not_human_data", **verify_bundle(output)}
    if args.command == "laboratory":
        data = laboratory()
        write_json_new(args.out, data)
        if not all(data["checks"].values()):
            raise RuntimeError("laboratory falsification check failed")
        return {"laboratory": args.out, "checks": data["checks"]}
    if args.command == "verify":
        root = Path(args.bundle)
        if (root / "session.json").is_file():
            from .observation.study_runner import verify_session
            return _session_summary(verify_session(root), root)
        kind = read_json(root / "manifest.json").get("format")
        if kind == "facemath.transfer_bundle.v14":
            from .observation.transfer_io import verify_transfer_bundle
            return verify_transfer_bundle(root)
        if kind == "facemath.transfer_pipeline.v14":
            from .observation.transfer_io import verify_transfer_pipeline
            return verify_transfer_pipeline(root)
        if kind == "facemath.dataset_bundle.v11":
            from .observation.dataset import verify_dataset
            return verify_dataset(root)
        if kind == "facemath.native_pipeline_bundle.v11":
            from .observation.dataset import verify_pipeline
            return verify_pipeline(root)
        if kind == "facemath.native_smoke_bundle.v10":
            from .observation.native_check import verify_smoke
            return verify_smoke(root)
        if kind == "facemath.acquisition_group_bundle.v9":
            from .observation.acquisition import load_group
            group = load_group(root)
            return {"verified": True, "kind": "measured_acquisition_group", "group_sha256": group.digest,
                    "source_measurements_recomputed": True, "category": group.category}
        if kind == "facemath.qualification_bundle.v5":
            from .observation.study_io import verify_qualification_bundle
            return verify_qualification_bundle(root)
        if kind in ("facemath.study_lock_manifest.v5", "facemath.study_lock_manifest.v16"):
            from .observation.study import load_study
            study = load_study(root)
            return {"verified": True, "kind": "frozen_study", "study_sha256": study.digest,
                    "enrolled": len(study.cases), "eligible": len(study.included), "inference_executed": False}
        if kind == "facemath.validation_bundle.v4":
            from .observation.validation_io import verify_validation_bundle
            return verify_validation_bundle(args.bundle)
        return verify_bundle(args.bundle)
    if args.command == "replay":
        return _replay(args)
    if args.command == "capabilities":
        data = inventory()
        if args.out:
            write_json_new(args.out, data)
        return data
    if args.command == "cohort-create":
        cohort = cohort_from_bundles(
            args.name, args.bundles, kind=args.kind, metrics=tuple(args.metrics),
            subjects_by_image=read_json(args.subjects) if args.subjects else None,
        )
        write_json_new(args.out, cohort.to_dict())
        return {"cohort": args.out, "kind": cohort.kind, "rows": len(cohort.rows),
                "sha256": cohort.fingerprint, "dependence": cohort.dependence()}
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "annotate":
        from .observation.annotation_ui import create_annotation_workspace
        path = create_annotation_workspace(load_rgb(args.image), args.out, source=args.source)
        return {"workbench": str(path), "predictions_shown": False, "labels_created": 0}
    if args.command == "import-ibug-reference":
        from .observation.reference import import_ibug68
        from .observation.contracts import array_digest
        image = load_rgb(args.image)
        ref = import_ibug68(Path(args.pts).read_text(encoding="utf-8"), image_sha256=array_digest(image),
                            image_shape=image.shape[:2], coordinate_origin=args.origin, source=args.source)
        write_json_new(args.out, ref.to_dict())
        return {"reference": args.out, "points": 8, "mapping_requires_review": True}
    if args.command == "validate-observer":
        from .observation.validation_io import run_validation_study, verify_validation_bundle
        if args.model:
            with MediaPipeProvider(args.model, args.model_sha256) as provider:
                target = run_validation_study(args.study, args.out, provider=provider, alpha=args.alpha)
        else:
            target = run_validation_study(args.study, args.out, alpha=args.alpha)
        return {"bundle": str(target), **verify_validation_bundle(target)}
    if args.command == "validation-lab":
        from .observation.validation_lab import run_localization_laboratory
        return run_localization_laboratory(args.out, seed=args.seed)
    if args.command == "observer-pin":
        from hashlib import sha256
        from importlib.metadata import version, PackageNotFoundError
        from .observation.study import ObserverDeclaration
        model = Path(args.model)
        if not model.is_file():
            raise ProviderUnavailable("supply an existing local model; no implicit download")
        try:
            runtime = args.runtime_version or version("mediapipe")
        except PackageNotFoundError:
            raise ProviderUnavailable("MediaPipe is absent; specify the planned --runtime-version explicitly") from None
        declaration = ObserverDeclaration.mediapipe(sha256(model.read_bytes()).hexdigest(), runtime,
            noise_floor_px=args.sigma, augment=not args.no_augment)
        write_json_new(args.out, declaration.to_dict())
        return {"declaration": args.out, "model_sha256": declaration.model_sha256,
                "runtime_version": runtime, "inference_executed": False, "model_validated": False}
    if args.command == "study-freeze":
        from .observation.study import freeze_study
        from .observation.enrollment import read_registry
        study = freeze_study(args.study, args.out,
                             frozen_model=read_json(Path(args.frozen_model)) if args.frozen_model else None,
                             enrollment_registry=read_registry(args.registry) if args.registry is not None else None)
        return {"study": str(study.root), "sha256": study.digest, "enrolled": len(study.cases),
                "eligible": len(study.included), "inference_executed": False}
    if args.command == "study-run":
        from .observation.study import load_study
        from .observation.study_runner import NativeExecutor, run_session
        study = load_study(args.study)
        status = run_session(study.root, args.out, NativeExecutor(args.model, timeout_seconds=study.data["timeout_seconds"]),
                             maximum_new_cases=args.max_cases)
        return _session_summary(status, args.out)
    if args.command == "study-status":
        from .observation.study_runner import verify_session
        status = verify_session(args.session)
        return status if args.raw else _session_summary(status, args.session)
    if args.command == "study-finalize":
        from .observation.study_io import finalize_session, verify_qualification_bundle
        target = finalize_session(args.session, args.out)
        return {"bundle": str(target), **verify_qualification_bundle(target)}
    if args.command == "stratified-lab":
        from .observation.stratified_lab import run_stratified_laboratory
        return run_stratified_laboratory(args.out, seed=args.seed)
    if args.command == "joint-lab":
        from .observation.joint_lab import run_joint_laboratory
        return run_joint_laboratory(args.out, seed=args.seed)
    if args.command == "intrinsic-lab":
        from .observation.intrinsic_lab import run_intrinsic_laboratory
        return run_intrinsic_laboratory(args.out, seed=args.seed)
    if args.command == "study-lab":
        from .observation.study_lab import run_study_laboratory
        result = run_study_laboratory(args.out, seed=args.seed)
        if not all(result["checks"].values()):
            raise RuntimeError("prospective laboratory control failed")
        return result
    if args.command == "doctor":
        import importlib.metadata
        from hashlib import sha256
        from .observation.runtime import OPENCV_DISTRIBUTIONS
        packages = {}
        for name in ("facemath", "mediapipe", "numpy", "scipy", *OPENCV_DISTRIBUTIONS):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        model = Path(args.model) if args.model else None
        data = {"format": "facemath.runtime_check.v4", "packages": packages,
                "numeric_environment": numeric_environment(),
                "model_present": bool(model and model.is_file()),
                "model_sha256": sha256(model.read_bytes()).hexdigest() if model and model.is_file() else None,
                "inference_executed": False,
                "live_validation": "not established by availability checks",
                "network_policy": "FACEMATH never implicitly downloads a model; third-party runtime telemetry is not attested"}
        if args.out:
            write_json_new(args.out, data)
        return data
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in ("compare", "sequence"):
        print("Legacy Wave 2 route. Use analyze / inspect-video for the observation-first instrument.", file=sys.stderr)
        from .legacy_cli import main as legacy_main
        legacy_main(arguments)
        return
    args = build_parser().parse_args(arguments)
    try:
        result = execute(args)
    except (ObservationError, ProviderUnavailable, OSError, KeyError, json.JSONDecodeError) as exc:
        if hasattr(exc, "failure_path"):
            print(f"Retained failed run: {exc.failure_path}", file=sys.stderr)
        if hasattr(exc, "journal_jsonl") and getattr(args, "out", None):
            journal_path = Path(str(args.out) + ".failure.events.jsonl")
            try:
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                with journal_path.open("x", encoding="utf-8") as stream:
                    stream.write(exc.journal_jsonl)
                print(f"Actual partial journal: {journal_path}", file=sys.stderr)
            except OSError as journal_error:
                print(f"Journal was not overwritten: {journal_error}", file=sys.stderr)
        print(f"FACEMATH: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.command == "overview" and not args.raw:
        from .observation.readout import terminal_readout
        print(terminal_readout(result))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "runtime-profile" and not result.get("comparison", {}).get("compatible", True):
        raise SystemExit(2)
    if args.command == "enrollment" and args.enrollment_command == "check" and not result["disjoint"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
