"""Wave 3 CLI: observe, analyze, inspect, verify, replay, and calibrate offline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

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


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ObservationError(f"cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        eye_span_px=args.size * 0.3125,
        uncertainty_draws=args.draws,
        seed=args.seed,
        tolerance_ipd=args.tolerance,
        enable_flow=not args.no_flow,
        enable_edges=not args.no_edges,
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--size", type=int, default=256, help="canonical canvas size, 64..1024"
    )
    parser.add_argument(
        "--draws", type=int, default=96, help="conditional sensitivity draws, 4..4096"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="landmark tolerance in registered eye spans",
    )
    parser.add_argument(
        "--no-flow",
        action="store_true",
        help="disable independent optical-flow diagnostic",
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="disable landmark-guided image edge diagnostics",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facemath",
        description="One observed anatomy. One coordinate gauge. Raw, replayable evidence.",
    )
    parser.add_argument("--version", action="version", version="facemath 0.4.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Wave 3 integrated pair instrument")
    p.add_argument("image_a")
    p.add_argument("image_b")
    p.add_argument("--observation-a")
    p.add_argument("--observation-b")
    p.add_argument(
        "--model", help="local MediaPipe .task model when observations are not supplied"
    )
    p.add_argument("--model-sha256")
    p.add_argument(
        "--lineage-a",
        action="append",
        default=[],
        help="declared source A acquisition lineage; repeatable",
    )
    p.add_argument(
        "--lineage-b",
        action="append",
        default=[],
        help="declared source B acquisition lineage; repeatable",
    )
    p.add_argument(
        "--localization-calibration",
        help="independent reference error calibration JSON",
    )
    p.add_argument("--query-groups", nargs=2, metavar=("GROUP_A", "GROUP_B"))
    p.add_argument(
        "--domain", default="", help="calibration acquisition domain, not inferred"
    )
    p.add_argument("--out", required=True, help="new evidence bundle directory")
    _add_config(p)

    p = sub.add_parser(
        "observe", help="local MediaPipe observation with augmentation repeatability"
    )
    p.add_argument("image")
    p.add_argument("--model", required=True)
    p.add_argument("--model-sha256")
    p.add_argument("--out", required=True)
    p.add_argument("--sigma", type=float, default=0.75)

    p = sub.add_parser(
        "import-annotation", help="bind explicit semantic21 JSON to source pixels"
    )
    p.add_argument("image")
    p.add_argument("annotation")
    p.add_argument("--annotation-id", required=True)
    p.add_argument("--sigma", required=True, type=float)
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "locate", help="offline coarse face ROI proposals; not anatomical landmarks"
    )
    p.add_argument("image")
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "inspect-video",
        help="time-aware editing/mix diagnostics; never a similarity score",
    )
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))

    p = sub.add_parser(
        "demo", help="known-geometry synthetic bundle, no detector models needed"
    )
    p.add_argument("--out", required=True)
    _add_config(p)

    p = sub.add_parser("laboratory", help="execute quantitative falsification controls")
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "verify", help="verify byte hashes, array contracts, and execution journal"
    )
    p.add_argument("bundle")

    p = sub.add_parser(
        "replay", help="recompute from stored source rasters and observations"
    )
    p.add_argument("bundle")
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "capabilities",
        help="audit original mappings without claiming mathematical equivalence",
    )
    p.add_argument("--out")

    p = sub.add_parser(
        "cohort-create",
        help="ingest verified, compatible run bundles as an explicit reference",
    )
    p.add_argument("bundles", nargs="+")
    p.add_argument("--name", required=True)
    p.add_argument("--kind", required=True, choices=("synthetic", "observational"))
    p.add_argument("--metrics", nargs="+", default=["geometry.all.rms"])
    p.add_argument(
        "--subjects", help="optional JSON mapping image SHA-256 to declared subject ID"
    )
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "calibrate", help="one measured metric against an exact-recipe reference cohort"
    )
    p.add_argument("bundle")
    p.add_argument("--cohort", required=True)
    p.add_argument("--metric", default="geometry.all.rms")
    p.add_argument("--out", required=True)
    p.add_argument("--higher-is-closer", action="store_true")
    p.add_argument("--query-subjects", nargs="*", default=[])
    p = sub.add_parser(
        "annotate",
        help="create offline prediction-blind eight-point reference workbench",
    )
    p.add_argument("image")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True, help="new standalone HTML file")

    p = sub.add_parser(
        "import-ibug-reference",
        help="import reviewed eight-point subset of a 68-point PTS annotation",
    )
    p.add_argument("image")
    p.add_argument("pts")
    p.add_argument("--origin", required=True, choices=(0, 1), type=int)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "validate-observer",
        help="run source-coordinate evaluation with independent reference labels",
    )
    p.add_argument(
        "study",
        help="local study manifest with development/calibration/evaluation groups",
    )
    p.add_argument(
        "--model", help="local model, needed only when saved observations are absent"
    )
    p.add_argument("--model-sha256")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", required=True)

    p = sub.add_parser(
        "validation-lab",
        help="execute synthetic stable-bias localization controls; not human accuracy",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=24091)

    p = sub.add_parser(
        "doctor",
        help="report runtime/model availability; never claims inference was run",
    )
    p.add_argument("--model")
    p.add_argument("--out")
    return parser


def _analyze(args: argparse.Namespace) -> dict:
    image_a, image_b = load_rgb(args.image_a), load_rgb(args.image_b)
    obs_a = (
        Observation.from_dict(read_json(args.observation_a))
        if args.observation_a
        else None
    )
    obs_b = (
        Observation.from_dict(read_json(args.observation_b))
        if args.observation_b
        else None
    )
    config = _configuration(args)
    calibration = None
    if args.localization_calibration:
        from .observation.error_model import LocalizationCalibration

        calibration = LocalizationCalibration.from_dict(
            read_json(args.localization_calibration)
        )
        if not args.query_groups or not args.domain:
            raise ObservationError(
                "calibrated analysis requires --query-groups and --domain"
            )
    extra = dict(
        localization_calibration=calibration,
        query_groups=tuple(args.query_groups or ("", "")),
        domain=args.domain,
        query_lineages=(tuple(args.lineage_a), tuple(args.lineage_b)),
    )
    if obs_a is not None and obs_b is not None:
        result = FacialInstrument(config, **extra).run(image_a, image_b, obs_a, obs_b)
    else:
        if not args.model:
            raise ObservationError(
                "supply both observation files or --model; no fabricated anatomical fallback"
            )
        with MediaPipeProvider(args.model, args.model_sha256) as provider:
            result = FacialInstrument(config, provider, **extra).run(
                image_a, image_b, obs_a, obs_b
            )
    output = write_bundle(result, args.out, image_a, image_b)
    return {
        "bundle": str(output),
        "report": str(output / "report.html"),
        **verify_bundle(output),
    }


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
        cell[10 : 10 + thumb.shape[0], 10 : 10 + thumb.shape[1]] = thumb
        label = (
            f"frame {record['frame_index']} | {record['timestamp_cfr_seconds']:.3f}s"
        )
        cv2.putText(
            cell,
            label,
            (10, 281),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cells.append(cell)
    while len(cells) % 4:
        cells.append(np.full_like(cells[0], 245))
    sheet = np.vstack([np.hstack(cells[i : i + 4]) for i in range(0, len(cells), 4)])
    encoded = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(target / "contact.png"), encoded):
        raise ObservationError("failed to write video inspection contact sheet")
    return {
        "directory": str(target),
        "frames": len(frames),
        "facial_similarity_evidence": False,
    }


def _replay(args: argparse.Namespace) -> dict:
    root = Path(args.bundle)
    if (
        read_json(root / "manifest.json").get("format")
        == "facemath.validation_bundle.v4"
    ):
        from .observation.validation_io import replay_validation_bundle

        return replay_validation_bundle(root, args.out)
    verify_bundle(root)
    env = read_json(root / "environment.json")
    if env["engine_code_sha256"] != code_fingerprint():
        raise ObservationError(
            "source implementation differs; perform a new analysis, not an exact replay"
        )
    for name, version in numeric_environment().items():
        previous = env["numeric_environment"][name]
        if version != previous:
            raise ObservationError(
                f"numeric runtime differs for {name}: {previous} vs {version}"
            )
    evidence = read_json(root / "evidence.json")
    observations = read_json(root / "observations.json")
    allowed = {f.name for f in fields(InstrumentConfig)}
    config = InstrumentConfig(
        **{k: v for k, v in evidence["configuration"].items() if k in allowed}
    )
    image_a, image_b = load_rgb(root / "source_a.png"), load_rgb(root / "source_b.png")
    calibration = None
    envelope = evidence.get("localization_envelope")
    if envelope:
        from .observation.error_model import LocalizationCalibration

        calibration = LocalizationCalibration.from_dict(
            read_json(root / "localization_calibration.json")
        )
        if calibration.digest != envelope["calibration_sha256"]:
            raise ObservationError("localization calibration binding mismatch")
    result = FacialInstrument(
        config,
        localization_calibration=calibration,
        query_groups=tuple(envelope["query_groups"]) if envelope else ("", ""),
        query_lineages=tuple(tuple(x) for x in envelope.get("query_lineages", ((), ())))
        if envelope
        else ((), ()),
        domain=envelope["domain"] if envelope else "",
    ).run(
        image_a,
        image_b,
        Observation.from_dict(observations["a"]),
        Observation.from_dict(observations["b"]),
    )
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
            if saved[name].shape != array.shape or not np.allclose(
                saved[name], array, atol=1e-9, rtol=1e-7
            ):
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
    evidence, observations = (
        read_json(root / "evidence.json"),
        read_json(root / "observations.json"),
    )
    cohort = Cohort.from_dict(read_json(args.cohort))
    value = next(
        (
            m["value"]
            for m in evidence["measurement_profile"]
            if m["name"] == args.metric
        ),
        None,
    )
    if value is None:
        raise ObservationError("metric missing or withheld in query evidence")
    data = cohort.calibrate(
        args.metric,
        value,
        evidence["recipe_sha256"],
        (observations["a"]["image_sha256"], observations["b"]["image_sha256"]),
        lower_is_closer=not args.higher_is_closer,
        query_subjects=tuple(args.query_subjects),
    )
    write_json_new(args.out, data)
    return data


def execute(args: argparse.Namespace) -> dict:
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "observe":
        with MediaPipeProvider(args.model, args.model_sha256, args.sigma) as provider:
            observation = provider.observe(load_rgb(args.image))
        write_json_new(args.out, observation.to_dict())
        return {
            "observation": args.out,
            "landmarks": len(observation.xy),
            "samples": len(observation.samples),
        }
    if args.command == "import-annotation":
        from .observation.anatomy import SEMANTIC_21

        obj = read_json(args.annotation)
        observation = annotated_observation(
            load_rgb(args.image),
            np.asarray(obj["xy"]),
            annotation_id=args.annotation_id,
            sigma_px=args.sigma,
            names=tuple(obj.get("names", SEMANTIC_21)),
            source_notes=tuple(obj.get("notes", [])),
        )
        write_json_new(args.out, observation.to_dict())
        return {"observation": args.out, "provider": observation.provider}
    if args.command == "locate":
        data = {
            "kind": "coarse_ROI_proposals_not_landmarks",
            "regions": locate_faces(load_rgb(args.image)),
        }
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
            image_a,
            image_b,
            phantom_observation(image_a, points_a),
            phantom_observation(image_b, points_b),
        )
        output = write_bundle(result, args.out, image_a, image_b)
        return {
            "bundle": str(output),
            "kind": "synthetic_known_geometry_not_human_data",
            **verify_bundle(output),
        }
    if args.command == "laboratory":
        data = laboratory()
        write_json_new(args.out, data)
        if not all(data["checks"].values()):
            raise RuntimeError("laboratory falsification check failed")
        return {"laboratory": args.out, "checks": data["checks"]}
    if args.command == "verify":
        if (
            read_json(Path(args.bundle) / "manifest.json").get("format")
            == "facemath.validation_bundle.v4"
        ):
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
            args.name,
            args.bundles,
            kind=args.kind,
            metrics=tuple(args.metrics),
            subjects_by_image=read_json(args.subjects) if args.subjects else None,
        )
        write_json_new(args.out, cohort.to_dict())
        return {
            "cohort": args.out,
            "kind": cohort.kind,
            "rows": len(cohort.rows),
            "sha256": cohort.fingerprint,
            "dependence": cohort.dependence(),
        }
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "annotate":
        from .observation.annotation_ui import create_annotation_workspace

        path = create_annotation_workspace(
            load_rgb(args.image), args.out, source=args.source
        )
        return {"workbench": str(path), "predictions_shown": False, "labels_created": 0}
    if args.command == "import-ibug-reference":
        from .observation.contracts import array_digest
        from .observation.reference import import_ibug68

        image = load_rgb(args.image)
        ref = import_ibug68(
            Path(args.pts).read_text(encoding="utf-8"),
            image_sha256=array_digest(image),
            image_shape=image.shape[:2],
            coordinate_origin=args.origin,
            source=args.source,
        )
        write_json_new(args.out, ref.to_dict())
        return {"reference": args.out, "points": 8, "mapping_requires_review": True}
    if args.command == "validate-observer":
        from .observation.validation_io import (
            run_validation_study,
            verify_validation_bundle,
        )

        if args.model:
            with MediaPipeProvider(args.model, args.model_sha256) as provider:
                target = run_validation_study(
                    args.study, args.out, provider=provider, alpha=args.alpha
                )
        else:
            target = run_validation_study(args.study, args.out, alpha=args.alpha)
        return {"bundle": str(target), **verify_validation_bundle(target)}
    if args.command == "validation-lab":
        from .observation.validation_lab import run_localization_laboratory

        return run_localization_laboratory(args.out, seed=args.seed)
    if args.command == "doctor":
        import importlib.metadata
        from hashlib import sha256

        packages = {}
        for name in ("facemath", "mediapipe", "numpy", "scipy", "opencv-python"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        model = Path(args.model) if args.model else None
        data = {
            "format": "facemath.runtime_check.v4",
            "packages": packages,
            "model_present": bool(model and model.is_file()),
            "model_sha256": sha256(model.read_bytes()).hexdigest()
            if model and model.is_file()
            else None,
            "inference_executed": False,
            "live_validation": "not established by availability checks",
            "network_policy": "FACEMATH never implicitly downloads a model; third-party runtime telemetry is not attested",
        }
        if args.out:
            write_json_new(args.out, data)
        return data
    raise AssertionError(args.command)


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in ("compare", "sequence"):
        print(
            "Legacy Wave 2 route. Use analyze / inspect-video for the observation-first instrument.",
            file=sys.stderr,
        )
        from .legacy_cli import main as legacy_main

        legacy_main(arguments)
        return
    args = build_parser().parse_args(arguments)
    try:
        result = execute(args)
    except (
        ObservationError,
        ProviderUnavailable,
        OSError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FACEMATH: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
