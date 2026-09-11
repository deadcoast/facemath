"""A falsification laboratory for localization uncertainty, not human accuracy.

The synthetic observer is intentionally very repeatable and systematically biased.
References come from independent known render coordinates. Failure cases are fixed
before evaluation. Both calibration and evaluation flow through production APIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .anatomy import SEMANTIC_21
from .annotation_ui import create_annotation_workspace
from .artifacts import verify_bundle, write_bundle
from .canonical import InstrumentConfig
from .contracts import Observation, array_digest, canonical_json, object_digest
from .error_model import LocalizationCalibration
from .instrument import FacialInstrument
from .lab import phantom_points, render_phantom
from .reference import REFERENCE_NAMES, ReferenceAnnotation
from .validation_io import run_validation_study, verify_validation_bundle

DOMAIN = "schematic-biased-observer-controls-v1-NOT-HUMAN"
PROVIDER_SPEC = {
    "provider": "synthetic_biased_observer",
    "version": 1,
    "shared_bias_px": [1.3, -0.9],
    "case_shared_jitter_sd_px": 0.45,
    "local_bias_sd_px": 0.40,
    "perturbation_jitter_sd_px": 0.015,
    "attempted_samples": 7,
    "assumed_noise_floor_px": 0.25,
    "calibration_failure_indices": [17],
    "evaluation_failure_indices": [49],
    "scope": "deterministic falsification fixture, not learned face inference",
}


def _write(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_localization_laboratory(destination: str | Path, *, seed: int = 24091) -> dict:
    """Build forty calibration groups and twenty separate evaluation groups."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    assets = root / "inputs"
    assets.mkdir()
    rng = np.random.default_rng(seed)
    entries, saved, sample_dispersion = [], [], []
    for i in range(60):
        # Broad shape changes and source texture vary independently of observer bias.
        truth = phantom_points() + rng.normal(0, 0.65, (21, 2))
        rgb = render_phantom(truth, seed=seed + i)
        xy = (
            truth
            + np.array([1.3, -0.9])
            + rng.normal(0, 0.45, (1, 2))
            + rng.normal(0, 0.40, truth.shape)
        )
        samples = xy[None] + rng.normal(0, 0.015, (7, 21, 2))
        samples[0] = xy
        obs = Observation(
            array_digest(rgb),
            rgb.shape[:2],
            "semantic21.v1",
            SEMANTIC_21,
            xy,
            samples,
            "synthetic_biased_observer",
            object_digest(PROVIDER_SPEC),
            0.25,
            7,
            tuple(f"scripted_perturbation_{k}" for k in range(7)),
            (
                "synthetic predictions with explicit systematic error; not real detector output",
            ),
            canonical_json(PROVIDER_SPEC),
        )
        ids = [SEMANTIC_21.index(name) for name in REFERENCE_NAMES]
        reference = ReferenceAnnotation(
            obs.image_sha256,
            obs.image_shape,
            truth[ids],
            np.ones(8, bool),
            "known-render-coordinates",
            "synthetic_fixture",
            "analytically specified schematic control coordinates",
        )
        base = f"control_{i:03d}"
        cv2.imwrite(str(assets / f"{base}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        _write(assets / f"{base}_reference.json", reference.to_dict())
        split = "calibration" if i < 40 else "evaluation"
        entry = dict(
            case_id=base,
            group_id=f"independent_synthetic_draw_{i:03d}",
            split=split,
            lineage_ids=[f"renderer_draw_{seed}_{i}"],
            image=f"{base}.png",
            references=[f"{base}_reference.json"],
        )
        if i in (17, 49):
            entry["failure"] = (
                "prespecified synthetic nondetection; no model was invoked"
            )
        else:
            _write(assets / f"{base}_observation.json", obs.to_dict())
            entry["observation"] = f"{base}_observation.json"
            saved.append((i, rgb, obs, truth, entry))
        entries.append(entry)
        sample_dispersion.append(
            float(np.sqrt(np.mean(np.sum((samples - samples.mean(0)) ** 2, axis=2))))
        )
    spec = dict(
        format="facemath.validation_study.v4",
        domain=DOMAIN,
        minimum_raters=1,
        cases=entries,
    )
    _write(assets / "study.json", spec)
    validation = run_validation_study(
        assets / "study.json", root / "validation", alpha=0.1
    )
    report = json.loads((validation / "validation.json").read_text())
    coverage = json.loads((validation / "coverage.json").read_text())
    calibration = LocalizationCalibration.from_dict(
        json.loads((validation / "localization_calibration.json").read_text())
    )
    evaluation = [r for r in report["cases"] if r["split"] == "evaluation"]
    successful = [r for r in evaluation if r["simultaneous_score"] is not None]
    # A naive 3-sigma ellipse is a deliberate failure control, not a nominal
    # simultaneous coverage claim for eight points or correlated observations.
    naive_covered = sum(
        r["simultaneous_score"] is not None and r["simultaneous_score"] <= 3
        for r in evaluation
    )
    mean_error = float(np.mean([np.mean(r["error_norm_px"]) for r in successful]))
    query = [x for x in saved if x[0] >= 40][:2]
    (_, a, oa, ta, ea), (_, b, ob, tb, eb) = query
    result = FacialInstrument(
        InstrumentConfig(uncertainty_draws=48, enable_flow=True, edge_samples=32),
        localization_calibration=calibration,
        query_groups=(ea["group_id"], eb["group_id"]),
        query_lineages=(tuple(ea["lineage_ids"]), tuple(eb["lineage_ids"])),
        domain=DOMAIN,
    ).run(a, b, oa, ob)
    write_bundle(result, root / "qualified_pair", a, b)
    from .provider import apply_points

    ids = [SEMANTIC_21.index(name) for name in REFERENCE_NAMES]
    gap = (
        np.linalg.norm(
            apply_points(ta[ids], result.state.source_to_canonical_a)
            - apply_points(tb[ids], result.state.source_to_canonical_b),
            axis=1,
        )
        / result.state.normalization_span_px
    )
    envelope = result.localization
    create_annotation_workspace(
        a,
        root / "reference_workbench.html",
        source="SYNTHETIC browser control; no human accuracy labels",
        kind="synthetic_fixture",
    )
    summary = {
        "format": "facemath.localization_lab.v4",
        "kind": "synthetic_falsification_NOT_HUMAN_DETECTOR_VALIDATION",
        "seed": seed,
        "provider_spec": PROVIDER_SPEC,
        "cases": 60,
        "calibration_groups": 40,
        "evaluation_groups": 20,
        "mean_perturbation_rms_px": float(np.mean(sample_dispersion)),
        "heldout_mean_localization_error_px": mean_error,
        "bias_to_repeatability_ratio": mean_error / float(np.mean(sample_dispersion)),
        "naive_3sigma_simultaneous_hit_fraction_including_failures": naive_covered
        / len(evaluation),
        "naive_rule_is_not_a_coverage_guarantee": True,
        "calibrated_empirical_group_coverage": coverage["empirical_group_coverage"],
        "conformal_rank": calibration.rank,
        "conformal_quantile": calibration.quantile,
        "calibration_failed_groups": sum(
            s is None for _, s in calibration.group_scores
        ),
        "evaluation_failed_cases": coverage["failures"],
        "known_fixed_gauge_gap_ipd": gap.tolist(),
        "fixed_gauge_bounds_contain_rendered_reference": bool(
            np.all(gap >= envelope.lower_ipd - 1e-12)
            and np.all(gap <= envelope.upper_ipd + 1e-12)
        ),
        "pair_localization_statuses": [r["status"] for r in envelope.to_dict()["rows"]],
        "verification": {
            "validation": verify_validation_bundle(validation),
            "qualified_pair": verify_bundle(root / "qualified_pair"),
        },
        "limit": "a controlled detector-error model validates this workflow, not MediaPipe or human facial localization",
    }
    _write(root / "laboratory.json", summary)
    return summary
