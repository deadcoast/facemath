"""Integrated, explicitly synthetic heterogeneous-acquisition falsification study.

This uses the existing frozen-study runner, source-coordinate evaluator, shared
joint fit, pair instrument and artifact writer. It does not substitute for a live
landmark detector or independent human annotations. The seed and acquisition
schedule are fixed before observing coverage or query intervals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .anatomy import SEMANTIC_21
from .artifacts import write_bundle, verify_bundle
from .bias import load_localization_calibration, ocular_basis
from .canonical import InstrumentConfig
from .contracts import Observation, ObservationError, array_digest, canonical_json, object_digest
from .instrument import FacialInstrument, InstrumentFailure
from .intrinsic import nominal_values, relation_registry
from .intrinsic_lab import _so2_minimum
from .joint_lab import _record as geometry_record, INDICES, BETA
from .reference import ReferenceAnnotation
from .stratified import MODE, StratifiedJointCalibration
from .study import ObserverDeclaration, freeze_study, load_rgb, read_json, save_rgb, write_json_new
from .study_io import finalize_session, verify_qualification_bundle
from .study_runner import AttemptOutcome, run_session

DOMAIN = "synthetic-heterogeneous-acquisition-v8-NOT-HUMAN"
CATEGORIES = ("clear", "degraded", "sparse", "unseen")
SPEC = {"name": "synthetic_acquisition_stratified_observer", "version": 8,
    "common_translation_std_eye_spans": .020,
    "local_std_eye_spans": {"clear": .0008, "degraded": .008, "sparse": .003, "unseen": .014},
    "hidden_shift_eye_spans": [.055, .065], "bias": BETA.tolist(), "sample_jitter_px": .015,
    "noise_floor_px": .18, "kind": "scripted error process, NOT real model predictions",
    "acquisition_categories": "clear=no blur; degraded=fixed Gaussian blur; sparse/unseen=declared experimental acquisition configurations",
    "hidden_shift": "same clear image condition with deliberately shifted observer errors, absent from calibration"}
DECLARATION = ObserverDeclaration(SPEC["name"], object_digest(SPEC), "semantic21.v1", .18, 7, "synthetic_fixture")


def observation(record: dict[str, Any]) -> Observation:
    image = record["image"]
    return Observation(array_digest(image), image.shape[:2], "semantic21.v1", SEMANTIC_21,
        record["prediction"], record["samples"], DECLARATION.provider, DECLARATION.provider_fingerprint,
        .18, 7, tuple(f"synthetic_jitter_{i}" for i in range(7)),
        ("synthetic observer with known imposed errors; not a human or model observation",), canonical_json(SPEC))


def record(seed: int, category: str, *, expansion: float = 0., hidden_shift: bool = False) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise ObservationError("unknown synthetic acquisition category")
    base = geometry_record(seed, expansion=expansion)
    truth = base["truth"]
    image = base["image"]
    if category == "degraded":
        image = cv2.GaussianBlur(image, (7, 7), 1.6)
    rng = np.random.default_rng(seed+880000)
    ocular = ocular_basis(truth[INDICES])
    common = rng.normal(0, SPEC["common_translation_std_eye_spans"], (1, 2))
    local = rng.normal(0, SPEC["local_std_eye_spans"][category], (8, 2))
    latent = BETA+common+local
    if hidden_shift:
        latent[4:] += SPEC["hidden_shift_eye_spans"]
    prediction = truth+common@ocular.T
    prediction[INDICES] = truth[INDICES]+latent@ocular.T
    samples = prediction[None]+rng.normal(0, .015, (7, 21, 2)); samples[0] = prediction
    return {"truth": truth, "image": image, "prediction": prediction, "samples": samples,
            "category": category, "latent_errors": latent, "hidden_shift": hidden_shift, "failure": False}


@dataclass
class StratifiedFixtureExecutor:
    records: dict[str, dict[str, Any]]
    calls: int = 0

    def preflight(self, declaration: ObserverDeclaration) -> dict[str, Any]:
        if declaration != DECLARATION:
            raise ObservationError("synthetic stratified executor recipe mismatch")
        return {"ready": True, "kind": "synthetic_fixture", "live_inference_executed": False,
                "spec_sha256": object_digest(SPEC)}

    def execute(self, image_path: Path, declaration: ObserverDeclaration, directory: Path) -> AttemptOutcome:
        self.calls += 1
        source = load_rgb(image_path)
        rec = self.records[array_digest(source)]
        if rec["failure"]:
            return AttemptOutcome("no_face", detail="prespecified synthetic failure; not native inference")
        return AttemptOutcome("observed", observation(rec))


def make_stratified_study(root: Path, *, seed: int = 28091, compact: bool = False
                           ) -> tuple[Path, StratifiedFixtureExecutor]:
    root.mkdir(parents=True, exist_ok=False)
    if compact:
        development = 16
        cal_counts = {"clear": 20, "degraded": 12, "sparse": 2, "unseen": 0}
        eval_counts = {"clear": 6, "degraded": 4, "sparse": 1, "unseen": 1}
        shifted_clear = 2
    else:
        development = 32
        cal_counts = {"clear": 60, "degraded": 20, "sparse": 4, "unseen": 0}
        eval_counts = {"clear": 20, "degraded": 12, "sparse": 3, "unseen": 3}
        shifted_clear = 8
    schedule = [("development", "clear" if i % 2 else "degraded", False) for i in range(development)]
    for split, counts in (("calibration", cal_counts), ("evaluation", eval_counts)):
        for category, n in counts.items():
            for j in range(n):
                shift = split == "evaluation" and category == "clear" and j >= n-shifted_clear
                schedule.append((split, category, shift))
    entries, records, ground_truth, local_errors, shifts = [], {}, [], [], []
    local_counts: dict[tuple[str, str], int] = {}
    for i, (split, category, shift) in enumerate(schedule+[("evaluation", "clear", False)]):
        r = record(seed+i, category, hidden_shift=shift)
        counter = local_counts.get((split, category), 0)
        # One calibration and one evaluation nondetection, fixed before fitting.
        r["failure"] = (split == "calibration" and category == "clear" and counter == 3) or (
                        split == "evaluation" and category == "degraded" and counter == 2)
        local_counts[(split, category)] = counter+1
        image_name, reference_name = f"source_{i:04d}.png", f"source_{i:04d}.reference.json"
        save_rgb(root/image_name, r["image"])
        ref = ReferenceAnnotation(array_digest(r["image"]), r["image"].shape[:2], r["truth"][INDICES],
            np.ones(8, bool), "known-render-coordinates", "synthetic_fixture",
            "known synthetic render geometry, not a detector-generated human reference")
        write_json_new(root/reference_name, ref.to_dict())
        case = {"case_id": f"acquisition_{i:04d}", "group_id": f"group_{seed}_{i}", "split": split,
                "stratum": category, "lineage_ids": [f"source_lineage_{seed}_{i}"],
                "image": image_name, "references": [reference_name]}
        if i == len(schedule):
            case.update(eligibility="excluded", exclusion_reason="prespecified enrollment control; never invoke observer")
        entries.append(case); records[array_digest(r["image"])] = r
        ground_truth.append(r["truth"][INDICES]); local_errors.append(r["latent_errors"]); shifts.append(shift)
    specification = {"format": "facemath.prospective_study.v5", "domain": DOMAIN,
        "observer": DECLARATION.to_dict(), "minimum_raters": 1, "maximum_cases_per_group": 1,
        "minimum_development_groups": 5, "alpha": .1, "bias_mode": MODE,
        "joint_error": {"shrinkage": .02, "floor_eye_spans": .001}, "strata": list(CATEGORIES),
        "stratified_error": {"minimum_calibration_groups": 9,
            "definition": SPEC["acquisition_categories"]+"; assigned before observer errors are computed"},
        "cases": entries}
    write_json_new(root/"study.json", specification)
    write_json_new(root/"fixture_contract.json", {"seed": seed, "spec": SPEC, "compact": compact,
        "schedule": schedule, "calibration_counts": cal_counts, "evaluation_counts": eval_counts,
        "hidden_shift_clear_evaluation_cases": shifted_clear,
        "query_rule": "fresh seed+10000 clear paired with seed+20000+i degraded at expansions 0,4,8,12",
        "no_selection_from_observed_results": True})
    np.savez_compressed(root/"fixture_ground_truth.npz", truth=np.stack(ground_truth),
                        imposed_errors=np.stack(local_errors), hidden_shift=np.array(shifts))
    return root/"study.json", StratifiedFixtureExecutor(records)


def run_mixed_radius_controls(destination: str | Path, *, samples: int = 1024, seed: int = 28801) -> dict[str, Any]:
    """Analytic source-radius counterexample plus retained full16 admissible draws."""
    from .joint_lab import _source, CONTROL_POINTS
    from .intrinsic_sets import SourceRelations, absolute_difference
    from .intrinsic import _bound, bound_refitted_shape
    from .intrinsic_lab import _exact_relation_batch
    from .error_model import ellipse_gap_bounds
    if type(samples) is not int or not 64 <= samples <= 100000:
        raise ObservationError("mixed-radius controls require 64..100000 draws")
    root = Path(destination); root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    common = np.tile(np.eye(2), (8, 1))*.04
    local = rng.normal(0, .001, (16, 16))
    covariance = common@common.T+local@local.T+np.eye(16)*.001**2
    b = CONTROL_POINTS.copy(); b[5, 0] -= .10; b[6, 0] += .10
    sources = (_source(CONTROL_POINTS, covariance, .7), _source(b, covariance*1.2, 2.4))
    evaluators = tuple(SourceRelations(s) for s in sources)
    relations = relation_registry(InstrumentConfig())
    intervals = [[_bound(e, r) for r in relations] for e in evaluators]
    gaps = [absolute_difference(a, b) for a, b in zip(*intervals)]
    lower = np.array([g.lower for g in gaps]); upper = np.array([g.upper for g in gaps])
    available = np.array([g.available for g in gaps])
    shape = bound_refitted_shape(np.stack([s.centers for s in sources]), evaluators, lower, True)
    draws, latent = [], []
    for source in sources:
        z = rng.normal(size=(samples, 16)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        radius = np.ones(samples); radius[samples//2:] = rng.uniform(size=samples-samples//2)**(1/16)
        z *= radius[:, None]; latent.append(z)
        draws.append(source.centers+(source.quantile*z@source.joint_factor.reshape(16, 16).T).reshape(samples, 8, 2))
    values = [_exact_relation_batch(x, relations) for x in draws]
    actual = abs(values[0]-values[1]); refits = np.array([_so2_minimum(a,b) for a,b in zip(*draws)])
    delta = np.array([[.5,0.]]); circles = np.array([[np.eye(2)*.01],[np.eye(2)*.04]])
    correct = ellipse_gap_bounds(delta, circles, [1.,3.])
    smaller = ellipse_gap_bounds(delta, circles, 1.)
    maximum = ellipse_gap_bounds(delta, circles, 3.)
    witness = np.array([1.2,0.])
    arrays = dict(source_centers=np.stack([s.centers for s in sources]),
        source_covariance=np.stack([s.joint_covariance for s in sources]),
        source_factors=np.stack([s.joint_factor for s in sources]), source_radii=np.array([.7,2.4]),
        latent_draws=np.stack(latent), shape_draws=np.stack(draws), exact_relation_values=np.stack(values),
        exact_relation_gaps=actual, gap_lower=lower, gap_upper=upper, gap_available=available,
        exact_refitted_shape=refits, shape_bounds=np.array([shape.lower,shape.upper]),
        circle_delta=delta, circle_covariances=circles, circle_source_radii=np.array([1.,3.]), circle_witness=witness)
    np.savez_compressed(root/'controls.npz', **arrays)
    result = {"format":"facemath.mixed_source_radius_controls.v8", "kind":"analytic_and_synthetic_NOT_HUMAN",
        "seed":seed, "sampled_shape_pairs":samples, "scalar_relation_checks":int(actual[:,available].size),
        "sampled_relation_violations":int(np.sum(available & ((actual<lower-1e-10)|(actual>upper+1e-10)))),
        "sampled_shape_violations":int(np.sum((refits<shape.lower-1e-10)|(refits>shape.upper+1e-10))),
        "shape_bound":shape.to_dict(), "circle_counterexample":{"source_radii":[1.,3.],
            "admissible_witness_gap":float(np.linalg.norm(witness)),
            "correct_source_specific_upper":float(correct.upper_ipd[0]),
            "unsafe_smaller_single_radius_upper":float(smaller.upper_ipd[0]),
            "overconservative_larger_single_radius_upper":float(maximum.upper_ipd[0])},
        "arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"sha256":array_digest(v)} for k,v in arrays.items()},
        "limit":"sampling tests the analytic outer bounds; it is not proof by sampling or real reference calibration"}
    write_json_new(root/'controls.json',result)
    return result


def run_stratified_laboratory(destination: str | Path, *, seed: int = 28091) -> dict[str, Any]:
    root = Path(destination); root.mkdir(parents=True, exist_ok=False)
    set_controls = run_mixed_radius_controls(root/"set_controls")
    path, executor = make_stratified_study(root/"inputs", seed=seed)
    frozen = freeze_study(path, root/"frozen")
    run_session(frozen.root, root/"acquisition", executor, maximum_new_cases=7); first = executor.calls
    status = run_session(frozen.root, root/"acquisition", executor); complete = executor.calls
    run_session(frozen.root, root/"acquisition", executor)
    qualification = finalize_session(root/"acquisition", root/"qualification")
    model = load_localization_calibration(read_json(qualification/"localization_calibration.json"))
    if not isinstance(model, StratifiedJointCalibration):
        raise ObservationError("laboratory did not produce a stratified joint error model")
    cfg = InstrumentConfig(canvas_size=224, eye_span_px=70, uncertainty_draws=16,
                           enable_flow=False, edge_samples=16)
    a = record(seed+10000, "clear"); oa = observation(a)
    specs = relation_registry(cfg); summaries = []; truths = []
    for i, expansion in enumerate((0., 4., 8., 12.)):
        b = record(seed+20000+i, "degraded", expansion=expansion); ob = observation(b)
        groups, lineages = (f"query_a_{seed}", f"query_b_{seed}_{i}"), ((f"new_source_a_{seed}",), (f"new_source_b_{seed}_{i}",))
        instrument = FacialInstrument(cfg, localization_calibration=model, domain=DOMAIN,
                                      query_groups=groups, query_lineages=lineages, query_strata=("clear", "degraded"))
        result = instrument.run(a["image"], b["image"], oa, ob)
        bare = FacialInstrument(cfg).run(a["image"], b["image"], oa, ob)
        unchanged = all(np.array_equal(v, result.raw_geometry_arrays()[key]) for key, v in bare.raw_geometry_arrays().items())
        if not unchanged:
            raise ObservationError("stratified qualification changed the observed geometry")
        target = write_bundle(result, root/f"pair_{i:02d}_mouth_{int(expansion)}px", a["image"], b["image"])
        actual = _so2_minimum(a["truth"][INDICES], b["truth"][INDICES])
        va, ma = nominal_values(a["truth"][INDICES], np.ones(8, bool), specs)
        vb, mb = nominal_values(b["truth"][INDICES], np.ones(8, bool), specs)
        gap = abs(va-vb); p = result.intrinsic
        admissible = p.gap_available & ma & mb
        q = result.localization.base.source_quantiles
        summaries.append({"bundle": target.name, "mouth_expansion_px": expansion,
            "query_strata": ["clear", "degraded"], "source_quantiles": q.tolist(),
            "known_reference_shape_rms": actual, "shape_range": p.shape.to_dict(),
            "known_shape_enclosed": bool(p.shape.available and p.shape.lower-1e-10 <= actual <= p.shape.upper+1e-10),
            "bounded_relations": int(admissible.sum()),
            "known_relation_gap_violations": int(np.sum(admissible & ((gap < p.gap_lower-1e-10) | (gap > p.gap_upper+1e-10)))),
            "raw_arrays_unchanged": unchanged, "raw_array_count": len(bare.raw_geometry_arrays()),
            "array_count": len(result.arrays()), "verification": verify_bundle(target)})
        truths.append(np.stack([a["truth"][INDICES], b["truth"][INDICES]]))
    np.savez_compressed(root/"query_ground_truth.npz", source_reference_coordinates=np.stack(truths))
    # Preserve the actual rejected attempt; it must never appear as a passed bundle.
    b = record(seed+30000, "unseen"); ob = observation(b)
    rejected = []
    for category in ("unseen", "sparse"):
        try:
            FacialInstrument(cfg, localization_calibration=model, domain=DOMAIN,
                query_groups=("query_supported", "query_unsupported"), query_strata=("clear", category)).run(a["image"], b["image"], oa, ob)
        except InstrumentFailure as exc:
            name = "refused_"+category+".events.jsonl"
            (root/name).write_text(exc.journal_jsonl, encoding="utf-8")
            rejected.append({"stratum": category, "reason": str(exc), "actual_events": name,
                             "no_measurement_bundle_created": True})
        else:
            raise ObservationError("unsupported category obtained a calibrated pair")
    report = read_json(qualification/"qualification.json")
    # Expose hidden within-category shift separately using prespecified fixture flags.
    contract = read_json(root/"inputs/fixture_contract.json")
    hidden_ids = {f"acquisition_{i:04d}" for i, item in enumerate(contract["schedule"]) if item[2]}
    outcomes = report["acquisition_routing"]["coverage"]["cases"]
    hidden = [x for x in outcomes if x["case_id"] in hidden_ids]
    answer = {"format": "facemath.stratified_laboratory.v8", "kind": "SYNTHETIC_CONTROL_NOT_HUMAN",
        "seed": seed, "acquisition": {"enrolled": status["enrolled"], "eligible": status["eligible"],
            "calls_after_first_batch": first, "calls_after_complete": complete, "calls_after_resume": executor.calls,
            "outcomes": status["outcomes"], "excluded": status["prospectively_excluded"]},
        "category_radii": [r.to_dict() for r in model.radii],
        "selected_coverage": report["acquisition_routing"]["coverage"],
        "pooled_joint_diagnostic": report["acquisition_routing"]["pooled_joint_diagnostic"],
        "hidden_within_clear_shift": {"prespecified_groups": len(hidden),
            "covered": sum(x["covered"] is True for x in hidden),
            "not_detected_by_category_label": True},
        "queries": summaries, "unsupported_queries": rejected, "set_controls": set_controls,
        "qualification_verified": verify_qualification_bundle(qualification),
        "limits": ["category correctness is declared, not inferred from the face",
            "routing never chooses a radius by inspecting evaluation or pair similarity",
            "unsupported conditions abstain without deleting their evaluation denominator",
            "real detector inference and independently labeled portraits remain unvalidated"]}
    write_json_new(root/"laboratory.json", answer)
    return answer
