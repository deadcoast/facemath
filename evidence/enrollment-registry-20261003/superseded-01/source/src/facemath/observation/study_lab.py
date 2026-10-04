"""A prospective, rotated/scaled stable-bias + domain-shift falsification study.

No learned model is used. Known render coordinates generate independent fixture
references; the scripted observer injects a documented bias and noise process.
This is a test of the integrated instrument, never human detector validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .anatomy import SEMANTIC_21
from .artifacts import write_bundle, verify_bundle
from .bias import BiasCalibration, ocular_basis
from .canonical import InstrumentConfig
from .contracts import Observation, array_digest, canonical_json, object_digest
from .instrument import FacialInstrument
from .lab import phantom_points, render_phantom
from .provider import apply_points
from .reference import REFERENCE_NAMES, ReferenceAnnotation
from .study import ObserverDeclaration, freeze_study, load_rgb, read_json, save_rgb, write_json_new
from .study_io import finalize_session, verify_qualification_bundle
from .study_runner import AttemptOutcome, run_session, verify_session

DOMAIN = "schematic-ocular-bias-and-unseen-shift-controls-v5-NOT-HUMAN"
BETA = np.array([[.018,-.012]]*4+[[.035,-.015],[.018,.025],[.018,.025],[-.015,.055]])
SPEC = {"name": "synthetic_ocular_bias_observer", "version": 1, "beta": BETA.tolist(),
        "source_noise_px": .13, "sample_jitter_px": .015, "noise_floor_px": .18,
        "scope": "known renderer controls, no live detector"}


@dataclass
class FixtureExecutor:
    """Only for explicitly synthetic study declarations; records actual call count."""
    records: dict[str, dict[str, Any]]
    calls: int = 0
    fail_on_call: int | None = None

    def preflight(self, declaration: ObserverDeclaration) -> dict[str, Any]:
        if declaration.execution_kind != "synthetic_fixture":
            raise ValueError("fixture cannot impersonate a native observer")
        return {"ready": True, "kind": "synthetic_fixture", "inference_executed": False,
                "real_model": False, "fixture_spec_sha256": object_digest(SPEC)}

    def execute(self, image_path: Path, declaration: ObserverDeclaration, directory: Path) -> AttemptOutcome:
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise KeyboardInterrupt("prespecified fixture interruption before receipt commit")
        image = load_rgb(image_path)
        record = self.records[array_digest(image)]
        write_json_new(directory / "fixture_execution.json", {"kind": "scripted_synthetic_observer",
                        "call_index": self.calls, "fixture_spec_sha256": object_digest(SPEC), "real_model": False})
        if record.get("failure"):
            return AttemptOutcome("no_face", detail="prespecified synthetic nondetection control, not real model output")
        xy = record["prediction"]
        observation = Observation(array_digest(image), image.shape[:2], "semantic21.v1", SEMANTIC_21,
            xy, record["samples"], declaration.provider, declaration.provider_fingerprint,
            declaration.noise_floor_px, declaration.attempted_samples,
            tuple(f"controlled_perturbation_{j}" for j in range(7)),
            ("scripted bias fixture; not actual learned face inference",), canonical_json(SPEC))
        return AttemptOutcome("observed", observation)


def make_fixture_study(root: str | Path, *, seed: int = 25091, development: int = 20,
                       calibration: int = 40, evaluation: int = 30,
                       include_excluded: bool = True) -> tuple[Path, FixtureExecutor, dict[str, dict[str, Any]]]:
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    observer = ObserverDeclaration("synthetic_ocular_bias_observer", object_digest(SPEC), "semantic21.v1", .18, 7, "synthetic_fixture")
    ids = [SEMANTIC_21.index(name) for name in REFERENCE_NAMES]
    entries, records, by_case = [], {}, {}
    total = development+calibration+evaluation
    for i in range(total+int(include_excluded)):
        split = "development" if i < development else "calibration" if i < development+calibration else "evaluation"
        excluded = i == total
        drift = split == "evaluation" and i >= development+calibration+(evaluation*2//3)
        stratum = "unseen_shift" if drift else ("degraded" if i%3 == 0 else "clear")
        angle = np.deg2rad(rng.uniform(-11,11)); scale = rng.uniform(.82,1.18)
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        truth = (phantom_points()-[128,105]) @ rotation.T*scale+[150,125]
        truth += rng.normal(0,.35,truth.shape)
        image = render_phantom(truth, seed=seed+i, shape=(280,320))
        eye_frame = ocular_basis(truth[ids])
        prediction = truth + np.array([.55,-.30])
        noise = .13*(1.45 if stratum == "degraded" else 1.)
        prediction[ids] = truth[ids] + BETA@eye_frame.T + rng.normal(0,noise,(8,2))
        if drift:
            prediction[ids[4:]] += np.array([.06,.065]) @ eye_frame.T
        samples = prediction[None] + rng.normal(0,.015,(7,21,2));samples[0]=prediction
        case_id = f"control_{i:03d}"
        failure = i == development+17 and calibration>17 or i == development+calibration+7 and evaluation>7
        image_name=f"{case_id}.png";save_rgb(root/image_name,image)
        reference = ReferenceAnnotation(array_digest(image),image.shape[:2],truth[ids],np.ones(8,bool),
            "known-render-coordinates","synthetic_fixture","known schematic renderer coordinates, no model-derived labels")
        ref_name=f"{case_id}_reference.json";write_json_new(root/ref_name,reference.to_dict())
        entry={"case_id":case_id,"group_id":f"independent_draw_{i:03d}","split":split,
            "lineage_ids":[f"render_{seed}_{i}"],"image":image_name,"references":[ref_name],"stratum":stratum}
        if excluded:
            entry.update(eligibility="excluded",exclusion_reason="prespecified excluded enrollment control; never sent to observer")
        entries.append(entry)
        record={"truth":truth,"prediction":prediction,"samples":samples,"failure":bool(failure),"entry":entry,"image":image}
        records[array_digest(image)]=record;by_case[case_id]=record
    spec={"format":"facemath.prospective_study.v5","domain":DOMAIN,"observer":observer.to_dict(),
        "minimum_raters":1,"maximum_cases_per_group":1,"minimum_development_groups":5,
        "alpha":.1,"bias_mode":"development_ocular","strata":["clear","degraded","unseen_shift"],"cases":entries}
    path=root/"study.json";write_json_new(path,spec)
    write_json_new(root/"fixture_contract.json", {"seed":seed,"spec":SPEC,"injected_beta":BETA.tolist(),
        "development":development,"calibration":calibration,"evaluation":evaluation,
        "unseen_shift_is_a_prespecified_nonexchangeable_stress_control":True})
    return path,FixtureExecutor(records),by_case


def run_study_laboratory(destination: str | Path, *, seed: int = 25091) -> dict[str, Any]:
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    spec,executor,records=make_fixture_study(root/"inputs",seed=seed)
    study=freeze_study(spec,root/"frozen")
    first=run_session(study.root,root/"acquisition",executor,maximum_new_cases=7)
    after_first=executor.calls
    completed=run_session(study.root,root/"acquisition",executor)
    after_complete=executor.calls
    again=run_session(study.root,root/"acquisition",executor)
    output=finalize_session(root/"acquisition",root/"qualification")
    report=read_json(output/"qualification.json")
    calibration=BiasCalibration.from_dict(read_json(output/"localization_calibration.json"))
    # Two held-out clear/degraded cases, selected by fixed index rather than by
    # observed error/coverage. Selection is declared here, not optimized.
    ca,cb=records['control_060'],records['control_061']
    def saved(index):
        return Observation.from_dict(read_json(root/f"acquisition/cases/case_{index:05d}/observation.json"))
    oa,ob=saved(60),saved(61)
    config=InstrumentConfig(uncertainty_draws=24,canvas_size=192,eye_span_px=60,enable_flow=False,edge_samples=16)
    bare=FacialInstrument(config).run(ca['image'],cb['image'],oa,ob)
    instrument=FacialInstrument(config,localization_calibration=calibration,domain=DOMAIN,
        query_groups=(ca['entry']['group_id'],cb['entry']['group_id']),
        query_lineages=(tuple(ca['entry']['lineage_ids']),tuple(cb['entry']['lineage_ids'])))
    qualified=instrument.run(ca['image'],cb['image'],oa,ob)
    write_bundle(qualified,root/"qualified_pair",ca['image'],cb['image'])
    common=bare.raw_geometry_arrays()
    unchanged=all(np.array_equal(a,qualified.arrays()[k]) for k,a in common.items())
    ids=[SEMANTIC_21.index(n) for n in REFERENCE_NAMES]
    known_gap=np.linalg.norm(apply_points(ca['truth'][ids],qualified.state.source_to_canonical_a)-
                            apply_points(cb['truth'][ids],qualified.state.source_to_canonical_b),axis=1)/qualified.state.normalization_span_px
    envelope=qualified.localization.base
    contained=(envelope.lower_ipd<=known_gap)&(known_gap<=envelope.upper_ipd)
    write_json_new(root/"pair_ground_truth_check.json",{"known_render_gap_ipd":known_gap.tolist(),
        "contained_by_bounds":contained.tolist(),"fixed_observed_gauge":True,"human_accuracy":False})
    summary={"format":"facemath.study_lab.v5","kind":"synthetic_falsification_NOT_HUMAN_DETECTOR_VALIDATION",
        "seed":seed,"enrolled":completed['enrolled'],"eligible":completed['eligible'],"excluded":completed['prospectively_excluded'],
        "calls_first_batch":after_first,"calls_completed":after_complete,"calls_after_idempotent_resume":executor.calls,
        "outcomes":completed['outcomes'],"beta_max_absolute_error":float(np.max(np.abs(calibration.bias.beta-BETA))),
        "raw_quantile":report['raw_baseline']['quantile'],"centered_quantile":calibration.quantile,
        "coverage":report['selected_error_model']['coverage'],"slices":report['slices'],
        "unchanged_raw_pair_arrays":len(common),"raw_geometry_arrays_identical":unchanged,
        "known_query_reference_gaps_contained":bool(contained.all()),
        "checks":{"first_batch_committed_seven":first['committed']==7,
                  "resume_no_duplicate_calls":executor.calls==completed['eligible'],
                  "excluded_never_observed":completed['enrolled']-executor.calls==1,
                  "bias_recovered":float(np.max(np.abs(calibration.bias.beta-BETA)))<.005,
                  "raw_geometry_unchanged":unchanged,
                  "stress_stratum_has_no_calibration_support":not report['slices'][-1]['has_calibration_representation'],
                  "stress_undercoverage_exposed":report['slices'][-1]['evaluation']['empirical_coverage']<.5}}
    write_json_new(root/"laboratory.json",summary)
    verify_qualification_bundle(output);verify_bundle(root/"qualified_pair")
    return summary
