"""Measured-pixel routing through the inherited prospective instrument.

Schematic rendered sources, independent known render coordinates, and scripted
localization errors. Raster diagnostics are ACTUALLY computed; the observer is
explicitly synthetic. Altered observer errors within the same pixel condition
remain a deliberate, visible counterexample to hidden-shift detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np

from .acquisition import (AcquisitionPolicy, AcquisitionRule, ApertureSource, MeasuredGroup, Predicate,
                          FEATURE_NAMES, UNMATCHED, AMBIGUOUS, measure_source, write_group)
from .artifacts import write_bundle, verify_bundle
from .bias import load_localization_calibration
from .canonical import InstrumentConfig
from .contracts import ObservationError, array_digest, canonical_json
from .instrument import FacialInstrument, InstrumentFailure
from .intrinsic import nominal_values, relation_registry
from .intrinsic_lab import _so2_minimum
from .joint_lab import INDICES
from .reference import ReferenceAnnotation
from .stratified_lab import DECLARATION, record, observation, StratifiedFixtureExecutor
from .study import freeze_study, read_json, save_rgb, write_json_new
from .study_io import finalize_session, verify_qualification_bundle
from .study_runner import run_session

DOMAIN = "synthetic-measured-aperture-v9-NOT-HUMAN"
APERTURE = (35, 40, 210, 230)


def demonstration_policy(maximum_group_members: int = 2) -> AcquisitionPolicy:
    """Illustrative thresholds selected on source-only schematic development probes.

    These values are NOT validated portrait-quality or clinical thresholds.
    Neither references nor detector outcomes choose them in this experiment.
    """
    common = (Predicate("min.native_short_edge_px", ">=", 96.),
              Predicate("min.code_range_90", ">=", .10),
              Predicate("max.low_code_fraction", "<", .20),
              Predicate("max.high_code_fraction", "<", .20))
    return AcquisitionPolicy(
        "Schematic stored-raster apertures; all enrolled group members. Clear: minimum detail >=0.30; "
        "smooth: minimum detail in [0.05,0.30). Both need >=96px, range>=0.10 and code-tail fractions<0.20. "
        "Engineering demonstration selected using three source-only development probes, not human recommendations.",
        (AcquisitionRule("pixel_clear", common+(Predicate("min.detail_ratio", ">=", .30),)),
         AcquisitionRule("pixel_smooth", common+(Predicate("min.detail_ratio", ">=", .05), Predicate("min.detail_ratio", "<", .30)))),
        maximum_group_members=maximum_group_members)


def measured_record(seed: int, condition: str, *, expansion: float = 0., hidden_shift: bool = False) -> dict:
    base = record(seed, "degraded" if condition == "smooth" else "clear", expansion=expansion, hidden_shift=hidden_shift)
    if condition == "washed":
        # Actual source pixels, not an injected feature vector. Preserve distinct
        # images and all known renderer reference coordinates.
        base["image"] = np.clip(252. + base["image"].astype(float)*.012, 0, 255).astype(np.uint8)
    elif condition not in ("clear", "smooth"):
        raise ObservationError("unknown measured-lab condition")
    base["condition"] = condition
    return base


def make_measured_study(root: Path, *, seed: int = 29091, compact: bool = False):
    root.mkdir(parents=True, exist_ok=False)
    policy = demonstration_policy()
    development = 12 if compact else 24
    calibration = {"clear": 20 if compact else 36, "smooth": 12 if compact else 20, "washed": 10 if compact else 12}
    evaluation = {"clear": 6 if compact else 12, "smooth": 4 if compact else 8, "washed": 2 if compact else 4}
    shifts = 2 if compact else 4
    schedule = [("development", "clear" if i%2 else "smooth", False) for i in range(development)]
    for split, counts in (("calibration", calibration), ("evaluation", evaluation)):
        for condition, count in counts.items():
            for j in range(count):
                schedule.append((split, condition, split == "evaluation" and condition == "clear" and j >= count-shifts))
    # One repeated calibration group has two distinct source qualities. Its
    # complete-group extrema, not a selected convenient frame, determine routing.
    mixed_start = len(schedule)
    schedule += [("calibration", "clear", False), ("calibration", "smooth", False)]
    schedule += [("evaluation", "washed", False)]  # prospective exclusion, still measured
    entries, records, truth, imposed, per_source = [], {}, [], [], []
    counters: dict[tuple[str, str], int] = {}
    for i, (split, condition, shifted) in enumerate(schedule):
        item = measured_record(seed+i, condition, hidden_shift=shifted)
        k = counters.get((split, condition), 0); counters[(split, condition)] = k+1
        item["failure"] = ((split == "calibration" and condition == "clear" and k == 3)
                           or (split == "evaluation" and condition == "smooth" and k == 1))
        image_name, reference_name = f"source_{i:04d}.png", f"source_{i:04d}.reference.json"
        save_rgb(root/image_name, item["image"])
        ref = ReferenceAnnotation(array_digest(item["image"]), item["image"].shape[:2], item["truth"][INDICES],
            np.ones(8, bool), "known-schematic-render-coordinates", "synthetic_fixture",
            "known synthetic renderer geometry; not detector-generated human reference")
        write_json_new(root/reference_name, ref.to_dict())
        group = f"measured_{seed}_{'mixed' if i in (mixed_start,mixed_start+1) else i}"
        entry = {"case_id": f"source_{i:04d}", "group_id": group, "split": split,
                 "lineage_ids": [f"lineage_{group}"], "image": image_name, "references": [reference_name],
                 "acquisition_aperture": list(APERTURE)}
        if i == len(schedule)-1:
            entry.update(eligibility="excluded", exclusion_reason="frozen excluded-enrollment control; no observer call")
        entries.append(entry); records[array_digest(item["image"])] = item
        truth.append(item["truth"][INDICES]); imposed.append(item["latent_errors"])
        probe = MeasuredGroup(policy, group, (ApertureSource(item["image"], tuple(entry["lineage_ids"]), APERTURE),))
        per_source.append({"case_id":entry["case_id"],"condition":condition,"hidden_error_shift":shifted,
                           "individual_pixel_category":probe.category,"actual_features":probe.features[0].tolist()})
    spec = {"format":"facemath.prospective_study.v5", "domain":DOMAIN,"observer":DECLARATION.to_dict(),
            "minimum_raters":1,"maximum_cases_per_group":2,"minimum_development_groups":5,"alpha":.1,
            "bias_mode":"development_joint_ocular_stratified", "joint_error":{"shrinkage":.02,"floor_eye_spans":.001},
            "strata":list(policy.categories),"stratified_error":{"definition":policy.definition,
                "minimum_calibration_groups":9,"measurement_policy":policy.to_dict()},"cases":entries}
    write_json_new(root/"study.json",spec)
    write_json_new(root/"measurement_policy.json",policy.to_dict())
    write_json_new(root/"fixture_contract.json",{"format":"facemath.measured_source_fixture.v9", "seed":seed,
        "kind":"synthetic renderer and scripted observer; pixel measurements actually executed",
        "threshold_selection":"three distinct source-only development probes 29001/29002/29003; no reference errors or heldout tuning",
        "development_cases":development,"calibration_schedule":calibration,"evaluation_schedule":evaluation,
        "mixed_group_member_indices":[mixed_start,mixed_start+1], "per_source":per_source,
        "feature_order":list(FEATURE_NAMES),"prediction_exposure_to_measured_routing":False,
        "query_rule":"four fresh pairs at expansions0/4/8/12; B group also contains a fresh smooth context source",
        "limits":"blur and saturation alter real schematic pixels; observer error response is scripted, not empirical"})
    np.savez_compressed(root/"fixture_ground_truth.npz", reference=np.stack(truth),imposed_errors=np.stack(imposed))
    return root/"study.json", StratifiedFixtureExecutor(records)


def run_pixel_controls(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    p = demonstration_policy(2)
    a = measured_record(39011,"clear"); b=measured_record(39012,"smooth")
    source_a=ApertureSource(a['image'],('control-a',),APERTURE)
    source_b=ApertureSource(b['image'],('control-b',),APERTURE)
    individual=MeasuredGroup(p,'single',(source_a,)); mixed=MeasuredGroup(p,'mixed',(source_a,source_b))
    # Changes outside the declared aperture must not affect any raw feature.
    modified=a['image'].copy(); x0,y0,x1,y1=APERTURE
    outside=np.ones(modified.shape[:2],bool);outside[y0:y1,x0:x1]=False
    modified[outside]=np.random.default_rng(39013).integers(0,256,(int(outside.sum()),3),dtype=np.uint8)
    altered=ApertureSource(modified,('control-altered-background',),APERTURE)
    fa,grid=measure_source(source_a,p);fb,_=measure_source(altered,p)
    noise=np.random.default_rng(39014).integers(30,220,(128,128,1),dtype=np.uint8)
    noise=np.repeat(noise,3,2);smooth=cv2.GaussianBlur(noise,(21,21),4.)
    noisy=MeasuredGroup(p,'noise',(ApertureSource(noise,('noise',)),))
    lowpass=MeasuredGroup(p,'lowpass',(ApertureSource(smooth,('lowpass',)),))
    zero=np.full((128,128,3),128,np.uint8)
    flat=MeasuredGroup(p,'flat',(ApertureSource(zero,('flat',)),))
    overlapping=AcquisitionPolicy('deliberately overlapping test rules',(
        AcquisitionRule('first',(Predicate('min.native_short_edge_px','>=',4),)),
        AcquisitionRule('second',(Predicate('min.native_short_edge_px','>=',4),))),maximum_group_members=2)
    ambiguous=MeasuredGroup(overlapping,'ambiguous',(source_a,))
    for name,g in [('single',individual),('mixed',mixed),('noise',noisy),('lowpass',lowpass),('flat',flat),('ambiguous',ambiguous)]:
        write_group(g,destination/name)
    result={'format':'facemath.measured_source_controls.v9','source_only':True,
        'outside_aperture_features_identical':bool(np.array_equal(fa,fb)),
        'single_source_category':individual.category,'whole_group_category':mixed.category,
        'noise_category':noisy.category,'noise_detail':float(noisy.features[0,-1]),
        'smoothed_noise_category':lowpass.category,'smoothed_noise_detail':float(lowpass.features[0,-1]),
        'flat_category':flat.category,'flat_detail':float(flat.features[0,-1]),
        'overlapping_rule_category':ambiguous.category,
        'noise_counterexample':'high-frequency noise is not usable anatomical detail; this routing is NOT a face-quality detector',
        'no_predictions_or_reference_labels_consumed':True}
    np.savez_compressed(destination/'controls.npz',original_image=a['image'],outside_modified=modified,
                        aperture=np.array(APERTURE),original_features=fa,outside_modified_features=fb,
                        noise=noise,smoothed_noise=smooth,noise_features=noisy.features,lowpass_features=lowpass.features)
    write_json_new(destination/'controls.json',result)
    return result


def run_measured_laboratory(destination: str | Path, *, seed: int = 29091) -> dict:
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    controls=run_pixel_controls(root/'pixel_controls')
    path,executor=make_measured_study(root/'inputs',seed=seed)
    frozen=freeze_study(path,root/'frozen')
    run_session(frozen.root,root/'acquisition',executor,maximum_new_cases=5);first=executor.calls
    status=run_session(frozen.root,root/'acquisition',executor);completed=executor.calls
    run_session(frozen.root,root/'acquisition',executor)
    qualified=finalize_session(root/'acquisition',root/'qualification')
    model=load_localization_calibration(read_json(qualified/'localization_calibration.json'))
    policy=model.policy.measurement_policy
    cfg=InstrumentConfig(canvas_size=224,eye_span_px=70,uncertainty_draws=16,enable_flow=False,edge_samples=16)
    a=measured_record(seed+10000,'clear');oa=observation(a)
    ga=MeasuredGroup(policy,f'fresh_a_{seed}',(ApertureSource(a['image'],(f'fresh_lineage_a_{seed}',),APERTURE),))
    write_group(ga,root/'query_group_a')
    summaries=[];query_truth=[]
    for i,expansion in enumerate((0.,4.,8.,12.)):
        b=measured_record(seed+20000+i,'clear',expansion=expansion);ob=observation(b)
        context=measured_record(seed+30000+i,'smooth')
        gb=MeasuredGroup(policy,f'fresh_b_{seed}_{i}',(
            ApertureSource(b['image'],(f'fresh_lineage_b_{seed}_{i}',),APERTURE),
            ApertureSource(context['image'],(f'fresh_context_b_{seed}_{i}',),APERTURE)))
        write_group(gb,root/f'query_group_b_{i:02d}')
        result=FacialInstrument(cfg,localization_calibration=model,domain=DOMAIN,acquisition_groups=(ga,gb)).run(a['image'],b['image'],oa,ob)
        bare=FacialInstrument(cfg).run(a['image'],b['image'],oa,ob)
        unchanged=all(np.array_equal(v,result.raw_geometry_arrays()[k]) for k,v in bare.raw_geometry_arrays().items())
        if not unchanged:
            raise ObservationError('measured acquisition changed raw observed geometry')
        bundle=write_bundle(result,root/f'pair_{i:02d}_mouth_{int(expansion)}px',a['image'],b['image'])
        actual=_so2_minimum(a['truth'][INDICES],b['truth'][INDICES]);shape=result.intrinsic.shape
        specs=relation_registry(cfg);va,ma=nominal_values(a['truth'][INDICES],np.ones(8,bool),specs)
        vb,mb=nominal_values(b['truth'][INDICES],np.ones(8,bool),specs);gap=abs(va-vb);profile=result.intrinsic
        supported=ma&mb&profile.gap_available
        summaries.append({'bundle':bundle.name,'mouth_expansion_px':expansion,'derived_categories':[ga.category,gb.category],
            'selected_b_individual_category':MeasuredGroup(policy,'single-b', (ApertureSource(b['image'],('single-b',),APERTURE),)).category,
            'source_quantiles':result.localization.base.source_quantiles.tolist(),
            'known_reference_shape_rms':actual,'shape_range':shape.to_dict(),
            'known_reference_shape_enclosed':bool(shape.available and shape.lower-1e-10<=actual<=shape.upper+1e-10),
            'known_relation_gap_violations':int(np.sum(supported&((gap<profile.gap_lower-1e-10)|(gap>profile.gap_upper+1e-10)))),
            'raw_arrays_unchanged':unchanged,'raw_array_count':len(bare.raw_geometry_arrays()),'pair_array_count':len(result.arrays()),
            'verified':verify_bundle(bundle)['verified']})
        query_truth.append(np.stack([a['truth'][INDICES],b['truth'][INDICES]]))
    bad=measured_record(seed+99000,'washed')
    bad_group=MeasuredGroup(policy,'refused-source',(ApertureSource(bad['image'],('refused-new-lineage',),APERTURE),))
    write_group(bad_group,root/'rejected_group')
    class SpyProvider:
        calls=0
        def observe(self,image):
            self.calls+=1
            raise AssertionError('rejected acquisition must not invoke observer')
    spy=SpyProvider();refusal=None
    try:
        FacialInstrument(cfg,spy,localization_calibration=model,domain=DOMAIN,acquisition_groups=(ga,bad_group)).run(a['image'],bad['image'])
    except InstrumentFailure as error:
        refusal=str(error);(root/'rejected_before_observer.events.jsonl').write_text(error.journal_jsonl)
    report=read_json(qualified/'qualification.json')
    result={'format':'facemath.measured_acquisition_laboratory.v9','kind':'SYNTHETIC_CONTROLS_NOT_HUMAN_VALIDATION','seed':seed,
        'study_sha256':frozen.digest,'observer_calls':{'first':first,'complete':completed,'after_completed_resume':executor.calls},
        'enrollment':{k:status[k] for k in ('enrolled','eligible','prospectively_excluded','outcomes')},
        'category_counts':{s:sum(c['stratum']==s and c['eligibility']=='included' for c in frozen.cases) for s in policy.categories},
        'calibration_categories':[r.to_dict() for r in model.radii],
        'heldout_coverage':report['selected_error_model']['coverage'],
        'pixel_controls':controls,'query_pairs':summaries,'rejected_query':{'reason':refusal,'observer_calls':spy.calls},
        'qualification_verified':verify_qualification_bundle(qualified)['verified'],
        'limits':['scripted observer errors do not validate a real detector','source apertures are declared, not detected anatomy',
                  'all groups retain their declared members; no program can attest unreported repeats',
                  'gradient/code-value rules cannot detect hidden shifts in observer errors']}
    np.savez_compressed(root/'query_ground_truth.npz',reference_shapes=np.stack(query_truth))
    write_json_new(root/'laboratory.json',result)
    return result
