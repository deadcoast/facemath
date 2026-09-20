"""One prospective synthetic study and deterministic full-shape set controls.

Reference coordinates come from the renderer. The observer is scripted and
explicitly NOT a native model or evidence of real-face accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .anatomy import SEMANTIC_21
from .artifacts import write_bundle, verify_bundle
from .bias import ocular_basis, load_localization_calibration
from .canonical import InstrumentConfig
from .contracts import Observation, ObservationError, array_digest, canonical_json, object_digest
from .instrument import FacialInstrument
from .intrinsic import relation_registry, nominal_values, bound_refitted_shape, _bound
from .intrinsic_lab import CONTROL_POINTS, _exact_relation_batch, _so2_minimum
from .intrinsic_sets import EllipticSource, SourceRelations, displacement, absolute_difference
from .joint import JointCalibration, JOINT_MODE
from .journal import environment
from .lab import phantom_points, render_phantom
from .reference import REFERENCE_NAMES, ReferenceAnnotation
from .study import ObserverDeclaration, freeze_study, load_rgb, read_json, save_rgb, write_json_new
from .study_io import finalize_session, verify_qualification_bundle
from .study_runner import AttemptOutcome, run_session

DOMAIN = "schematic-joint-ocular-error-and-heldout-shift-v7-NOT-HUMAN"
BETA = np.array([[.012, -.008]]*4+[[.018, -.012], [.01, .018], [.01, .018], [-.008, .025]])
SPEC = {"name": "synthetic_joint_shape_observer", "version": 7, "beta": BETA.tolist(),
        "common_translation_std_eye_spans": .035, "local_std_eye_spans": .0015,
        "sample_jitter_px": .015, "noise_floor_px": .18,
        "execution": "scripted renderer errors, NOT live detector inference"}
INDICES = [SEMANTIC_21.index(n) for n in REFERENCE_NAMES]
DECLARATION = ObserverDeclaration(SPEC['name'], object_digest(SPEC), 'semantic21.v1', .18, 7, 'synthetic_fixture')


def fixture_observation(image: np.ndarray, prediction: np.ndarray, samples: np.ndarray) -> Observation:
    return Observation(array_digest(image), image.shape[:2], 'semantic21.v1', SEMANTIC_21,
        prediction, samples, DECLARATION.provider, DECLARATION.provider_fingerprint,
        .18, 7, tuple(f'controlled_jitter_{j}' for j in range(7)),
        ('scripted joint-error control; not learned anatomical inference',), canonical_json(SPEC))


def _record(seed: int, *, expansion: float = 0., shifted: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    angle, scale = rng.uniform(-.12, .12), rng.uniform(.9, 1.1)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    base = phantom_points().copy(); base[14, 0] -= expansion; base[15, 0] += expansion
    truth = (base-[128, 105])@rotation.T*scale+[150, 125]
    image = render_phantom(truth, seed=seed, shape=(280, 320))
    eye = ocular_basis(truth[INDICES])
    common = rng.normal(0, SPEC['common_translation_std_eye_spans'], (1,2))
    local = rng.normal(0, SPEC['local_std_eye_spans'], (8,2))
    latent = BETA+common+local
    if shifted:
        latent[4:] += [.055, .065]  # Prespecified nonexchangeable stress condition.
    prediction = truth + common@eye.T
    prediction[INDICES] = truth[INDICES] + latent@eye.T
    samples = prediction[None]+rng.normal(0, SPEC['sample_jitter_px'], (7,21,2)); samples[0]=prediction
    return {'truth': truth, 'image': image, 'prediction': prediction, 'samples': samples,
            'latent_errors': latent, 'shifted': shifted, 'failure': False}


@dataclass
class JointFixtureExecutor:
    records: dict[str, dict[str, Any]]
    calls: int = 0

    def preflight(self, declaration: ObserverDeclaration) -> dict[str, Any]:
        if declaration != DECLARATION:
            raise ObservationError('scripted joint executor only accepts its exact synthetic declaration')
        return {'ready': True, 'kind': 'synthetic_fixture', 'live_inference_executed': False,
                'fixture_spec_sha256': object_digest(SPEC)}

    def execute(self, image_path: Path, declaration: ObserverDeclaration, directory: Path) -> AttemptOutcome:
        self.calls += 1
        image = load_rgb(image_path); record = self.records[array_digest(image)]
        write_json_new(directory/'fixture_execution.json', {'kind': 'synthetic_joint_shape_observer',
            'call_index': self.calls, 'fixture_spec_sha256': object_digest(SPEC), 'live_inference': False})
        if record['failure']:
            return AttemptOutcome('no_face', detail='prespecified scripted failure, not a real model result')
        return AttemptOutcome('observed', fixture_observation(image, record['prediction'], record['samples']))


def make_joint_study(root: Path, *, seed: int = 27091, development: int = 32,
                     calibration: int = 40, evaluation: int = 24) -> tuple[Path, JointFixtureExecutor]:
    root.mkdir(parents=True, exist_ok=False)
    total = development+calibration+evaluation
    entries, records, truths, latent = [], {}, [], []
    for i in range(total+1):
        split = 'development' if i < development else 'calibration' if i < development+calibration else 'evaluation'
        shifted = split == 'evaluation' and i >= development+calibration+evaluation*2//3
        record = _record(seed+i, shifted=shifted)
        record['failure'] = i in (development+17, development+calibration+7)
        image_name, ref_name = f'joint_{i:03d}.png', f'joint_{i:03d}_reference.json'
        image = record['image']; save_rgb(root/image_name, image)
        reference = ReferenceAnnotation(array_digest(image), image.shape[:2], record['truth'][INDICES],
            np.ones(8,bool), 'known-render-coordinates', 'synthetic_fixture',
            'independent known renderer coordinates; no predicted labels')
        write_json_new(root/ref_name, reference.to_dict())
        entry = {'case_id': f'joint_{i:03d}', 'group_id': f'joint_draw_{seed}_{i}', 'split': split,
                 'lineage_ids': [f'joint_source_{seed}_{i}'], 'image': image_name,
                 'references': [ref_name], 'stratum': 'unseen_shift' if shifted else 'stationary'}
        if i == total:
            entry.update(eligibility='excluded', exclusion_reason='prespecified enrollment control; never observed')
        entries.append(entry); records[array_digest(image)]=record
        truths.append(record['truth'][INDICES]); latent.append(record['latent_errors'])
    study = {'format': 'facemath.prospective_study.v5', 'domain': DOMAIN,
             'observer': DECLARATION.to_dict(), 'minimum_raters': 1, 'maximum_cases_per_group': 1,
             'minimum_development_groups': 5, 'alpha': .1, 'bias_mode': JOINT_MODE,
             'joint_error': {'shrinkage': .02, 'floor_eye_spans': .001},
             'strata': ['stationary', 'unseen_shift'], 'cases': entries}
    write_json_new(root/'study.json', study)
    write_json_new(root/'fixture_contract.json', {'seed': seed, 'spec': SPEC,
        'development': development, 'calibration': calibration, 'evaluation': evaluation,
        'fixed_shifted_evaluation_fraction': 'last third', 'injected_failures': [development+17, development+calibration+7],
        'query_selection': 'fresh seed+1000 and seed+2000..2003; fixed mouth expansion 0,4,8,12'})
    np.savez_compressed(root/'fixture_ground_truth.npz', reference_coordinates=np.stack(truths),
                        injected_ocular_error=np.stack(latent), reference_indices=np.asarray(INDICES))
    return root/'study.json', JointFixtureExecutor(records)


def _source(points: np.ndarray, covariance: np.ndarray, q: float = 1.) -> EllipticSource:
    blocks = covariance.reshape(8,2,8,2)
    marginal = np.stack([blocks[i,:,i,:] for i in range(8)])
    return EllipticSource(points, marginal, np.ones(8,bool), q, joint_covariance=covariance)


def run_joint_set_controls(root: Path, *, seed: int = 27091, samples: int = 1024) -> dict[str, Any]:
    if type(samples) is not int or not 64 <= samples <= 100000:
        raise ObservationError('joint controls require 64..100000 samples')
    root.mkdir(parents=True, exist_ok=False); rng=np.random.default_rng(seed)
    center_b = CONTROL_POINTS.copy(); center_b[5,0]-=.1; center_b[6,0]+=.1
    # A dominant rigid error, smaller local deformation, no fabricated detections.
    translation=np.tile(np.eye(2),(8,1))*.04
    random=rng.normal(0,.001,(16,16))
    covariance=translation@translation.T+random@random.T+np.eye(16)*.001**2
    a,b=_source(CONTROL_POINTS,covariance),_source(center_b,covariance*1.2)
    ea,eb=SourceRelations(a),SourceRelations(b);specs=relation_registry(InstrumentConfig())
    intervals=[[_bound(e,s) for s in specs] for e in (ea,eb)]
    gaps=[absolute_difference(*v) for v in zip(*intervals)]
    gaplo=np.array([g.lower for g in gaps]);gaphi=np.array([g.upper for g in gaps])
    shape=bound_refitted_shape(np.stack([a.centers,b.centers]),(ea,eb),gaplo,True)
    draws,unit=[],[]
    for source in (a,b):
        z=rng.normal(size=(samples,16));z/=np.linalg.norm(z,axis=1,keepdims=True)
        radius=np.ones(samples);radius[samples//2:]=rng.uniform(size=samples-samples//2)**(1/16)
        z*=radius[:,None];unit.append(z)
        draws.append(source.centers+(source.quantile*z@source.joint_factor.reshape(16,16).T).reshape(samples,8,2))
    values=[_exact_relation_batch(d,specs) for d in draws];actual_gaps=abs(values[0]-values[1])
    refits=np.array([_so2_minimum(x,y) for x,y in zip(*draws)])
    # Identical marginals, opposite cross-point correlations: different relative errors.
    rigid=np.tile(np.eye(2),(8,1))*.08
    counter=rigid.copy();counter[4:8]*=-1
    rigid_source=_source(CONTROL_POINTS,rigid@rigid.T)
    counter_source=_source(CONTROL_POINTS,counter@counter.T)
    er,ec=SourceRelations(rigid_source),SourceRelations(counter_source)
    relaxed=SourceRelations(EllipticSource(CONTROL_POINTS,rigid_source.covariance,np.ones(8,bool),1.))
    observed=np.stack([CONTROL_POINTS,CONTROL_POINTS])
    pure_shape=bound_refitted_shape(observed,(er,er),np.zeros(256),True)
    relaxed_shape=bound_refitted_shape(observed,(relaxed,relaxed),np.zeros(256),True)
    result={'format':'facemath.joint_set_controls.v7','kind':'analytic_and_synthetic_NOT_HUMAN',
        'seed':seed,'sampled_shape_pairs':samples,'scalar_relation_checks':int(actual_gaps.size),
        'sampled_relation_violations':int(np.sum((actual_gaps<gaplo-1e-10)|(actual_gaps>gaphi+1e-10))),
        'sampled_shape_violations':int(np.sum((refits<shape.lower-1e-10)|(refits>shape.upper+1e-10))),
        'bounds':shape.to_dict(),'sampled_refit_range':[float(refits.min()),float(refits.max())],
        'same_marginals_different_dependence':{'marginal_blocks_identical':bool(np.array_equal(rigid_source.covariance,counter_source.covariance)),
            'rigid_eye_vector_uncertainty_radius':er.radius(np.array([-.5,-.5,.5,.5,0,0,0,0])),
            'opposed_eye_vector_uncertainty_radius':ec.radius(np.array([-.5,-.5,.5,.5,0,0,0,0]))},
        'translation_cancellation':{'joint_shape':pure_shape.to_dict(),'same_set_marginal_relaxation':relaxed_shape.to_dict()},
        'limits':['finite samples are falsification checks, not proofs',
                  'one latent ball per source, no statistical source independence asserted',
                  'conservative float bounds, not exact interval extrema or formal interval arithmetic']}
    np.savez_compressed(root/'controls.npz',draws_a=draws[0],draws_b=draws[1],unit_ball_a=unit[0],unit_ball_b=unit[1],
        centers=np.stack([a.centers,b.centers]),joint_covariances=np.stack([a.joint_covariance,b.joint_covariance]),
        factors=np.stack([a.joint_factor,b.joint_factor]),values_a=values[0],values_b=values[1],actual_gaps=actual_gaps,
        gap_lower=gaplo,gap_upper=gaphi,refitted_rms=refits,rigid_covariance=rigid_source.joint_covariance,
        opposed_covariance=counter_source.joint_covariance,source_lower=np.array([[x.lower for x in s]for s in intervals]),
        source_upper=np.array([[x.upper for x in s]for s in intervals]))
    write_json_new(root/'relations.json',{'relations':[s.metadata() for s in specs]})
    write_json_new(root/'results.json',result);return result


def run_joint_laboratory(destination: str | Path, *, seed: int = 27091) -> dict[str, Any]:
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    controls=run_joint_set_controls(root/'set_controls',seed=seed)
    path,executor=make_joint_study(root/'inputs',seed=seed)
    frozen=freeze_study(path,root/'frozen')
    run_session(frozen.root,root/'acquisition',executor,maximum_new_cases=9);first_calls=executor.calls
    status=run_session(frozen.root,root/'acquisition',executor);complete_calls=executor.calls
    run_session(frozen.root,root/'acquisition',executor)
    qualification=finalize_session(root/'acquisition',root/'qualification')
    model=load_localization_calibration(read_json(qualification/'localization_calibration.json'))
    if not isinstance(model,JointCalibration) or model.quantile is None:
        raise ObservationError('synthetic joint control failed calibration; all acquisition records retained')
    config=InstrumentConfig(canvas_size=224,eye_span_px=70,uncertainty_draws=16,enable_flow=False,edge_samples=16)
    specs=relation_registry(config);record_a=_record(seed+1000)
    oa=fixture_observation(record_a['image'],record_a['prediction'],record_a['samples'])
    query_results,truths=[] ,[]
    for i,expansion in enumerate((0.,4.,8.,12.)):
        rb=_record(seed+2000+i,expansion=expansion);ob=fixture_observation(rb['image'],rb['prediction'],rb['samples'])
        engine=FacialInstrument(config,localization_calibration=model,domain=DOMAIN,
            query_groups=(f'queryA_{seed}',f'queryB_{seed}_{i}'),
            query_lineages=((f'queryA_source_{seed}',),(f'queryB_source_{seed}_{i}',)))
        result=engine.run(record_a['image'],rb['image'],oa,ob)
        bare=FacialInstrument(config).run(record_a['image'],rb['image'],oa,ob)
        name=f'pair_{i:02d}_mouth_{expansion:g}px';write_bundle(result,root/name,record_a['image'],rb['image'])
        old=bare.raw_geometry_arrays();unchanged=all(np.array_equal(v,result.arrays()[k])for k,v in old.items())
        ta,tb=record_a['truth'][INDICES],rb['truth'][INDICES];truths.append(np.stack([ta,tb]))
        va,vb=nominal_values(ta,np.ones(8,bool),specs)[0],nominal_values(tb,np.ones(8,bool),specs)[0]
        true_gaps=abs(va-vb);p=result.intrinsic;g=p.gap_available;true_shape=_so2_minimum(ta,tb)
        joint_width=p.shape.upper-p.shape.lower if p.shape.available else None
        relaxed=p.joint.marginal_shape;relaxed_width=relaxed.upper-relaxed.lower if relaxed.available else None
        query_results.append({'pair':name,'mouth_expansion_per_side_px':expansion,'known_reference_shape_rms':true_shape,
            'observed_shape_rms':p.shape.observed_rms,'joint_shape':p.shape.to_dict(),'same_set_marginal_shape':relaxed.to_dict(),
            'width_ratio_joint_to_relaxation':joint_width/relaxed_width if joint_width is not None and relaxed_width else None,
            'known_shape_contained':bool(p.shape.available and p.shape.lower-1e-10<=true_shape<=p.shape.upper+1e-10),
            'bounded_relations':int(g.sum()),'known_relation_violations':int(np.sum(g & ((true_gaps<p.gap_lower-1e-10)|(true_gaps>p.gap_upper+1e-10)))),
            'raw_arrays_unchanged':unchanged,'raw_array_count':len(old),'total_arrays':len(result.arrays()),
            'joint_audit':p.joint.to_dict(),'verification':verify_bundle(root/name)})
    np.savez_compressed(root/'query_ground_truth.npz',reference_coordinates=np.stack(truths))
    report=read_json(qualification/'qualification.json')
    summary={'format':'facemath.joint_lab.v7','kind':'scripted_synthetic_controls_NOT_HUMAN_INFERENCE',
        'seed':seed,'calls_first_batch':first_calls,'calls_complete':complete_calls,'calls_after_resume':executor.calls,
        'enrolled':status['enrolled'],'outcomes':status['outcomes'],'excluded':status['prospectively_excluded'],
        'selected_error_model':report['selected_error_model'],'slices':report['slices'],
        'queries':query_results,'set_controls':controls,'environment':environment(),
        'qualification_verification':verify_qualification_bundle(qualification),
        'boundary':'no human annotations, real detector accuracy, original-face score, or Gaussian error claim'}
    write_json_new(root/'laboratory.json',summary);return summary
