"""End-to-end milestone controls, with deliberately explicit non-human provenance.

The passing branch is an engineered deterministic containment control: evaluation
errors sit at the development mean, calibration probes surround it. It is NOT an
exchangeable real-observer benchmark. A separately prespecified shifted branch
proves that the SAME thresholds prohibit deployment. Neither policy is selected
by its outcome. Real studies must satisfy the sampling assumptions themselves.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .acquisition import ApertureSource, MeasuredGroup, write_group
from .acquisition_lab import APERTURE, demonstration_policy, measured_record
from .admission import AdmissionPolicy
from .anatomy import SEMANTIC_21
from .artifacts import write_bundle, verify_bundle
from .bias import ocular_basis, load_localization_calibration
from .canonical import InstrumentConfig
from .contracts import Observation, ObservationError, array_digest, object_digest, canonical_json
from .deployment import ValidatedQualification
from .instrument import FacialInstrument, InstrumentFailure
from .intrinsic_lab import _so2_minimum
from .joint_lab import INDICES
from .reference import ReferenceAnnotation
from .study import ObserverDeclaration, load_rgb, read_json, write_json_new, save_rgb
from .study_io import finalize_session
from .study_runner import AttemptOutcome, run_session
from .workspace import SPEC_FORMAT, prepare_workspace, import_reference, workspace_status, freeze_workspace

DOMAIN = 'synthetic-wave10-workflow-controls-NOT-A-HUMAN-STUDY'
SPEC = {'name': 'synthetic_milestone_containment_observer', 'version': 10,
    'bias_eye_spans': [.012, -.008], 'development_translation_probe': .006,
    'calibration_translation_probe': .018, 'evaluation_center_error': 0.,
    'negative_evaluation_local_shift': [.10, .08], 'jitter_px': .015,
    'kind': 'deterministic positive/negative SOFTWARE controls, not empirical observer accuracy'}
DECLARATION = ObserverDeclaration(SPEC['name'],object_digest(SPEC),'semantic21.v1',.18,7,'synthetic_fixture')


def _control_record(seed: int, condition: str, *, probe: float = 0., shifted: bool = False,
                    expansion: float = 0.) -> dict[str, Any]:
    r = measured_record(seed, condition, expansion=expansion)
    ocular = ocular_basis(r['truth'][INDICES])
    latent = np.array(SPEC['bias_eye_spans']) + np.array([probe, -.5*probe])
    prediction = r['truth'] + latent @ ocular.T
    if shifted:
        prediction[INDICES[4:]] += np.array(SPEC['negative_evaluation_local_shift']) @ ocular.T
    rng=np.random.default_rng(seed+1010000)
    samples=prediction[None]+rng.normal(0,.015,(7,21,2)); samples[0]=prediction
    r.update(prediction=prediction,samples=samples,failure=False)
    return r


def control_observation(r: dict) -> Observation:
    image=r['image']
    return Observation(array_digest(image),image.shape[:2],'semantic21.v1',SEMANTIC_21,
        r['prediction'],r['samples'],DECLARATION.provider,DECLARATION.provider_fingerprint,
        .18,7,tuple(f'control_jitter_{i}' for i in range(7)),
        ('deterministic synthetic workflow control, not real observer output',),canonical_json(SPEC))


@dataclass
class ControlExecutor:
    records: dict[str,dict]
    calls: int = 0

    def preflight(self,declaration:ObserverDeclaration)->dict:
        if declaration!=DECLARATION:raise ObservationError('milestone fixture recipe mismatch')
        return {'ready':True,'kind':'synthetic_fixture','native_inference_executed':False,'spec_sha256':object_digest(SPEC)}

    def execute(self,image_path:Path,declaration:ObserverDeclaration,directory:Path)->AttemptOutcome:
        self.calls+=1
        r=self.records[array_digest(load_rgb(image_path))]
        return AttemptOutcome('observed',control_observation(r))


def make_milestone_workspace_spec(root: Path, *, seed: int = 31091, shifted: bool = False,
                                  compact: bool = False) -> tuple[Path,ControlExecutor,list[tuple[str,Path]]]:
    root.mkdir(parents=True,exist_ok=False)
    measured=demonstration_policy(2)
    policy=AdmissionPolicy(('pixel_clear','pixel_smooth'),minimum_groups=2 if compact else 20,
        minimum_delivery=.10 if compact else .80,minimum_coverage=.10 if compact else .80,
        family_error=.10,evidence_kind='synthetic_control',independent_groups_declared=True,
        sampling_design='Deterministic branch/containment control. No population interpretation; does not establish real sampling assumptions.')
    ndev,ncal,neval=(8,12,3) if compact else (16,32,24)
    schedule=[('development','clear' if i%2 else 'smooth', .006*(1 if (i//2)%2 else -1)) for i in range(ndev)]
    for split,n in [('calibration',ncal),('evaluation',neval)]:
        for condition in ('clear','smooth'):
            for j in range(n):
                probe=(.018*(1 if j%2 else -1)) if split=='calibration' else 0.
                schedule.append((split,condition,probe))
    sources,imports,records,truth,errors=[],[],{},[],[]
    for i,(split,condition,probe) in enumerate(schedule+[('evaluation','clear',0.)]):
        is_shift=shifted and split=='evaluation' and condition=='smooth'
        r=_control_record(seed+i,condition,probe=probe,shifted=is_shift)
        image_name=f'source_{i:05d}.png';ref_name=f'source_{i:05d}.reference.json'
        save_rgb(root/image_name,r['image'])
        ref=ReferenceAnnotation(array_digest(r['image']),r['image'].shape[:2],r['truth'][INDICES],np.ones(8,bool),
            'known-render-coordinates','synthetic_fixture','known renderer coordinates; NOT a human annotation')
        write_json_new(root/ref_name,ref.to_dict())
        case_id=f'case_{i:05d}'
        source={'case_id':case_id,'group_id':f'group_{seed}_{i}','lineage_ids':[f'lineage_{seed}_{i}'],
                'split':split,'image':image_name,'acquisition_aperture':list(APERTURE)}
        if i==len(schedule):
            source.update(eligibility='excluded',exclusion_reason='prespecified never-invoked control')
        else:imports.append((case_id,root/ref_name))
        sources.append(source);records[array_digest(r['image'])]=r
        truth.append(r['truth'][INDICES]);errors.append(r['prediction'][INDICES]-r['truth'][INDICES])
    settings={'format':'facemath.prospective_study.v5','domain':DOMAIN,'observer':DECLARATION.to_dict(),
        'minimum_raters':1,'maximum_cases_per_group':2,'minimum_development_groups':5,'alpha':.1,
        'bias_mode':'development_joint_ocular_stratified','joint_error':{'shrinkage':.02,'floor_eye_spans':.001},
        'strata':list(measured.categories),'stratified_error':{'definition':measured.definition,
            'minimum_calibration_groups':9,'measurement_policy':measured.to_dict()},
        'admission_policy':policy.to_dict()}
    spec={'format':SPEC_FORMAT,'study':settings,'rater_ids':['known-render-coordinates'],
        'sources':sources,'source_description':'SYNTHETIC WORKFLOW CONTROL; never presented as a human study'}
    write_json_new(root/'workspace.spec.json',spec)
    write_json_new(root/'fixture_contract.json',{'format':'facemath.milestone_control.v10','seed':seed,'shifted':shifted,
        'compact':compact,'observer':SPEC,'schedule':schedule,'admission_policy':policy.to_dict(),
        'no_heldout_tuning':True,'positive_control':'evaluation at development mean; calibration surrounds it',
        'negative_control':'same source conditions, prespecified nonrigid error shift in smooth evaluation',
        'statistical_limit':'engineered deterministic branch test; passing does NOT validate exchangeable human observations'})
    np.savez_compressed(root/'ground_truth.npz',reference_shapes=np.stack(truth),imposed_errors=np.stack(errors))
    return root/'workspace.spec.json',ControlExecutor(records),imports


def run_control_study(root: Path, *, seed: int, shifted: bool, compact: bool = False) -> dict:
    root.mkdir(parents=True,exist_ok=False)
    path,executor,imports=make_milestone_workspace_spec(root/'inputs',seed=seed,shifted=shifted,compact=compact)
    workspace=prepare_workspace(path,root/'workspace')
    before=workspace_status(workspace)
    for case_id,reference in imports:import_reference(workspace,reference,case_id)
    duplicate=import_reference(workspace,imports[0][1],imports[0][0])
    ready=workspace_status(workspace)
    frozen=freeze_workspace(workspace,root/'frozen')
    run_session(frozen.root,root/'acquisition',executor,maximum_new_cases=4);first=executor.calls
    status=run_session(frozen.root,root/'acquisition',executor);completed=executor.calls
    run_session(frozen.root,root/'acquisition',executor)
    qualification=finalize_session(root/'acquisition',root/'qualification')
    report=read_json(qualification/'qualification.json')
    result={'source_count':len(frozen.cases),'labels_initially_missing':not before['ready_to_freeze_labels'],
        'ready_after_import':ready['ready_to_freeze_labels'],'retained_revisions':ready['retained_annotation_revisions'],
        'duplicate_import_idempotent':duplicate['idempotent'],'observer_calls':[first,completed,executor.calls],
        'enrollment':{k:status[k] for k in ('enrolled','eligible','prospectively_excluded','outcomes')},
        'admission':report['empirical_admission'],'frozen_origin_present':'workspace_origin' in frozen.data}
    write_json_new(root/'control.json',result)
    return result


def run_milestone_laboratory(destination:str|Path, *, seed:int=31091, compact:bool=False)->dict:
    root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    positive=run_control_study(root/'positive',seed=seed,shifted=False,compact=compact)
    negative=run_control_study(root/'negative',seed=seed+5000,shifted=True,compact=compact)
    accepted=ValidatedQualification.from_directory(root/'positive/qualification')
    rejected=ValidatedQualification.from_directory(root/'negative/qualification')
    accepted.require_admitted()
    if negative['admission']['passed']:
        raise ObservationError('prespecified shifted negative control unexpectedly met acceptance; inspect all raw results')
    a=_control_record(seed+10000,'clear');b=_control_record(seed+20000,'clear',expansion=8.)
    context=_control_record(seed+30000,'smooth')
    oa,ob=control_observation(a),control_observation(b)
    policy=accepted.calibration.policy.measurement_policy
    ga=MeasuredGroup(policy,'milestone-query-a',(ApertureSource(a['image'],('milestone-fresh-a',),APERTURE),))
    gb=MeasuredGroup(policy,'milestone-query-b',(ApertureSource(b['image'],('milestone-fresh-b',),APERTURE),
                                             ApertureSource(context['image'],('milestone-context-b',),APERTURE)))
    write_group(ga,root/'query_group_a');write_group(gb,root/'query_group_b')
    cfg=InstrumentConfig(canvas_size=160,eye_span_px=50,uncertainty_draws=8,enable_flow=False,edge_samples=8)
    result=FacialInstrument(cfg,qualification_evidence=accepted,domain=DOMAIN,acquisition_groups=(ga,gb)).run(a['image'],b['image'],oa,ob)
    bare=FacialInstrument(cfg).run(a['image'],b['image'],oa,ob)
    unchanged=all(np.array_equal(v,result.raw_geometry_arrays()[k]) for k,v in bare.raw_geometry_arrays().items())
    if not unchanged:raise ObservationError('empirical qualification altered raw observed geometry')
    bundle=write_bundle(result,root/'admitted_pair',a['image'],b['image'])
    actual=_so2_minimum(a['truth'][INDICES],b['truth'][INDICES])
    class Spy:
        calls=0
        def observe(self,image):
            self.calls+=1
            raise AssertionError('failed empirical acceptance must stop before observing a query')
    spy=Spy();refusal=None
    try:
        FacialInstrument(cfg,spy,qualification_evidence=rejected,domain=DOMAIN,acquisition_groups=(ga,gb)).run(a['image'],b['image'])
    except InstrumentFailure as exc:
        refusal=str(exc);(root/'withheld_query.events.jsonl').write_text(exc.journal_jsonl,encoding='utf-8')
    summary={'format':'facemath.milestone_laboratory.v10','seed':seed,'kind':'DETERMINISTIC SYNTHETIC WORKFLOW CONTROLS',
        'positive_control':positive,'negative_control':negative,
        'admitted_pair':{'raw_arrays_unchanged':unchanged,'raw_arrays_checked':len(bare.raw_geometry_arrays()),
            'array_count':len(result.arrays()),'known_reference_shape_rms':actual,'shape_bounds':result.intrinsic.shape.to_dict(),
            'known_reference_shape_enclosed':bool(result.intrinsic.shape.available and result.intrinsic.shape.lower-1e-10<=actual<=result.intrinsic.shape.upper+1e-10),
            'verified':verify_bundle(bundle)['verified'],'qualification_snapshot_replayed':True},
        'withheld_query':{'observer_calls':spy.calls,'reason':refusal},
        'limits':['positive branch was engineered, not sampled human performance',
                  'the separately frozen negative branch uses the same acceptance policy and keeps failing cases',
                  'native observer and independent real-reference validation remain outstanding',
                  'raw-state replay is self-consistency, not external attestation of provenance']}
    np.savez_compressed(root/'query_truth.npz',reference=np.stack([a['truth'][INDICES],b['truth'][INDICES]]))
    write_json_new(root/'laboratory.json',summary)
    return summary
