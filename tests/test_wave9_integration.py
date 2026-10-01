import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from facemath.cli import _replay, build_parser, execute
from facemath.observation.acquisition import (AcquisitionPolicy, ApertureSource, MeasuredGroup, UNMATCHED,
    write_group, load_group_files)
from facemath.observation.acquisition_lab import make_measured_study, measured_record, APERTURE, DOMAIN
from facemath.observation.stratified_lab import observation
from facemath.observation.stratified import BoundStratifiedCalibration, RoutingPolicy, StratumRadius
from facemath.observation.bias import load_localization_calibration
from facemath.observation.contracts import ObservationError, array_digest, object_digest
from facemath.observation.canonical import InstrumentConfig
from facemath.observation.instrument import FacialInstrument, InstrumentFailure
from facemath.observation.artifacts import write_bundle, verify_bundle
from facemath.observation.study import freeze_study, load_study, read_json, file_inventory
from facemath.observation.study_runner import run_session
from facemath.observation.study_io import finalize_session, verify_qualification_bundle, replay_qualification_bundle


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root=tmp_path_factory.mktemp('measured-integration')
    path,ex=make_measured_study(root/'inputs',compact=True)
    frozen=freeze_study(path,root/'frozen')
    run_session(frozen.root,root/'session',ex,maximum_new_cases=3);first=ex.calls
    status=run_session(frozen.root,root/'session',ex);done=ex.calls
    run_session(frozen.root,root/'session',ex)
    qualified=finalize_session(root/'session',root/'qualification')
    model=load_localization_calibration(read_json(qualified/'localization_calibration.json'))
    return root,ex,frozen,status,model,(first,done)


@pytest.fixture(scope='module')
def pair(study):
    model=study[4];a=measured_record(401501,'clear');b=measured_record(401502,'clear',expansion=8)
    context=measured_record(401503,'smooth')
    ga=MeasuredGroup(model.policy.measurement_policy,'qa',(ApertureSource(a['image'],('qa-lineage',),APERTURE),))
    gb=MeasuredGroup(model.policy.measurement_policy,'qb',(ApertureSource(b['image'],('qb-lineage',),APERTURE),
                                                       ApertureSource(context['image'],('qb-context',),APERTURE)))
    oa,ob=observation(a),observation(b)
    cfg=InstrumentConfig(canvas_size=160,eye_span_px=50,uncertainty_draws=8,enable_flow=False,edge_samples=8)
    kw=dict(localization_calibration=model,domain=DOMAIN,acquisition_groups=(ga,gb))
    result=FacialInstrument(cfg,**kw).run(a['image'],b['image'],oa,ob)
    output=write_bundle(result,study[0]/'pair',a['image'],b['image'])
    return result,output,a,b,oa,ob,cfg,kw


def test_categories_frozen_from_pixels_before_any_observation(study):
    _,_,frozen,_,model,_=study
    assert model.policy.measurement_policy is not None
    assert all('acquisition_group_sha256' in c for c in frozen.cases)
    assert all('acquisition_aperture' in c for c in frozen.cases)
    assert sum(len(g['members'])for g in frozen.data['measured_acquisition']['groups'])==len(frozen.cases)
    spec=read_json(study[0]/'inputs/study.json')
    assert all('stratum'not in c for c in spec['cases'])
    assert {c['stratum']for c in frozen.cases}=={'pixel_clear','pixel_smooth',UNMATCHED}


def test_repeated_group_uses_all_frames_and_no_independent_case_renaming(study):
    frozen=study[2];mixed=[c for c in frozen.cases if c['group_id'].endswith('_mixed')]
    assert len(mixed)==2 and {c['stratum']for c in mixed}=={'pixel_smooth'}
    group=next(g for g in frozen.data['measured_acquisition']['groups']if g['group_id'].endswith('_mixed'))
    assert len(group['members'])==2
    assert max(row[-1]for row in group['features'])>=.30
    assert min(row[-1]for row in group['features'])<.30


def test_idempotent_acquisition_keeps_rejected_and_failed_cases(study):
    _,ex,frozen,status,model,counts=study
    assert counts[0]==3 and counts[1]==status['eligible']==ex.calls
    assert status['prospectively_excluded']==1
    assert status['outcomes']['no_face']==2
    assert status['committed']==len(frozen.included)
    report=read_json(study[0]/'qualification/qualification.json')
    assert len(report['cases'])==status['eligible']
    assert report['measured_acquisition']==frozen.data['measured_acquisition']


def test_unmatched_remains_withheld_even_with_enough_calibration_scores(study):
    radius=study[4].radius(UNMATCHED)
    assert len(radius.group_scores)>=9
    assert radius.quantile is None and radius.status=='rejected_by_measured_acquisition_policy'
    assert not radius.deployment_permitted
    assert StratumRadius.from_dict(radius.to_dict())==radius


def test_measured_policy_and_rejected_model_context_roundtrip(study):
    model=study[4];r=load_localization_calibration(json.loads(json.dumps(model.to_dict())))
    assert r.digest==model.digest and r.policy.measurement_policy.digest==model.policy.measurement_policy.digest
    assert r.radius(UNMATCHED).quantile is None
    assert RoutingPolicy.from_dict(r.policy.to_dict()).digest==r.policy.digest


def test_cannot_strip_rejected_permission_and_make_a_radius_deployable(study):
    value=study[4].to_dict()
    radius=next(r for r in value['category_radii']if r['stratum']==UNMATCHED)
    radius.pop('deployment_permitted')
    with pytest.raises(ObservationError):load_localization_calibration(value)


def test_pixel_routing_does_not_change_any_observed_geometry(pair):
    result,_,a,b,oa,ob,cfg,_=pair
    bare=FacialInstrument(cfg).run(a['image'],b['image'],oa,ob)
    assert len(bare.raw_geometry_arrays())>=135
    for k,v in bare.raw_geometry_arrays().items():np.testing.assert_array_equal(v,result.raw_geometry_arrays()[k],err_msg=k)
    assert [g.category for g in result.acquisition_groups]==['pixel_clear','pixel_smooth']
    assert isinstance(result.localization.calibration,BoundStratifiedCalibration)


def test_measurement_stage_precedes_observer_and_binding(pair):
    records=[json.loads(x)for x in pair[0].journal_jsonl.splitlines()]
    stages=[r.get('stage',r.get('name',''))for r in records]
    text=pair[0].journal_jsonl
    assert text.index('acquisition.measure_original_sources_before_observer')<text.index('observation.acquire')<text.index('calibration.route_acquisition_sources')


def test_query_category_uses_all_declared_context_not_selected_image(pair):
    r=pair[0];gb=r.acquisition_groups[1]
    selected=next(s for s in gb.sources if s.image_sha256==pair[5].image_sha256)
    assert MeasuredGroup(gb.policy,'single',(selected,)).category=='pixel_clear'
    assert gb.category=='pixel_smooth'
    np.testing.assert_array_equal(r.intrinsic.source_quantiles,r.localization.base.source_quantiles)
    assert r.intrinsic.source_quantiles[0]<r.intrinsic.source_quantiles[1]


def test_original_source_pixels_survive_pair_bundle(pair):
    r,out,*_=pair
    groups=tuple(load_group_files(out,f'acquisition_{n}_')for n in('a','b'))
    assert [g.digest for g in groups]==[g.digest for g in r.acquisition_groups]
    assert verify_bundle(out)['verified']


def test_saved_route_without_raw_pixels_is_not_authorized(pair):
    c=load_localization_calibration(pair[0].localization.calibration.to_dict())
    assert c.measured_groups is None
    with pytest.raises(ObservationError,match='pixel revalidation'):c.source_quantile(pair[4])
    with pytest.raises(ObservationError,match='pixel revalidation'):c.require_query(pair[4],'qa',DOMAIN,('qa-lineage',))


def test_pair_replay_remeasures_apertures_and_recomputes_shape(pair,tmp_path):
    out=_replay(argparse.Namespace(bundle=str(pair[1]),out=str(tmp_path/'replay')))
    assert out['replay_verified']
    new=read_json(tmp_path/'replay/evidence.json')
    assert new['measured_acquisition']==pair[0].evidence()['measured_acquisition']


def test_qualification_replay_reuses_predictions_but_remeasures_source_routes(study,tmp_path):
    assert verify_qualification_bundle(study[0]/'qualification')['verified']
    result=replay_qualification_bundle(study[0]/'qualification',tmp_path/'qualification-replay')
    assert result['replay_verified'] and not result['live_inference_reexecuted']


@pytest.mark.parametrize('strata',[('pixel_clear','pixel_clear'),('__unmatched__','pixel_smooth'),('desired','desired')])
def test_manual_category_override_refused_before_geometry(pair,strata):
    _,_,a,b,oa,ob,cfg,kw=pair
    with pytest.raises(InstrumentFailure,match='conflicts'):
        FacialInstrument(cfg,**kw,query_strata=strata).run(a['image'],b['image'],oa,ob)


def test_unknown_source_cannot_reuse_an_existing_measured_receipt(pair):
    _,_,a,b,oa,ob,cfg,kw=pair
    image=a['image'].copy();image[0,0,0]^=1
    with pytest.raises(InstrumentFailure,match='not a member'):
        FacialInstrument(cfg,**kw).run(image,b['image'],oa,ob)


def test_complete_group_lineages_cannot_be_omitted(pair):
    _,_,a,b,oa,ob,cfg,kw=pair
    with pytest.raises(InstrumentFailure,match='complete'):
        FacialInstrument(cfg,**kw,query_lineages=(('qa-lineage',),('qb-lineage',))).run(a['image'],b['image'],oa,ob)


def test_context_image_calibration_leak_is_checked_before_observer(pair,study):
    r,_,a,b,oa,ob,cfg,kw=pair
    case=next(c for c in study[2].included if c['split']=='calibration'and c['stratum']=='pixel_clear')
    saved=study[2].image(case);gb=r.acquisition_groups[1]
    altered=MeasuredGroup(gb.policy,gb.group_id,(ApertureSource(b['image'],('qb-lineage',),APERTURE),
        ApertureSource(saved,tuple(case['lineage_ids']),APERTURE)))
    with pytest.raises(InstrumentFailure,match='leaks calibration'):
        FacialInstrument(cfg,**{**kw,'acquisition_groups':(r.acquisition_groups[0],altered)}).run(a['image'],b['image'],oa,ob)


def test_different_source_inventories_cannot_claim_the_same_group(pair):
    r,_,a,b,oa,ob,cfg,kw=pair
    ga,gb=r.acquisition_groups
    gb=MeasuredGroup(gb.policy,ga.group_id,gb.sources)
    with pytest.raises(InstrumentFailure,match='different measured source inventories'):
        FacialInstrument(cfg,**{**kw,'acquisition_groups':(ga,gb)}).run(a['image'],b['image'],oa,ob)


def test_policy_change_cannot_silently_relabel_a_query(pair):
    r,_,a,b,oa,ob,cfg,kw=pair;ga,gb=r.acquisition_groups
    changed=MeasuredGroup(replace(ga.policy,definition='altered'),ga.group_id,ga.sources)
    with pytest.raises(InstrumentFailure,match='policy differs'):
        FacialInstrument(cfg,**{**kw,'acquisition_groups':(changed,gb)}).run(a['image'],b['image'],oa,ob)


def test_unmatched_source_stops_before_any_observer_call(pair):
    r,_,a,b,oa,ob,cfg,kw=pair
    bad=measured_record(499901,'washed');g=MeasuredGroup(r.acquisition_groups[0].policy,'reject',(ApertureSource(bad['image'],('reject',),APERTURE),))
    class Spy:
        calls=0
        def observe(self,x):self.calls+=1;raise AssertionError('observer called')
    spy=Spy()
    with pytest.raises(InstrumentFailure,match='unsupported')as err:
        FacialInstrument(cfg,spy,**{**kw,'acquisition_groups':(r.acquisition_groups[0],g)}).run(a['image'],bad['image'])
    assert spy.calls==0
    assert 'observation.acquire' not in err.value.journal_jsonl
    assert 'execution.complete' not in err.value.journal_jsonl


def test_unqualified_nominal_analysis_cannot_silently_inherit_a_gate(pair):
    r,_,a,b,oa,ob,cfg,_=pair
    with pytest.raises(InstrumentFailure,match='measured calibration'):
        FacialInstrument(cfg,acquisition_groups=r.acquisition_groups).run(a['image'],b['image'],oa,ob)


def test_reversing_sources_keeps_their_own_pixels_radii_and_shape_bounds(pair):
    r,_,a,b,oa,ob,cfg,kw=pair
    reverse=FacialInstrument(cfg,**{**kw,'acquisition_groups':kw['acquisition_groups'][::-1]}).run(b['image'],a['image'],ob,oa)
    np.testing.assert_array_equal(reverse.intrinsic.source_quantiles,r.intrinsic.source_quantiles[::-1])
    np.testing.assert_allclose(reverse.intrinsic.gap_lower,r.intrinsic.gap_lower,atol=1e-10)
    np.testing.assert_allclose(reverse.intrinsic.gap_upper,r.intrinsic.gap_upper,atol=1e-10)


def test_hidden_shift_same_pixel_contract_retained_in_evaluation(study):
    report=read_json(study[0]/'qualification/qualification.json')
    contract=read_json(study[0]/'inputs/fixture_contract.json')
    shifted={x['case_id']for x in contract['per_source']if x['hidden_error_shift']}
    cases=report['selected_error_model']['coverage']['cases']
    selected=[c for c in cases if c['case_id']in shifted]
    assert selected and all(c['stratum']=='pixel_clear'for c in selected)
    assert any(not c['covered']for c in selected)
    assert len(cases)==sum(c['split']=='evaluation' for c in study[2].included)


def test_cli_has_source_groups_and_no_required_manual_category():
    p=build_parser()
    a=p.parse_args(['analyze','a.png','b.png','--acquisition-a','ga','--acquisition-b','gb','--domain','d','--out','x'])
    assert a.acquisition_a=='ga'and a.query_strata is None
    assert p.parse_args(['acquisition-group','spec.json','--policy','p.json','--out','x']).command=='acquisition-group'
    assert p.parse_args(['measured-lab','--out','x']).command=='measured-lab'


def test_cli_refused_pixel_route_precedes_native_model_construction(study,tmp_path,monkeypatch):
    import facemath.cli as cli
    from facemath.observation.acquisition import write_group
    from facemath.observation.study import save_rgb
    model=study[4]
    policy=model.policy.measurement_policy
    a=measured_record(401101,"washed")["image"]
    b=measured_record(401102,"clear")["image"]
    ga=MeasuredGroup(policy,"lazy-a",(ApertureSource(a,("lazy-a",),APERTURE),))
    gb=MeasuredGroup(policy,"lazy-b",(ApertureSource(b,("lazy-b",),APERTURE),))
    write_group(ga,tmp_path/"a-group");write_group(gb,tmp_path/"b-group")
    save_rgb(tmp_path/"a.png",a);save_rgb(tmp_path/"b.png",b)
    model_path=tmp_path/"calibration.json";model_path.write_text(json.dumps(model.to_dict()))
    def forbidden(*args,**kwargs):
        pytest.fail("native model constructed before source-policy refusal")
    monkeypatch.setattr(cli,"MediaPipeProvider",forbidden)
    args=cli.build_parser().parse_args(["analyze",str(tmp_path/"a.png"),str(tmp_path/"b.png"),
        "--model",str(tmp_path/"absent.task"),"--localization-calibration",str(model_path),
        "--domain",DOMAIN,"--acquisition-a",str(tmp_path/"a-group"),"--acquisition-b",str(tmp_path/"b-group"),
        "--out",str(tmp_path/"out")])
    with pytest.raises(InstrumentFailure,match="unsupported.*rejected_by_measured_acquisition_policy"):
        cli._analyze(args)
