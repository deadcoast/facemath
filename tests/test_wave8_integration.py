"""Falsify source-specific routing across the actual study -> pair -> replay path."""
import argparse
import copy
from dataclasses import replace
import json
import re

import numpy as np
import pytest

from facemath.cli import _replay, build_parser
from facemath.observation.artifacts import write_bundle, verify_bundle
from facemath.observation.bias import load_localization_calibration
from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import ObservationError
from facemath.observation.instrument import FacialInstrument, InstrumentFailure
from facemath.observation.stratified import (StratifiedJointCalibration, BoundStratifiedCalibration, MODE)
from facemath.observation.stratified_lab import make_stratified_study, record, observation, DOMAIN
from facemath.observation.stratified_report import render_routing_study, render_routing_pair
from facemath.observation.intrinsic_report import render_intrinsic_section
from facemath.observation.joint_report import render_joint_section
from facemath.observation.study import freeze_study, read_json
from facemath.observation.study_io import finalize_session, verify_qualification_bundle
from facemath.observation.study_runner import run_session


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root=tmp_path_factory.mktemp('stratified-study')
    path,executor=make_stratified_study(root/'inputs',compact=True)
    frozen=freeze_study(path,root/'frozen')
    run_session(frozen.root,root/'session',executor,maximum_new_cases=3);first=executor.calls
    status=run_session(frozen.root,root/'session',executor);complete=executor.calls
    run_session(frozen.root,root/'session',executor)
    out=finalize_session(root/'session',root/'qualification')
    model=load_localization_calibration(read_json(out/'localization_calibration.json'))
    return root,executor,frozen,status,model,(first,complete)


@pytest.fixture(scope='module')
def pair(study):
    model=study[4];a,b=record(400101,'clear'),record(400102,'degraded',expansion=8)
    oa,ob=observation(a),observation(b)
    cfg=InstrumentConfig(canvas_size=160,eye_span_px=50,uncertainty_draws=8,enable_flow=False,edge_samples=8)
    kwargs=dict(localization_calibration=model,domain=DOMAIN,query_groups=('query-A','query-B'),
                query_lineages=(('fresh-source-A',),('fresh-source-B',)),query_strata=('clear','degraded'))
    result=FacialInstrument(cfg,**kwargs).run(a['image'],b['image'],oa,ob)
    bare=FacialInstrument(cfg).run(a['image'],b['image'],oa,ob)
    return result,bare,a,b,oa,ob,cfg,kwargs


def test_frozen_routing_invokes_no_excluded_or_completed_sources(study):
    root,ex,lock,status,m,calls=study
    assert isinstance(m,StratifiedJointCalibration)
    assert calls==(3,62) and ex.calls==62 and status['prospectively_excluded']==1
    assert status['outcomes']['no_face']==2
    assert lock.data['bias_mode']==MODE
    assert lock.data['stratified_error']==m.policy.to_dict()
    assert verify_qualification_bundle(root/'qualification')['verified']


def test_qualification_uses_same_shared_fit_and_different_group_radii(study):
    root,_,_,_,m,_=study;r=read_json(root/'qualification/qualification.json')
    assert m.radius('clear').quantile<m.radius('degraded').quantile
    assert r['selected_error_model']['radius_mode']=='per_source_acquisition_category'
    assert r['selected_error_model']['quantile'] is None
    assert r['selected_error_model']['coverage']['binomial_95_interval'] is None
    with np.load(root/'qualification/arrays.npz',allow_pickle=False)as z:
        np.testing.assert_array_equal(z['development_joint_covariance'],m.scatter.covariance)
        for i,name in enumerate(m.policy.strata):
            assert z['routing_category_available'][i]==(m.radius(name).quantile is not None)
        assert len(z['routing_calibration_group_score'])==34
        assert (~z['routing_calibration_group_score_available']).sum()==1


def test_qualification_replay_retains_frozen_categories_and_all_failures(study,tmp_path):
    result=_replay(argparse.Namespace(bundle=str(study[0]/'qualification'),out=str(tmp_path/'replay')))
    assert result['replay_verified']


def test_bound_query_preserves_all_observed_arrays(pair):
    r,bare,*_=pair
    for k,v in bare.raw_geometry_arrays().items():np.testing.assert_array_equal(v,r.raw_geometry_arrays()[k],err_msg=k)
    assert len(bare.raw_geometry_arrays())>=135
    assert isinstance(r.localization.calibration,BoundStratifiedCalibration)


def test_fixed_and_refitted_state_use_source_specific_radii_without_rescaling_covariance(pair):
    r,_,_,_,oa,ob,*_=pair;c=r.localization.calibration
    q=[c.source_quantile(oa),c.source_quantile(ob)]
    assert q[0]<q[1] and c.quantile is None
    np.testing.assert_array_equal(r.localization.base.source_quantiles,q)
    np.testing.assert_array_equal(r.intrinsic.source_quantiles,q)
    assert r.intrinsic.to_dict()['calibration_bound_state']=='finite'
    for obs in(oa,ob):
        np.testing.assert_array_equal(c.source_joint_covariance(obs),c.model.source_joint_covariance(obs))
    assert r.intrinsic.shape.available


def test_reverse_pair_keeps_radius_with_source_not_argument_position(pair):
    r,_,a,b,oa,ob,cfg,kwargs=pair
    args={**kwargs,'query_groups':kwargs['query_groups'][::-1], 'query_lineages':kwargs['query_lineages'][::-1],
          'query_strata':kwargs['query_strata'][::-1]}
    reverse=FacialInstrument(cfg,**args).run(b['image'],a['image'],ob,oa)
    np.testing.assert_array_equal(reverse.intrinsic.source_quantiles,r.intrinsic.source_quantiles[::-1])
    np.testing.assert_allclose(reverse.intrinsic.gap_lower,r.intrinsic.gap_lower,atol=1e-10)
    np.testing.assert_allclose(reverse.intrinsic.gap_upper,r.intrinsic.gap_upper,atol=1e-10)
    assert reverse.intrinsic.shape.lower==pytest.approx(r.intrinsic.shape.lower,abs=1e-10)
    assert reverse.intrinsic.shape.upper==pytest.approx(r.intrinsic.shape.upper,abs=1e-10)


def test_bound_model_roundtrip_is_json_stable_and_complete(pair):
    c=pair[0].localization.calibration
    result=load_localization_calibration(json.loads(json.dumps(c.to_dict())))
    assert result.digest==c.digest and result.routes==c.routes
    assert result.quantile is None
    assert result.to_dict()['source_quantiles']==c.to_dict()['source_quantiles']


def test_pair_bundle_replay_recomputes_bound_routes(pair,tmp_path):
    r,_,a,b,*_=pair
    path=write_bundle(r,tmp_path/'pair',a['image'],b['image'])
    assert verify_bundle(path)['verified']
    result=_replay(argparse.Namespace(bundle=str(path),out=str(tmp_path/'replayed')))
    assert result['replay_verified']
    assert read_json(tmp_path/'replayed/localization_calibration.json')==read_json(path/'localization_calibration.json')


@pytest.mark.parametrize('category',['sparse','unseen','made-up'])
def test_unsupported_queries_stop_before_canonical_state_with_no_pooled_fallback(pair,category):
    _,_,a,b,oa,ob,cfg,kwargs=pair
    with pytest.raises(InstrumentFailure)as e:
        FacialInstrument(cfg,**{**kwargs,'query_strata':('clear',category)}).run(a['image'],b['image'],oa,ob)
    events=[json.loads(x)for x in e.value.journal_jsonl.splitlines()]
    assert 'calibration.route_acquisition_sources' in e.value.journal_jsonl
    assert 'execution.complete' not in e.value.journal_jsonl
    assert not any('canonical' in str(x.get('stage','')) for x in events)
    assert any('no pooled fallback' in str(x)or'unknown acquisition' in str(x)for x in events)


def test_unbound_model_cannot_be_assessed_as_if_it_had_one_radius(pair):
    m=pair[0].localization.calibration.model
    with pytest.raises(ObservationError,match='bind acquisition'):m.source_quantile(pair[4])
    with pytest.raises(ObservationError,match='source-bound'):m.require_query(pair[4],'query-A',DOMAIN)


@pytest.mark.parametrize('change',['group','lineage','image','observation'])
def test_source_binding_rejects_substituted_input(pair,change):
    c=pair[0].localization.calibration;o=pair[4];group='query-A';ids=('fresh-source-A',)
    if change=='group':group='renamed-group'
    if change=='lineage':ids=('renamed-lineage',)
    if change=='image':o=replace(o,image_sha256='0'*64)
    if change=='observation':o=replace(o,xy=o.xy+[.01,0],samples=o.samples+[.01,0])
    with pytest.raises(ObservationError):c.require_query(o,group,DOMAIN,ids)


def test_bound_stratum_or_radius_cannot_be_edited_in_export(pair):
    d=pair[0].localization.calibration.to_dict();bad=copy.deepcopy(d)
    bad['source_quantiles'][0]*=2
    with pytest.raises(ObservationError):load_localization_calibration(bad)
    bad=copy.deepcopy(d);bad['routes'][0]['stratum']='degraded'
    with pytest.raises(ObservationError):load_localization_calibration(bad)


@pytest.mark.parametrize('relation',['same_image','same_group','same_lineage'])
def test_dependent_query_group_cannot_receive_conflicting_categories(pair,relation):
    m=pair[0].localization.calibration.model;oa,ob=pair[4:6]
    groups=('a','b');lineages=(('la',),('lb',))
    if relation=='same_image':ob=oa
    if relation=='same_group':groups=('a','a')
    if relation=='same_lineage':lineages=(('same',),('same',))
    with pytest.raises(ObservationError,match='different routing categories'):
        m.bind((oa,ob),groups,('clear','degraded'),DOMAIN,lineages)


def test_global_calibration_leakage_is_checked_across_all_categories(pair):
    m=pair[0].localization.calibration.model;oa,ob=pair[4:6]
    degraded_group=m.radius('degraded').group_scores[0][0]
    with pytest.raises(ObservationError):
        m.bind((oa,ob),(degraded_group,'new'),('clear','degraded'),DOMAIN,((),()))


def test_pair_rendered_ellipses_and_modes_use_their_own_q(pair):
    p=pair[0].intrinsic
    html=render_intrinsic_section(p)
    data=json.loads(re.search(r'<script id="intrinsic-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    for s in range(2):
        for i in range(8):
            expected=p.source_quantiles[s]*np.sqrt(np.linalg.eigvalsh(p.reference_error_covariance[s,i])[-1])
            assert data['display']['ellipse'][s][i][0]==pytest.approx(expected)
    html=render_joint_section(p)
    data=json.loads(re.search(r'<script id="joint-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert data['q']==list(p.source_quantiles)


def test_reports_show_support_not_unbounded_as_a_silent_single_q(study,pair):
    r=read_json(study[0]/'qualification/qualification.json')
    html=render_routing_study(r['acquisition_routing'])
    assert 'routing-category' in html and 'routing-scores' in html and 'no fallback' in html
    assert 'unavailable / +infinity' in html
    p=render_routing_pair(pair[0].localization.calibration)
    assert 'No category selector' in p and '<select' not in p
    assert not any(x in p+html for x in ('fetch(', 'src="http', 'href="http'))


@pytest.mark.parametrize('change',[None,{'definition':'','minimum_calibration_groups':9},
    {'definition':'declared','minimum_calibration_groups':True},{'definition':'declared','minimum_calibration_groups':9,'unknown':1}])
def test_bad_category_policy_rejected_before_any_acquisition(tmp_path,change):
    p,e=make_stratified_study(tmp_path/'inputs',compact=True)
    spec=read_json(p);spec['stratified_error']=change;p.write_text(json.dumps(spec))
    with pytest.raises(ObservationError):freeze_study(p,tmp_path/'frozen')
    assert e.calls==0


def test_group_category_cannot_change_within_prospective_study(tmp_path):
    p,e=make_stratified_study(tmp_path/'inputs',compact=True)
    spec=read_json(p);spec['cases'][1]['group_id']=spec['cases'][0]['group_id'];spec['maximum_cases_per_group']=2
    p.write_text(json.dumps(spec))
    with pytest.raises(ObservationError,match='crosses routing strata'):freeze_study(p,tmp_path/'frozen')
    assert e.calls==0


def test_cli_declares_both_query_categories_not_just_a_global_category():
    args=build_parser().parse_args(['stratified-lab','--out','runs/control'])
    assert args.command=='stratified-lab' and args.seed==28091
    args=build_parser().parse_args(['analyze','a.png','b.png','--query-strata','clear','degraded','--out','runs/pair'])
    assert args.query_strata==['clear','degraded']


def test_mixed_radius_control_retains_all_admissible_points_and_scores(tmp_path):
    from facemath.observation.stratified_lab import run_mixed_radius_controls
    r=run_mixed_radius_controls(tmp_path/'sets',samples=64)
    assert r['sampled_relation_violations']==r['sampled_shape_violations']==0
    assert r['scalar_relation_checks']==64*256
    with np.load(tmp_path/'sets/controls.npz',allow_pickle=False)as z:
        for s in range(2):
            expected=z['source_centers'][s]+(z['source_radii'][s]*z['latent_draws'][s]@z['source_factors'][s].reshape(16,16).T).reshape(64,8,2)
            np.testing.assert_array_equal(expected,z['shape_draws'][s])
    c=r['circle_counterexample']
    assert c['unsafe_smaller_single_radius_upper']<c['admissible_witness_gap']<=c['correct_source_specific_upper']
    assert c['correct_source_specific_upper']<c['overconservative_larger_single_radius_upper']


@pytest.mark.parametrize('alteration',['category','mode','group'])
def test_consistently_rehashed_lock_still_must_obey_routing_contract(study,tmp_path,alteration):
    import shutil
    from facemath.observation.study import load_study,file_inventory
    from facemath.observation.contracts import object_digest
    root=tmp_path/'lock';shutil.copytree(study[2].root,root)
    d=read_json(root/'study.lock.json')
    if alteration=='category':d['strata'].append('undeclared')
    if alteration=='mode':d['bias_mode']='none'
    if alteration=='group':d['cases'][1]['group_id']=d['cases'][0]['group_id']
    (root/'study.lock.json').write_text(json.dumps(d))
    m=read_json(root/'manifest.json');m.pop('content_sha256');m['study_sha256']=object_digest(d)
    m['files']=file_inventory(root,exclude=('manifest.json',));m['content_sha256']=object_digest(m)
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ObservationError):load_study(root)
