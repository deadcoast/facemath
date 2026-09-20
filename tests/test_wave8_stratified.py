from dataclasses import replace
import copy
import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.stratified import (RoutingPolicy, StratumRadius, StratifiedJointCalibration,
    stratify_joint, evaluate_stratified, validate_routes)
from facemath.observation.joint import fit_joint_calibration
from facemath.observation.bias import load_localization_calibration
from test_wave5_bias import row, BETA


def data_and_cases(seed=8):
    rng=np.random.default_rng(seed);rows=[];cases=[]
    for i in range(100):
        split='development' if i<20 else 'calibration' if i<76 else 'evaluation'
        if split=='calibration':category='clear' if i<56 else 'degraded' if i<72 else 'sparse'
        elif split=='evaluation':category='clear' if i<84 else 'degraded' if i<92 else 'unseen'
        else:category='clear' if i%2 else 'degraded'
        noise=rng.normal(0,.001 if category=='clear' else .008,(8,2))
        rows.append(row(i,split,bias=BETA+noise+rng.normal(0,.02,(1,2))))
        cases.append(dict(case_id=str(i),group_id=str(i),stratum=category,split=split,eligibility='included'))
    return tuple(rows),tuple(cases)


def fitted(seed=8):
    data,cases=data_and_cases(seed)
    c,scored=fit_joint_calibration(data,study_sha256=object_digest({'study':8}),domain='routes')
    p=RoutingPolicy(('clear','degraded','sparse','unseen'),'Prespecified acquisition recipe; not an error-derived cluster',9)
    return stratify_joint(c,scored,cases,p),scored,cases


@pytest.mark.parametrize('n,alpha,minimum,q,status',[(9,.1,9,9.,'supported'),(8,.1,1,None,'unbounded_order_statistic'),
    (10,.1,11,None,'below_frozen_minimum_groups'),(0,.1,1,None,'no_calibration_groups'),(19,.05,1,19.,'supported')])
def test_exact_radius_and_availability(n,alpha,minimum,q,status):
    r=StratumRadius('c',alpha,minimum,tuple((str(i),float(i+1))for i in range(n)),tuple(object_digest({'i':i})for i in range(n)),())
    assert r.quantile==q and r.status==status
    assert StratumRadius.from_dict(r.to_dict())==r


@pytest.mark.parametrize('alpha',[True,False,'0.1',None,np.nan,0.,.5,-1.])
def test_invalid_routing_error_rate(alpha):
    with pytest.raises(ObservationError):StratumRadius('c',alpha,1,(),(),())


@pytest.mark.parametrize('minimum',[0,-1,False,1.5,'9'])
def test_invalid_minimum_group_policy(minimum):
    with pytest.raises(ObservationError):RoutingPolicy(('c',),'Declared protocol',minimum)


def test_whole_group_scores_keep_failures():
    r=StratumRadius('c',.1,1,tuple((str(i),None if i==9 else float(i))for i in range(10)),
        tuple(object_digest({'i':i})for i in range(10)),())
    assert r.quantile is None and r.rank==10 and r.status=='unbounded_order_statistic'


def test_one_development_model_multiple_conditioned_radii():
    m,scored,cases=fitted()
    assert m.quantile is None
    assert m.radius('clear').quantile < m.radius('degraded').quantile
    assert m.radius('sparse').status=='below_frozen_minimum_groups'
    assert m.radius('unseen').status=='no_calibration_groups'
    assert sum(len(r.group_scores)for r in m.radii)==len(m.residual.group_scores)
    assert m.bias.fitted_cases==20


def test_heldout_errors_cannot_select_or_change_category_radii():
    a,scored,cases=fitted();data,_=data_and_cases()
    changed=tuple(row(int(r.case_id),r.split,bias=BETA*100)if r.split=='evaluation'else r for r in data)
    b,s=fit_joint_calibration(changed,study_sha256=object_digest({'study':8}),domain='routes')
    b=stratify_joint(b,s,cases,a.policy)
    assert a.digest==b.digest
    assert evaluate_stratified(a,scored,cases)['evaluation_used_to_select_radii'] is False


def test_evaluation_keeps_unsupported_groups_in_full_denominator():
    m,rows,cases=fitted();report=evaluate_stratified(m,rows,cases)
    assert report['evaluation_groups']==24
    assert report['unsupported_groups']==8 and report['finite_envelope_groups']==16
    assert len(report['cases'])==24
    assert all(r['covered'] is None for r in report['cases']if r['stratum']=='unseen')
    assert report['empirical_group_coverage']==report['covered_groups']/24
    assert report['supported_group_empirical_coverage']==report['covered_groups']/16


def test_calibration_category_group_partition_is_complete_and_nonoverlapping():
    m,scored,cases=fitted();data=m.to_dict();data['category_radii'][0]['group_scores'].pop()
    with pytest.raises(ObservationError):load_localization_calibration(data)


def test_serialization_context_and_determinism():
    m,*_=fitted();restored=load_localization_calibration(m.to_dict())
    assert isinstance(restored,StratifiedJointCalibration) and restored.digest==m.digest
    for fragment in [m.radii[0].to_dict(),m.policy.to_dict()]:
        with pytest.raises(ObservationError):load_localization_calibration(fragment)


@pytest.mark.parametrize('field,value',[('quantile',.00001),('rank',1),('status','supported'),('calibration_groups',100)])
def test_fabricated_category_summary_rejected(field,value):
    m,*_=fitted();data=m.to_dict();data['category_radii'][2][field]=value
    with pytest.raises(ObservationError):load_localization_calibration(data)


def test_partition_does_not_use_altered_evaluation_groups():
    m,rows,cases=fitted();bad=list(cases);bad[-1]={**bad[-1],'group_id':'0'}
    with pytest.raises(ObservationError):evaluate_stratified(m,rows,tuple(bad))


def test_group_with_mixed_categories_rejected_before_observation():
    policy=RoutingPolicy(('a','b'),'pre-acquisition')
    with pytest.raises(ObservationError,match='crosses'):
        validate_routes((dict(case_id='1',group_id='g',stratum='a'),dict(case_id='2',group_id='g',stratum='b')),policy)


@pytest.mark.parametrize('cases',[({'case_id':'a','group_id':'g','stratum':'unknown'},),
    ({'case_id':'a','group_id':'g','stratum':'a'},{'case_id':'a','group_id':'g','stratum':'a'})])
def test_undefined_or_duplicate_routes_rejected(cases):
    with pytest.raises(ObservationError):validate_routes(cases,RoutingPolicy(('a',),'frozen definition'))


def test_routed_evaluation_cannot_drop_difficult_cases():
    m,rows,cases=fitted()
    with pytest.raises(ObservationError,match='entire'):
        evaluate_stratified(m,tuple(r for r in rows if r.case_id!='99'),cases)


def test_wrong_reference_kind_and_observer_recipe_rejected():
    m,rows,cases=fitted()
    for changed in [replace(rows[-1],reference_kind='human_blind'),replace(rows[-1],observer_recipe=object_digest({'wrong':1}))]:
        with pytest.raises(ObservationError):evaluate_stratified(m,rows[:-1]+(changed,),cases)


def test_group_maxima_are_not_case_averages():
    data,cases=data_and_cases();data=list(data);cases=list(cases)
    # Put two same-category calibration cases into one declared dependent group.
    data[22]=replace(data[22],group_id=data[21].group_id)
    cases[22]={**cases[22],'group_id':cases[21]['group_id']}
    base,scored=fit_joint_calibration(tuple(data),study_sha256=object_digest({'group':8}),domain='routes')
    p=RoutingPolicy(('clear','degraded','sparse','unseen'),'preset',1)
    m=stratify_joint(base,scored,tuple(cases),p)
    expected=max(r.simultaneous_score for r in scored if r.group_id=='21')
    assert dict(m.radius('clear').group_scores)['21']==expected


def test_zero_pooled_fallback_even_if_other_category_supported():
    m,*_=fitted()
    with pytest.raises(ObservationError,match='no pooled fallback'):m.radius('brand-new')
    with pytest.raises(ObservationError,match='source-bound'):m.require_query(None,'q','routes')


def test_json_encoded_calibration_roundtrip_preserves_exact_canonical_content():
    import json
    model, _, _ = fitted()
    restored = load_localization_calibration(json.loads(json.dumps(model.to_dict())))
    assert restored.digest == model.digest
    assert restored.policy.digest == model.policy.digest
    assert [r.quantile for r in restored.radii] == [r.quantile for r in model.radii]
