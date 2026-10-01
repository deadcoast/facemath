from dataclasses import replace
from types import SimpleNamespace
import json
import math

import numpy as np
import pytest
from scipy.stats import binomtest, binom

from facemath.observation.admission import (AdmissionPolicy, lower_binomial, plan_admission, assess_admission)
from facemath.observation.contracts import ObservationError


def policy(**kw):
    return AdmissionPolicy(("a", "b"), minimum_groups=10, minimum_delivery=.6,
        minimum_coverage=.6, family_error=.05, evidence_kind="synthetic_control",
        independent_groups_declared=True, sampling_design="Independent scripted test groups", **kw)


@pytest.mark.parametrize('k,n', [(0,1),(1,1),(0,30),(1,30),(14,30),(29,30),(30,30),(73,100),(990,1000)])
@pytest.mark.parametrize('tail', [.025, .0125, .001])
def test_exact_one_sided_lower_matches_independent_scipy_contract(k,n,tail):
    expected=binomtest(k,n,alternative='greater').proportion_ci(confidence_level=1-tail,method='exact').low
    assert lower_binomial(k,n,tail)==pytest.approx(expected,abs=2e-12)  # scipy exact CI uses a root solve with ~2e-12 absolute tolerance


def test_no_trials_is_missing_not_zero_accuracy():
    assert lower_binomial(0,0,.01) is None


@pytest.mark.parametrize('args',[(True,4,.01),(1,True,.01),(-1,4,.01),(5,4,.01),(1,4,True),(1,4,0),(1,4,1),(1,4,float('nan')),(1.0,4,.01)])
def test_invalid_statistical_inputs_rejected(args):
    with pytest.raises(ObservationError):lower_binomial(*args)


@pytest.mark.parametrize('target',[.2,.6,.8,.9,.95])
def test_all_success_feasibility_is_mathematically_minimal(target):
    p=replace(policy(),minimum_groups=1,minimum_delivery=target,minimum_coverage=target)
    n=p.perfect_minimum
    assert lower_binomial(n,n,p.tail)>=target-1e-14
    if n>1:assert lower_binomial(n-1,n-1,p.tail)<target+1e-14


def test_policy_roundtrip_exact_and_immutable():
    p=policy(); assert AdmissionPolicy.from_dict(json.loads(json.dumps(p.to_dict())))==p
    obj=p.to_dict();obj['required_categories'].append('c');assert p.required_categories==('a','b')
    with pytest.raises(Exception):p.minimum_groups=1


@pytest.mark.parametrize('change',[{'minimum_groups':True},{'minimum_groups':0},{'minimum_groups':1.1},
    {'minimum_coverage':True},{'minimum_coverage':'0.8'},{'minimum_delivery':float('inf')},
    {'family_error':0.},{'evidence_kind':'real-looking'},{'independent_groups_declared':1},
    {'sampling_design':''},{'required_categories':('a','a')},{'required_categories':('a',{})}])
def test_policy_strict_contract(change):
    with pytest.raises(ObservationError):replace(policy(),**change)


def test_no_extra_serialized_policy_field():
    obj=policy().to_dict();obj['approve']=True
    with pytest.raises(ObservationError):AdmissionPolicy.from_dict(obj)


def trial(n=20,failures=0,second_case=False):
    cases=[];rows=[]
    for category in ('a','b','outside'):
        for i in range(n):
            group=f'{category}-{i}'
            for j in range(2 if second_case else 1):
                cid=f'{group}-{j}'
                cases.append(dict(case_id=cid,group_id=group,stratum=category,split='evaluation',eligibility='included'))
                score=None if i<failures and j==0 else .5
                rows.append(SimpleNamespace(case_id=cid,group_id=group,split='evaluation',simultaneous_score=score,
                    observed=np.ones(8,bool) if score is not None else np.zeros(8,bool),reference_visible=np.ones(8,bool),rater_count=np.ones(8,int)))
    study=dict(cases=cases,strata=['a','b','outside'],observer=dict(execution_kind='synthetic_fixture'),minimum_raters=1)
    model=SimpleNamespace(digest='f'*64,radius=lambda _:SimpleNamespace(quantile=1.))
    return study,rows,model


def test_group_max_denominator_not_case_count_and_failed_attempts_stay():
    study,rows,model=trial(failures=2,second_case=True)
    out,arrays=assess_admission(policy(),study,rows,model)
    assert all(r['groups']==20 and r['covered']==18 and r['delivered']==18 for r in out['categories'])
    assert out['all_evaluation_groups']==60 and out['outside_requested_scope_groups']==20
    np.testing.assert_array_equal(arrays['admission_group_case_count'],2)
    assert out['multiplicity']==4 and out['tail_probability_per_endpoint']==.0125


def test_all_required_categories_must_pass_no_cherry_pick():
    study,rows,model=trial()
    for r in rows:
        if r.group_id.startswith('b'):r.simultaneous_score=2.
    out,_=assess_admission(policy(),study,rows,model)
    assert out['categories'][0]['passes'] and not out['categories'][1]['passes']
    assert out['status']=='withheld' and not out['passed']
    assert out['categories'][1]['delivered']==20 and out['categories'][1]['covered']==0


def test_missing_category_has_unavailable_interval_and_no_approval():
    study,rows,model=trial()
    study['cases']=[c for c in study['cases'] if c['stratum']!='b']
    rows=[r for r in rows if not r.group_id.startswith('b')]
    out,_=assess_admission(policy(),study,rows,model)
    assert out['categories'][1]['groups']==0 and out['categories'][1]['coverage_lower'] is None
    assert not out['passed']


def test_infinite_selected_radius_cannot_become_delivery():
    study,rows,model=trial();model.radius=lambda _:SimpleNamespace(quantile=None)
    out,_=assess_admission(policy(),study,rows,model)
    assert out['all_evaluation_covered']==out['all_evaluation_delivered']==0


def test_unobservable_reference_does_not_reduce_denominator():
    study,rows,model=trial();rows[0].reference_visible[7]=False
    out,_=assess_admission(policy(),study,rows,model)
    assert out['categories'][0]['groups']==20 and out['categories'][0]['covered']==19


def test_unasserted_sampling_assumption_withholds_even_perfect():
    study,rows,model=trial()
    out,_=assess_admission(replace(policy(),independent_groups_declared=False),study,rows,model)
    assert all(r['covered']==20 for r in out['categories']) and not out['passed']


def test_native_cannot_borrow_fixture_provenance():
    study,rows,model=trial();study['observer']['execution_kind']='native_mediapipe'
    with pytest.raises(ObservationError,match='provenance'):assess_admission(policy(),study,rows,model)


def test_duplicate_or_omitted_evaluation_rows_rejected():
    study,rows,model=trial()
    with pytest.raises(ObservationError,match='inventory'):assess_admission(policy(),study,rows[:-1],model)
    with pytest.raises(ObservationError,match='inventory'):assess_admission(policy(),study,rows+[rows[0]],model)


def test_admission_order_invariant():
    study,rows,model=trial(failures=2)
    first,_=assess_admission(policy(),study,rows,model)
    second,_=assess_admission(policy(),study,list(reversed(rows)),model)
    assert first==second


def test_more_frozen_categories_increases_multiplicity_not_optimism():
    p=policy();extra=replace(p,required_categories=('a','b','c'))
    assert extra.tail<p.tail and lower_binomial(18,20,extra.tail)<lower_binomial(18,20,p.tail)

@pytest.mark.parametrize("n",[1,2,20,24,100,1000])
@pytest.mark.parametrize("tail",[.025,.001])
def test_all_success_closed_form_and_tail_equation(n,tail):
    bound=lower_binomial(n,n,tail)
    assert bound==pytest.approx(tail**(1/n),abs=2e-15)
    assert binom.sf(n-1,n,bound)==pytest.approx(tail,rel=5e-12,abs=2e-15)

@pytest.mark.parametrize("k,n",[(1,20),(12,20),(73,100)])
def test_bound_solves_exact_binomial_tail(k,n):
    bound=lower_binomial(k,n,.0125)
    assert binom.sf(k-1,n,bound)==pytest.approx(.0125,rel=5e-13,abs=2e-15)
