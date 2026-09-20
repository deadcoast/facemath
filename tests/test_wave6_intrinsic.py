from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.error_model import LocalizationCalibration
from facemath.observation.instrument import FacialInstrument
from facemath.observation.intrinsic import (
    _bound, bound_refitted_shape, measure_intrinsic, nominal_values,
    proper_shape_distance, relation_registry,
)
from facemath.observation.intrinsic_sets import (
    EYE_VECTOR, EllipticSource, SourceRelations, absolute_difference,
)
from facemath.observation.lab import phantom_points, phantom_observation, render_phantom
from facemath.observation.reference import observer_recipe
from test_wave6_sets import POINTS, draws, source


def inputs(delta=0., sigma=.1):
    pa=phantom_points();pb=pa.copy();pb[14,0]-=delta;pb[15,0]+=delta
    a=render_phantom(pa,seed=880);b=render_phantom(pb,seed=881)
    return a,b,phantom_observation(a,pa,sigma),phantom_observation(b,pb,sigma)


def configuration(**kwargs):
    return replace(InstrumentConfig(uncertainty_draws=4,enable_flow=False,enable_edges=False),**kwargs)


def calibration(observation,q=2):
    """Hand-specified synthetic error-set fixture, NOT an empirical human cohort."""
    return LocalizationCalibration(.1,q,9,tuple((f'cal-{i}',q) for i in range(9)),
        observer_recipe(observation),object_digest({'synthetic_set_fixture':True}),
        'analytic-test-only','synthetic_fixture',(),(),())


def run_pair(delta=0., q=2., config=None):
    a,b,oa,ob=inputs(delta)
    c=calibration(oa,q) if q is not None else None
    r=FacialInstrument(config or configuration(),localization_calibration=c,
        query_groups=('query-a','query-b'),domain='analytic-test-only').run(a,b,oa,ob)
    return r,(a,b,oa,ob)


def independent_minimum(a,b):
    """Closed-form SO(2) objective independent of the implementation's SVD."""
    a=(a-a.mean(0))/np.linalg.norm(EYE_VECTOR@a)
    b=(b-b.mean(0))/np.linalg.norm(EYE_VECTOR@b)
    m=a.T@b
    best=np.hypot(m[0,0]+m[1,1],m[1,0]-m[0,1])
    return np.sqrt(max(0.,(np.sum(a*a)+np.sum(b*b)-2*best)/8))


@pytest.mark.parametrize('seed',range(8))
def test_proper_shape_distance_matches_independent_so2_solution(seed):
    rng=np.random.default_rng(seed)
    a=POINTS+rng.normal(0,.03,(8,2));b=POINTS+rng.normal(0,.07,(8,2))
    value,r,_=proper_shape_distance(a,b)
    assert np.linalg.det(r)==pytest.approx(1)
    assert value==pytest.approx(independent_minimum(a,b),abs=1e-12)


def test_reflection_is_not_normalized_away():
    mirror=POINTS*np.array([-1,1])
    d,r,_=proper_shape_distance(POINTS,mirror)
    assert d>.2 and np.linalg.det(r)>0
    a=np.linalg.norm(POINTS[:,None]-POINTS[None,:],axis=-1)
    b=np.linalg.norm(mirror[:,None]-mirror[None,:],axis=-1)
    np.testing.assert_array_equal(a,b)
    registry=relation_registry(configuration())
    va,ok=nominal_values(POINTS,np.ones(8,bool),registry)
    vb,_=nominal_values(mirror,np.ones(8,bool),registry)
    area=np.array([r.family=='signed_double_area' for r in registry])
    np.testing.assert_allclose(va[area],-vb[area])
    assert np.max(abs(va[area]-vb[area]))>.3


@pytest.mark.parametrize('seed',range(7))
def test_refitted_shape_outer_bounds_contain_independently_refitted_reference_draws(seed):
    sa=source(seed,magnitude=.02)
    p=POINTS.copy();p[5,0]-=.1;p[6,0]+=.1;p[4,1]+=.03
    sb=source(seed+70,magnitude=.03,points=p)
    ea,eb=SourceRelations(sa),SourceRelations(sb)
    registry=relation_registry(configuration())
    gaps=[absolute_difference(_bound(ea,r),_bound(eb,r)).lower for r in registry[:28]]
    bound=bound_refitted_shape(np.stack([sa.centers,sb.centers]),(ea,eb),np.array(gaps),True)
    assert bound.available
    aa,bb=draws(sa,300,seed),draws(sb,300,seed+1)
    values=[independent_minimum(a,b) for a,b in zip(aa,bb)]
    assert min(values)>=bound.lower-1e-11
    assert max(values)<=bound.upper+1e-11
    assert bound.lower<=bound.centered_rms<=bound.upper


def test_exact_set_limit_collapses_to_refitted_metric_not_just_edm_lower_bound():
    p=POINTS.copy();p[4,0]+=.1
    a,b=source(q=0),source(q=0,points=p)
    ea,eb=SourceRelations(a),SourceRelations(b)
    specs=relation_registry(configuration())
    gaps=np.array([absolute_difference(_bound(ea,r),_bound(eb,r)).lower for r in specs[:28]])
    bound=bound_refitted_shape(np.stack([a.centers,b.centers]),(ea,eb),gaps,True)
    exact=independent_minimum(a.centers,b.centers)
    assert bound.upper-bound.lower < 1e-10
    assert bound.lower<=exact+1e-13 and bound.upper>=exact-1e-13


def test_all_relations_share_one_eight_point_state_not_independent_scores():
    result,_=run_pair(delta=3)
    report=result.intrinsic.to_dict()
    assert len(report['rows'])==256
    assert [x['prespecified_relations'] for x in report['families']]==[28,168,56,4]
    assert report['population_similarity_percent'] is None
    assert report['conditional_pair_coverage_lower_bound']==pytest.approx(.8)
    assert report['observation_digests']==[result.state.observation_a.digest,result.state.observation_b.digest]
    assert 'intrinsic_source_covariance' in result.arrays()
    assert 'measure.intrinsic_reference_shapes' in result.journal_jsonl


def test_uncalibrated_observed_relations_stay_available_without_fake_error_bounds():
    result,_=run_pair(q=None)
    p=result.intrinsic
    assert np.all(p.nominal_valid)
    assert not np.any(p.gap_available)
    assert p.shape.observed_rms is not None
    assert not p.shape.available
    assert all(f['within_tolerance_fraction_bounds'] is None for f in p.to_dict()['families'])
    assert np.all(p.statuses==0)


def test_attaching_intrinsic_profile_does_not_change_original_raw_arrays():
    result,args=run_pair(delta=2)
    legacy,_=run_pair(delta=2,config=configuration(enable_intrinsic=False))
    expected=legacy.arrays()
    for k,v in expected.items():
        np.testing.assert_array_equal(result.arrays()[k],v,err_msg=k)
    assert result.profile.measurements==legacy.profile.measurements
    assert result.intrinsic.shape.available


def test_raw_intrinsic_values_do_not_change_on_calibration_attachment():
    bare,_=run_pair(delta=4,q=None);qualified,_=run_pair(delta=4,q=3)
    np.testing.assert_array_equal(bare.intrinsic.nominal,qualified.intrinsic.nominal)
    np.testing.assert_array_equal(bare.intrinsic.raw_points,qualified.intrinsic.raw_points)
    assert bare.intrinsic.shape.observed_rms==qualified.intrinsic.shape.observed_rms


def test_exchange_order_preserves_abs_relation_gaps_and_refitted_shape_range():
    r,args=run_pair(delta=4)
    a,b,oa,ob=args
    s=FacialInstrument(configuration(),localization_calibration=calibration(oa),
        query_groups=('query-b','query-a'),domain='analytic-test-only').run(b,a,ob,oa)
    np.testing.assert_allclose(r.intrinsic.gap_lower,s.intrinsic.gap_lower,atol=1e-12)
    np.testing.assert_allclose(r.intrinsic.gap_upper,s.intrinsic.gap_upper,atol=1e-12)
    assert r.intrinsic.shape.lower==pytest.approx(s.intrinsic.shape.lower,abs=1e-12)
    assert r.intrinsic.shape.upper==pytest.approx(s.intrinsic.shape.upper,abs=1e-12)


def test_canonical_display_size_and_robust_fit_settings_do_not_change_intrinsic_relations():
    a,_=run_pair(delta=2)
    b,_=run_pair(delta=2,config=configuration(canvas_size=384,eye_span_px=120,huber_delta_ipd=.2,fit_iterations=3))
    for key,array in a.intrinsic.arrays().items():
        np.testing.assert_allclose(array,b.intrinsic.arrays()[key],atol=1e-12,err_msg=key)


@pytest.mark.parametrize('scale,angle',[(.8,-.1),(1.2,.2),(1.,.5)])
def test_raw_intrinsic_relations_ignore_separate_global_similarity_transforms(scale,angle):
    a,b,oa,ob=inputs(delta=2)
    raw=FacialInstrument(configuration()).run(a,b,oa,ob).intrinsic
    r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    p=(oa.xy-[128,100])@r.T*scale+[170,140]
    new=render_phantom(p,shape=(400,400),seed=889)
    other=FacialInstrument(configuration()).run(new,b,phantom_observation(new,p),ob).intrinsic
    np.testing.assert_allclose(raw.nominal,other.nominal,atol=1e-12)
    assert raw.shape.observed_rms==pytest.approx(other.shape.observed_rms,abs=1e-12)


def test_normalizer_uncertainty_collapse_does_not_emit_finite_shape_bounds():
    r,_=run_pair(delta=0,q=1000)
    assert not r.intrinsic.shape.available
    assert r.intrinsic.shape.reason=='normalizer_not_separated_from_zero'
    assert r.intrinsic.shape.upper is None
    ids=[i for i,x in enumerate(r.intrinsic.relations) if x.family in ('distance','signed_double_area')]
    assert not np.any(r.intrinsic.gap_available[ids])
    assert np.all(r.intrinsic.nominal_valid[:,ids])


def test_missing_reference_vertex_does_not_shrink_denominators_or_imply_agreement():
    r,args=run_pair()
    a,b,oa,ob=args
    p=ob.xy.copy();p[ob.names.index('chin')]=[-1,-1]
    ob=replace(ob,xy=p,samples=p[None])
    bad=FacialInstrument(configuration(),localization_calibration=calibration(oa),
        query_groups=('a','b'),domain='analytic-test-only').run(a,b,oa,ob)
    report=bad.intrinsic.to_dict()
    assert not bad.intrinsic.shape.available
    distances=report['families'][0]
    assert distances['prespecified_relations']==28
    assert distances['bounded_relations']==21
    assert distances['unresolved_or_unavailable']>=7


@pytest.mark.parametrize('change',[
    {'enable_intrinsic':1},{'intrinsic_directions':33},{'intrinsic_directions':0},
    {'intrinsic_length_tolerance':-1},{'intrinsic_area_tolerance':float('nan')},
    {'intrinsic_angle_tolerance_rad':4},{'intrinsic_asymmetry_tolerance':3},
])
def test_new_configuration_contracts(change):
    with pytest.raises(ObservationError):configuration(**change)


def test_output_arrays_cannot_be_mutated_or_unfrozen():
    result,_=run_pair()
    with pytest.raises(ValueError):result.intrinsic.gap_lower[0]=2
    with pytest.raises(ValueError):result.intrinsic.source_covariance.flags.writeable=True


def test_identical_exact_points_all_relations_agree_without_becoming_similarity_probability():
    result,_=run_pair(delta=0,q=0)
    p=result.intrinsic.to_dict()
    assert all(f['within']==f['prespecified_relations'] for f in p['families'])
    assert p['shape_distance']['upper']<1e-10
    assert p['population_similarity_percent'] is None


def test_unbounded_calibration_is_not_misrepresented_as_no_calibration():
    a,b,oa,ob=inputs()
    c=calibration(oa,None)
    with pytest.raises(ObservationError, match='unbounded'):
        FacialInstrument(configuration(),localization_calibration=c,
            query_groups=('query-a','query-b'),domain='analytic-test-only').run(a,b,oa,ob)
    # The low-level profile remains truthful even when explicitly handed an
    # unbounded error object; the production instrument refuses it upstream.
    result,_=run_pair()
    profile=measure_intrinsic(result.state,replace(result.localization,calibration=c))
    data=profile.to_dict()
    assert data['calibrated']
    assert data['calibration_bound_state']=='unbounded'
    assert not profile.shape.available
    assert profile.shape.reason=='unbounded_localization_calibration'
    assert np.all(profile.statuses==4)
    assert all(f['within_tolerance_fraction_bounds']==[0.,1.] for f in data['families'])


@pytest.mark.parametrize('name', ['raw_points','source_centers','source_chart','gap_available','statuses'])
def test_malformed_profile_arrays_are_rejected(name):
    from dataclasses import replace
    r,_=run_pair()
    with pytest.raises(ObservationError,match='shape'):
        replace(r.intrinsic,**{name:getattr(r.intrinsic,name)[:1]})


def test_exported_intrinsic_array_collection_is_immutable():
    r,_=run_pair()
    for array in r.intrinsic.arrays().values():
        assert not array.flags.writeable
        with pytest.raises(ValueError):array.flags.writeable=True
