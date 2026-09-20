from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from facemath.observation.contracts import ObservationError
from facemath.observation.intrinsic_sets import (
    EYE_VECTOR, EllipticSource, Interval, SourceRelations, absolute_difference,
    coefficients, displacement, divide,
)

POINTS = np.array([[-.7,0],[-.3,0],[.3,0],[.7,0],[.02,.65],[-.32,1.05],[.36,1.06],[.03,1.7]])


def source(seed=0, q=1., magnitude=.015, points=POINTS, directions=128):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(8,2,2))*magnitude
    cov = factor @ factor.swapaxes(-1,-2)
    return EllipticSource(points, cov, np.ones(8,bool), q, directions)


def draws(s, count=1200, seed=12):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2*np.pi, (count,8))
    radii = np.sqrt(rng.uniform(size=(count,8)))
    radii[:count//3] = 1.  # Include boundaries, not only interior samples.
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=-1)*radii[...,None]
    eig, vec = np.linalg.eigh(s.covariance)
    root = (vec*np.sqrt(np.maximum(eig,0))[:,None,:]) @ vec.swapaxes(-1,-2)
    return s.centers+s.quantile*np.einsum('kni,nji->knj',unit,root)


def contained(interval, values):
    assert interval.available
    assert np.min(values) >= interval.lower-1e-11
    assert np.max(values) <= interval.upper+1e-11


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('directions', [32,128])
def test_linear_combination_bounds_contain_elliptic_boundary_draws(seed,directions):
    s=source(seed,directions=directions)
    rng=np.random.default_rng(seed+16)
    a=rng.normal(size=8);a-=a.mean()
    points=draws(s)
    values=np.linalg.norm(np.einsum('n,knd->kd',a,points),axis=1)
    e=SourceRelations(s)
    contained(e.norm(a),values)
    assert e.norm(a)==e.norm(-a)


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('kind', ['dot','cross'])
def test_bilinear_second_order_bounds_include_shared_apex(seed,kind):
    s=source(seed,magnitude=.07)
    p=draws(s)
    a,b=displacement(0,4),displacement(7,4)
    u=np.einsum('n,knd->kd',a,p);v=np.einsum('n,knd->kd',b,p)
    value=(u*v).sum(1) if kind=='dot' else u[:,0]*v[:,1]-u[:,1]*v[:,0]
    contained(SourceRelations(s).bilinear(a,b,kind),value)


@pytest.mark.parametrize('seed', range(6))
def test_angles_include_all_sampled_true_angles(seed):
    s=source(seed,magnitude=.025);p=draws(s)
    e=SourceRelations(s);a,b=displacement(0,4),displacement(3,4)
    u=p[:,0]-p[:,4];v=p[:,3]-p[:,4]
    value=np.arctan2(abs(u[:,0]*v[:,1]-u[:,1]*v[:,0]),(u*v).sum(1))
    contained(e.angle(a,b),value)


def test_shared_expression_cancels_before_uncertainty_propagation():
    e=SourceRelations(source(q=2))
    assert e.length_ratio(EYE_VECTOR)==Interval(1.,1.,True,'shared_expression_cancels')
    a=displacement(0,4)
    assert e.bilinear(a,a,'cross').lower==e.bilinear(a,a,'cross').upper==0
    assert e.asymmetry(a,-a).lower==e.asymmetry(a,-a).upper==0
    assert e.angle(a,a).upper==0
    assert e.angle(a,-a).lower==pytest.approx(np.pi)


def test_uncertain_denominator_is_not_replaced_by_observed_span():
    c=np.zeros((8,2,2));c[:4]=np.diag([.08**2,.002**2])
    s=EllipticSource(POINTS,c,np.ones(8,bool),1)
    p=POINTS.copy();p[:2,0]-=.08;p[2:4,0]+=.08
    true=np.linalg.norm(p[5]-p[6])/np.linalg.norm(EYE_VECTOR@p)
    fixed=np.linalg.norm(POINTS[5]-POINTS[6])/np.linalg.norm(EYE_VECTOR@POINTS)
    assert abs(true-fixed)>.08
    contained(SourceRelations(s).length_ratio(displacement(5,6)),[true])


def test_normalizer_collapse_withholds_ratio_instead_of_epsilon_clipping():
    e=SourceRelations(source(q=200))
    assert e.eye_span.lower==0
    assert not e.length_ratio(displacement(5,6)).available
    assert not e.signed_area_ratio(displacement(0,4),displacement(3,4)).available


def test_zero_length_arm_withholds_angle():
    p=POINTS.copy();p[4]=p[0]
    e=SourceRelations(source(points=p))
    assert not e.angle(displacement(0,4),displacement(3,4)).available


@pytest.mark.parametrize('i,j,k',[(0,3,4),(4,5,6),(3,6,7),(0,1,2)])
def test_signed_area_contains_boundary_draws_and_collinear_nominal(i,j,k):
    s=source(3);p=draws(s);u=p[:,j]-p[:,i];v=p[:,k]-p[:,i]
    values=(u[:,0]*v[:,1]-u[:,1]*v[:,0])/np.sum(np.einsum('n,knd->kd',EYE_VECTOR,p)**2,axis=1)
    contained(SourceRelations(s).signed_area_ratio(displacement(j,i),displacement(k,i)),values)


@pytest.mark.parametrize('seed',range(4))
def test_asymmetry_is_signed_and_bounded(seed):
    s=source(seed);p=draws(s)
    a,b=displacement(4,5),displacement(4,6)
    u=np.linalg.norm(p[:,4]-p[:,5],axis=1);v=np.linalg.norm(p[:,4]-p[:,6],axis=1)
    e=SourceRelations(s)
    bound=e.asymmetry(a,b);contained(bound,(u-v)/(u+v))
    swapped=e.asymmetry(b,a)
    np.testing.assert_allclose([bound.lower,bound.upper],[-swapped.upper,-swapped.lower],atol=1e-12)


def test_unobserved_point_propagates_only_to_dependent_relations():
    s=source();mask=s.supported.copy();mask[7]=False
    e=SourceRelations(replace(s,supported=mask))
    assert e.length_ratio(displacement(0,4)).available
    assert not e.length_ratio(displacement(0,7)).available
    assert not e.bilinear(displacement(0,4),displacement(0,7),'dot').available


@pytest.mark.parametrize('change',[
    {'quantile':-1},{'quantile':float('nan')},{'directions':31},{'directions':True},
    {'directions':65},{'directions':2050},{'centers':np.zeros((7,2))},
    {'supported':np.ones(8)},{'covariance':np.tile(-np.eye(2),(8,1,1))},
    {'covariance':np.tile([[1,1],[0,1]],(8,1,1))},
])
def test_source_contract_rejects_malformed_input(change):
    with pytest.raises(ObservationError):replace(source(),**change)


@pytest.mark.parametrize('value',[np.ones(8),np.ones(7),np.r_[np.nan,np.zeros(7)]])
def test_non_intrinsic_or_malformed_coefficients_rejected(value):
    with pytest.raises(ObservationError):coefficients(value)


def test_arrays_are_owned_and_truly_immutable():
    p=POINTS.copy();s=source(points=p);p[:]=20
    np.testing.assert_array_equal(s.centers,POINTS)
    with pytest.raises(ValueError):s.centers.flags.writeable=True
    with pytest.raises(ValueError):s.covariance[0,0,0]=1


def test_interval_operations_keep_unavailable_distinct_from_zero():
    bad=Interval.missing('unsupported')
    assert not divide(bad,Interval(1,2)).available
    assert not divide(Interval(1,2),Interval(0,2)).available
    assert not absolute_difference(bad,Interval(0,0)).available
    assert absolute_difference(Interval(1,2),Interval(3,4)).lower==pytest.approx(1.)
    assert absolute_difference(Interval(1,3),Interval(2,4)).lower==0
