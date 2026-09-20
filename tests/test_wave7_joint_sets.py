from dataclasses import replace
import numpy as np
import pytest

from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import ObservationError
from facemath.observation.intrinsic import _bound, relation_registry, bound_refitted_shape, proper_shape_distance
from facemath.observation.intrinsic_lab import CONTROL_POINTS, _exact_relation_batch
from facemath.observation.intrinsic_sets import EllipticSource, SourceRelations, EYE_VECTOR, displacement, absolute_difference


def source(seed=0, sigma=.015, common=.04, q=2., points=CONTROL_POINTS):
    rng=np.random.default_rng(seed)
    factor=rng.normal(size=(16,16))*sigma/4
    translation=np.tile(np.eye(2),(8,1))*common
    cov=factor@factor.T+translation@translation.T
    blocks=cov.reshape(8,2,8,2)
    marginal=np.stack([blocks[i,:,i,:] for i in range(8)])
    return EllipticSource(points,marginal,np.ones(8,bool),q,joint_covariance=cov)


def sample(s, n, seed):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(n,16));z/=np.linalg.norm(z,axis=1)[:,None]
    z[n//2:]*=rng.random(n-n//2)[:,None]**(1/16)
    return s.centers+s.quantile*np.einsum('kd,njd->knj',z,s.joint_factor)


@pytest.mark.parametrize('seed',range(12))
def test_all_four_relation_families_contain_whole_shape_draws(seed):
    s=source(seed);e=SourceRelations(s);specs=relation_registry(InstrumentConfig())
    values=_exact_relation_batch(sample(s,80,seed+100),specs)
    for k,spec in enumerate(specs):
        bound=_bound(e,spec)
        if bound.available:
            assert values[:,k].min()>=bound.lower-1e-10,(seed,spec.key)
            assert values[:,k].max()<=bound.upper+1e-10,(seed,spec.key)


@pytest.mark.parametrize('seed',range(8))
def test_joint_enclosure_is_never_wider_than_same_set_marginal_relaxation(seed):
    s=source(seed);j=SourceRelations(s);p=SourceRelations(replace(s,joint_covariance=None))
    specs=relation_registry(InstrumentConfig())
    for spec in specs:
        a,b=_bound(j,spec),_bound(p,spec)
        if b.available:
            assert a.available
            assert a.lower>=b.lower-1e-11,(seed,spec.key,a,b)
            assert a.upper<=b.upper+1e-11,(seed,spec.key,a,b)


def test_common_translation_is_not_independent_feature_deformation():
    s=source(sigma=0,common=.12,q=3)
    joint=SourceRelations(s);relaxed=SourceRelations(replace(s,joint_covariance=None))
    a=displacement(0,2)
    assert joint.radius(a)<1e-7
    assert relaxed.radius(a)>.7
    assert joint.eye_span.upper-joint.eye_span.lower <1e-6
    b=joint.length_ratio(displacement(5,6))
    assert b.upper-b.lower<1e-6
    specs=relation_registry(InstrumentConfig());gaps=np.zeros(len(specs))
    bound=bound_refitted_shape(np.stack([s.centers,s.centers]),(joint,joint),gaps,True)
    assert bound.available and bound.upper<1e-6
    assert proper_shape_distance(s.centers+[123,-97],s.centers)[0]<1e-12


def test_diagonal_joint_ball_is_not_a_product_of_independent_balls():
    cov=np.eye(16)*.01; blocks=np.broadcast_to(np.eye(2)*.01,(8,2,2)).copy()
    j=SourceRelations(EllipticSource(CONTROL_POINTS,blocks,np.ones(8,bool),1,joint_covariance=cov))
    p=SourceRelations(EllipticSource(CONTROL_POINTS,blocks,np.ones(8,bool),1))
    assert j.radius(displacement(0,1))==pytest.approx(np.sqrt(.02),abs=1e-12)
    assert p.radius(displacement(0,1))==pytest.approx(.2,abs=1e-12)


def test_identical_marginals_can_hide_opposite_pairwise_uncertainty():
    positive=np.zeros((16,1));positive[0]=.1;positive[2]=.1
    negative=positive.copy();negative[2]*=-1
    results=[]
    for factor in [positive,negative]:
        cov=factor@factor.T;b=cov.reshape(8,2,8,2)
        marginal=np.stack([b[i,:,i,:] for i in range(8)])
        e=SourceRelations(EllipticSource(CONTROL_POINTS,marginal,np.ones(8,bool),1,joint_covariance=cov))
        results.append((marginal,e.radius(displacement(0,1))))
    np.testing.assert_array_equal(results[0][0],results[1][0])
    assert results[0][1]<1e-8 and results[1][1]>.19


@pytest.mark.parametrize('kind',['dot','cross'])
@pytest.mark.parametrize('seed',range(5))
def test_joint_bilinear_contains_shared_ball_not_independent_apex_errors(kind,seed):
    s=source(seed,sigma=.03,common=.02);e=SourceRelations(s)
    a,b=displacement(0,4),displacement(2,4)
    bound=e.bilinear(a,b,kind);points=sample(s,200,seed+23)
    u=np.einsum('n,knd->kd',a,points);v=np.einsum('n,knd->kd',b,points)
    values=(u*v).sum(1) if kind=='dot' else u[:,0]*v[:,1]-u[:,1]*v[:,0]
    assert values.min()>=bound.lower-1e-12 and values.max()<=bound.upper+1e-12
    if kind=='cross':
        c=e.bilinear(a,a,kind);assert c.lower==c.upper==0


@pytest.mark.parametrize('seed',range(8))
def test_refitted_shape_bounds_enclose_joint_draws_and_reverse_order(seed):
    a=source(seed);points=CONTROL_POINTS.copy();points[5,0]-=.1;points[6,0]+=.1
    b=source(seed+50,points=points)
    ea,eb=SourceRelations(a),SourceRelations(b);specs=relation_registry(InstrumentConfig())
    gaps=np.array([absolute_difference(_bound(ea,s),_bound(eb,s)).lower for s in specs])
    bound=bound_refitted_shape(np.stack([a.centers,b.centers]),(ea,eb),gaps,True)
    reverse=bound_refitted_shape(np.stack([b.centers,a.centers]),(eb,ea),gaps,True)
    assert bound.lower==pytest.approx(reverse.lower,abs=1e-12)
    assert bound.upper==pytest.approx(reverse.upper,abs=1e-12)
    for pa,pb in zip(sample(a,70,seed),sample(b,70,seed+3)):
        d=proper_shape_distance(pa,pb)[0]
        assert bound.lower-1e-12<=d<=bound.upper+1e-12


@pytest.mark.parametrize('angle',[-2.,-.4,.7,2.8])
def test_joint_relations_rotate_scale_and_translate_covariantly(angle):
    s=source();R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])*2.7
    T=np.kron(np.eye(8),R);joint=T@s.joint_covariance@T.T
    marginal=np.einsum('ij,njk,lk->nil',R,s.covariance,R)
    t=EllipticSource(s.centers@R.T+[5,6],marginal,s.supported,s.quantile,joint_covariance=joint)
    ea,eb=SourceRelations(s),SourceRelations(t)
    specs=relation_registry(InstrumentConfig())
    # Angular support grids rotate relative to the ellipsoid; the outer bounds
    # need not be bitwise identical. All exact rotated values must remain inside.
    values=_exact_relation_batch(sample(s,40,22)@R.T+[5,6],specs)
    for k,sp in enumerate(specs):
        bound=_bound(eb,sp)
        if bound.available:
            assert values[:,k].min()>=bound.lower-1e-10
            assert values[:,k].max()<=bound.upper+1e-10


@pytest.mark.parametrize('bad',[np.zeros((15,15)),np.full((16,16),np.nan),-np.eye(16)])
def test_invalid_joint_matrix_rejected(bad):
    with pytest.raises(ObservationError):replace(source(),joint_covariance=bad)


def test_wrong_marginal_rejected_not_silently_replaced():
    with pytest.raises(ObservationError,match='marginal'):replace(source(),covariance=np.zeros((8,2,2)))


def test_joint_arrays_are_immutable():
    s=source()
    for a in [s.centers,s.covariance,s.joint_covariance,s.joint_factor]:
        with pytest.raises(ValueError):a.setflags(write=True)


def test_missing_point_withholds_same_inventory_not_denominator_shrinkage():
    s=source();mask=s.supported.copy();mask[7]=False;e=SourceRelations(replace(s,supported=mask))
    registry=relation_registry(InstrumentConfig())
    assert len(registry)==256
    assert not e.length_ratio(displacement(7,4)).available
    assert e.length_ratio(displacement(0,1)).available
