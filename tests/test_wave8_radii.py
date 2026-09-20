import numpy as np
import pytest
from facemath.observation.error_model import ellipse_gap_bounds
from facemath.observation.contracts import ObservationError


@pytest.mark.parametrize('q',[(1,3),(0,2),(2,0),(.3,5),(1,1)])
def test_source_specific_isotropic_bounds_recover_closed_form(q):
    d=np.array([[.3,.4],[0.,0.],[.1,.2]])
    c=np.stack([np.broadcast_to(np.eye(2)*.01,(3,2,2)),np.broadcast_to(np.eye(2)*.04,(3,2,2))])
    b=ellipse_gap_bounds(d,c,np.array(q))
    radius=.1*q[0]+.2*q[1]
    np.testing.assert_allclose(b.lower_ipd,np.maximum(0,np.linalg.norm(d,axis=1)-radius),atol=1e-12)
    np.testing.assert_allclose(b.upper_ipd,np.linalg.norm(d,axis=1)+radius,atol=1e-12)


def test_using_the_smaller_single_radius_can_exclude_an_admissible_point():
    d=np.array([[.5,0.]]);c=np.array([[np.eye(2)*.01],[np.eye(2)*.04]])
    correct=ellipse_gap_bounds(d,c,[1.,3.])
    wrong=ellipse_gap_bounds(d,c,1.)
    all_max=ellipse_gap_bounds(d,c,3.)
    witness=np.linalg.norm(d[0]+np.array([.1+.6,0]))
    assert wrong.upper_ipd[0]<witness<=correct.upper_ipd[0]
    assert correct.upper_ipd[0]<all_max.upper_ipd[0]


@pytest.mark.parametrize('seed',range(8))
def test_random_anisotropic_ellipses_enclose_different_radius_draws(seed):
    rng=np.random.default_rng(880+seed);d=rng.normal(0,.2,(1,2))
    a=rng.normal(0,.04,(2,1,2,2));c=a@a.swapaxes(-1,-2)+np.eye(2)*1e-5
    q=rng.uniform(.1,4,2);b=ellipse_gap_bounds(d,c,q,directions=32)
    angles=rng.uniform(0,2*np.pi,(2,2048));unit=np.stack([np.cos(angles),np.sin(angles)],axis=-1)
    perturb=(q[:,None,None]*np.einsum('sij,skj->ski',np.linalg.cholesky(c[:,0]),unit)).sum(0)
    norms=np.linalg.norm(d+perturb,axis=1)
    assert np.all(norms>=b.lower_ipd[0]-1e-12) and np.all(norms<=b.upper_ipd[0]+1e-12)
    rev=ellipse_gap_bounds(-d,c[::-1],q[::-1],directions=32)
    np.testing.assert_allclose(rev.lower_ipd,b.lower_ipd,atol=1e-12)
    np.testing.assert_allclose(rev.upper_ipd,b.upper_ipd,atol=1e-12)


@pytest.mark.parametrize('q',[True,'2',[True,2.],[-1,1],[np.nan,1],[np.inf,1],[1],[1,2,3],[[1,2]],None])
def test_invalid_source_radius_representations_do_not_coerce_to_geometry(q):
    with pytest.raises(ObservationError):
        ellipse_gap_bounds(np.zeros((1,2)),np.broadcast_to(np.eye(2),(2,1,2,2)),q)


def test_numerically_overflowing_radius_is_not_exported_as_infinite_geometry():
    with np.errstate(over='ignore',invalid='ignore'),pytest.raises(ObservationError,match='representable'):
        ellipse_gap_bounds(np.ones((1,2)),np.broadcast_to(np.eye(2)*16,(2,1,2,2)),[1e308,1e308])


def test_equal_source_radii_are_exactly_the_legacy_scalar_computation():
    rng=np.random.default_rng(2);d=rng.normal(size=(6,2));a=rng.normal(size=(2,6,2,2));c=a@a.swapaxes(-1,-2)+np.eye(2)
    single=ellipse_gap_bounds(d,c,2.25);pair=ellipse_gap_bounds(d,c,[2.25,2.25])
    for field in ('lower_ipd','upper_ipd','combined_support'):
        np.testing.assert_array_equal(getattr(single,field),getattr(pair,field))
