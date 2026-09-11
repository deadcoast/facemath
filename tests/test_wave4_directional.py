import numpy as np
import pytest
from facemath.observation.error_model import ellipse_gap_bounds
from facemath.observation.contracts import ObservationError


def test_isotropic_envelope_recovers_closed_form_circle_bounds():
    d=np.array([[.3,.4],[0,0],[.01,0.]])
    cov=np.broadcast_to(np.eye(2)*.01,(2,3,2,2)).copy()
    b=ellipse_gap_bounds(d,cov,1)
    np.testing.assert_allclose(b.lower_ipd,np.maximum(0,np.linalg.norm(d,axis=1)-.2),atol=1e-12)
    np.testing.assert_allclose(b.upper_ipd,np.linalg.norm(d,axis=1)+.2,atol=1e-12)


def test_directional_uncertainty_resolves_what_enclosing_circles_cannot():
    d=np.array([[.5,0.]])
    c=np.broadcast_to(np.diag([.005**2,.10**2]),(2,1,2,2)).copy()
    bound=ellipse_gap_bounds(d,c,1)
    assert bound.lower_ipd[0]>.489999
    assert bound.upper_ipd[0]<.55
    # Naive enclosing circles produce [.30,.70]; same calibrated evidence,
    # with no re-fitting, supplies tighter geometric outer bounds.
    assert bound.lower_ipd[0]>.3 and bound.upper_ipd[0]<.7


def test_bounds_enclose_dense_ellipse_minkowski_samples():
    rng=np.random.default_rng(61)
    for _ in range(15):
        d=rng.normal(0,.2,(1,2));a=rng.normal(0,.04,(2,1,2,2));c=a@a.swapaxes(-1,-2)+np.eye(2)*1e-5
        b=ellipse_gap_bounds(d,c,2,directions=32)
        angles=rng.uniform(0,2*np.pi,(2,12000));unit=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
        perturb=2*np.einsum('aij,amj->ami',np.linalg.cholesky(c[:,0]),unit).sum(0)
        lengths=np.linalg.norm(d+perturb,axis=1)
        assert np.all(lengths>=b.lower_ipd[0]-1e-12)
        assert np.all(lengths<=b.upper_ipd[0]+1e-12)


def test_finite_angle_correction_is_necessary_and_present():
    angle=np.pi/8;d=np.array([[np.cos(angle),np.sin(angle)]])
    c=np.broadcast_to(np.eye(2)*1e-8,(2,1,2,2)).copy()
    b=ellipse_gap_bounds(d,c,1,directions=8)
    sample_max=np.max(d@b.normals.T+b.combined_support)
    assert sample_max<1.0 and b.upper_ipd[0]>=1.0002-1e-12


def test_zero_radius_contains_exact_gap():
    d=np.array([[.3,.4]])
    c=np.broadcast_to(np.eye(2),(2,1,2,2)).copy()
    b=ellipse_gap_bounds(d,c,0)
    assert b.lower_ipd[0]<=.5<=b.upper_ipd[0]
    assert b.upper_ipd[0]-b.lower_ipd[0]<1e-12


@pytest.mark.parametrize('directions',[1,4,7,4097,16.5,True])
def test_angular_protocol_is_validated(directions):
    with pytest.raises(ObservationError):ellipse_gap_bounds(np.zeros((1,2)),np.broadcast_to(np.eye(2),(2,1,2,2)),1,directions)


def test_indefinite_covariance_does_not_produce_bounds():
    c=np.broadcast_to(np.diag([1.,-1]),(2,1,2,2))
    with pytest.raises(ObservationError):ellipse_gap_bounds(np.zeros((1,2)),c,1)
