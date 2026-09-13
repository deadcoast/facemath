import numpy as np
import pytest

from facemath.observation.field import affine_triangle_field
from facemath.observation.contracts import ObservationError


def points():
    x,y=np.meshgrid([25.,45.,65.],[25.,45.,65.])
    return np.c_[x.ravel(),y.ravel()]


@pytest.mark.parametrize('theta', [0.,.1,.3,.7,1.2])
def test_finite_rotation_has_zero_green_strain(theta):
    a=points()
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    b=(a-45)@rotation.T+45
    field=affine_triangle_field(a,b,(96,96),np.ones((96,96),bool),np.ones((96,96),bool))
    np.testing.assert_allclose(field.jacobians, np.broadcast_to(rotation,field.jacobians.shape), atol=1e-12)
    np.testing.assert_allclose(field.green_strain,0,atol=1e-12)
    np.testing.assert_allclose(field.principal_log_stretch,0,atol=1e-12)


@pytest.mark.parametrize('matrix', [np.diag([1.1,.9]),np.array([[1,.2],[0,1]]),np.diag([.8,1.2]),np.eye(2)])
def test_triangle_jacobians_match_analytic_affine_field(matrix):
    a=points();b=(a-45)@matrix.T+45+[2,-1]
    field=affine_triangle_field(a,b,(96,96),np.ones((96,96),bool),np.ones((96,96),bool))
    np.testing.assert_allclose(field.jacobians,np.broadcast_to(matrix,field.jacobians.shape),atol=1e-12)
    expected=.5*(matrix.T@matrix-np.eye(2))
    np.testing.assert_allclose(field.green_strain,np.broadcast_to(expected,field.green_strain.shape),atol=1e-12)
    y,x=np.where(field.valid)
    ground=(np.c_[x,y]-45)@matrix.T+45+[2,-1]-np.c_[x,y]
    np.testing.assert_allclose(field.uv[y,x],ground,atol=1e-12)


def test_folded_triangles_are_retained_and_flagged_not_counted_as_agreement():
    a=points();b=a.copy();b[:,0]=90-a[:,0]
    field=affine_triangle_field(a,b,(96,96),np.ones((96,96),bool),np.ones((96,96),bool))
    assert field.folded_fraction==1
    assert not field.valid.any()
    assert not field.valid_triangles.any()
    assert np.all(field.determinant<0)


def test_outside_hull_has_no_anatomical_support():
    a=points()
    field=affine_triangle_field(a,a,(96,96),np.ones((96,96),bool),np.ones((96,96),bool))
    assert not field.valid[:20].any()
    assert np.all(field.simplex_index[:20]<0)
    assert field.valid[40,40]


def test_missing_raster_support_is_never_reflected_back_into_evidence():
    a=points(); source=np.ones((96,96),bool); target=source.copy();target[:,:48]=False
    field=affine_triangle_field(a,a,(96,96),source,target)
    assert not field.valid[:,:48].any()
    assert field.valid[:,48:].any()


@pytest.mark.parametrize('bad', ['duplicate','collinear','nonfinite'])
def test_degenerate_mesh_inputs_fail(bad):
    a=points()
    if bad=='duplicate':a[1]=a[0]
    elif bad=='collinear':a[:,1]=0
    else:a[0,0]=np.nan
    with pytest.raises(ObservationError):
        affine_triangle_field(a,a,(96,96),np.ones((96,96),bool),np.ones((96,96),bool))
