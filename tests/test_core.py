import numpy as np
from facemath.geometry import generalized_procrustes, similarity_transform, apply_homography, epsilon_isometry_distortion
from facemath.deformation import FlowField, strain_energy
from facemath.temporal import slow_feature_analysis, robust_pca_pcp
from facemath.coverage import SECTION_TO_IMPLEMENTATION


def test_similarity_transform_recovers_mapping():
    a=np.array([[0,0],[1,0],[0,1],[1,1]],float)
    th=.2; s=2.3
    r=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    b=(s*(r@a.T)).T+np.array([4,-2])
    m,_,rmse=similarity_transform(a,b)
    assert rmse < 1e-10
    assert np.allclose(apply_homography(a,m),b)


def test_procrustes_invariant_to_similarity():
    a=np.array([[0,0],[2,0],[0,1],[1,3]],float)
    b=a*7+np.array([10,-9])
    assert generalized_procrustes(a,b).distance < 1e-10


def test_epsilon_isometry_zero_for_equal_shapes():
    p=np.array([[0,0],[1,0],[0,1]],float)
    eps,_=epsilon_isometry_distortion(p,p)
    assert eps==0


def test_uniform_translation_flow_has_zero_strain():
    uv=np.ones((16,16,2))*np.array([3.,-2.])
    assert strain_energy(FlowField(uv)) < 1e-12


def test_sfa_shapes():
    t=np.linspace(0,10,100)
    x=np.c_[np.sin(t),np.sin(t*.1),np.random.default_rng(0).normal(size=100)]
    z,vals=slow_feature_analysis(x,2)
    assert z.shape==(100,2)
    assert len(vals)==2


def test_rpca_reconstructs():
    rng=np.random.default_rng(1)
    u=rng.normal(size=(20,2)); v=rng.normal(size=(2,12)); l=u@v
    s=np.zeros_like(l); s[0,0]=8
    ll,ss=robust_pca_pcp(l+s,max_iter=200)
    assert np.linalg.norm((ll+ss)-(l+s))/np.linalg.norm(l+s) < 1e-5


def test_full_conversation_math_coverage():
    assert set(SECTION_TO_IMPLEMENTATION)==set(range(20,181))
