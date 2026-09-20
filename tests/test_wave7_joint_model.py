from dataclasses import replace
import copy
import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.bias import fit_development_bias, load_localization_calibration, ocular_basis
from facemath.observation.joint import (DevelopmentScatter, JointCalibration, fit_development_scatter,
    fit_joint_calibration, joint_scored_rows, regularized_scatter)
from test_wave5_bias import row, BETA


def rows(seed=70):
    rng=np.random.default_rng(seed);out=[]
    for i in range(72):
        split='development' if i<20 else 'calibration' if i<52 else 'evaluation'
        common=rng.normal(0,.04,(1,2));local=rng.normal(0,.002,(8,2))
        out.append(row(i,split,bias=BETA+common+local,angle=i*3,scale=1+i*.005))
    return tuple(out)


def fitted(data=None):
    return fit_joint_calibration(rows() if data is None else data,study_sha256=object_digest({'study':7}),domain='joint-controls')


def test_full_scatter_includes_observable_cross_landmark_dependence():
    c,_=fitted();S=c.scatter.covariance
    assert S.shape==(16,16) and S[0,2]>0
    assert S[0,2]/np.sqrt(S[0,0]*S[2,2])>.8
    assert np.linalg.eigvalsh(S).min()>0


def test_group_moments_retain_within_group_error_not_only_group_mean():
    data=tuple(row(i,bias=BETA+(.03 if i%2 else -.03),group='g'+str(i//2)) for i in range(12))
    bias=fit_development_bias(data);scatter=fit_development_scatter(data,bias,shrinkage=0,floor_eye_spans=.001)
    np.testing.assert_allclose(bias.beta,BETA,atol=1e-14)
    assert scatter.covariance[0,0]==pytest.approx(.03**2+.001**2)
    assert scatter.group_case_counts==(2,)*6


def test_group_weights_do_not_reward_repeated_acquisitions():
    base=tuple(row(i,bias=BETA+.01*i) for i in range(8))
    repeated=base+tuple(replace(row(i,bias=BETA+.07,group='7'),lineage_ids=('extra'+str(i),)) for i in range(100,125))
    b1=fit_development_bias(base);b2=fit_development_bias(repeated)
    s1=fit_development_scatter(base,b1);s2=fit_development_scatter(repeated,b2)
    np.testing.assert_allclose(s1.covariance,s2.covariance,atol=1e-15)


@pytest.mark.parametrize('split',['calibration','evaluation'])
def test_scatter_fit_refuses_non_development_rows(split):
    data=rows()[:20];b=fit_development_bias(data)
    with pytest.raises(ObservationError,match='ONLY'):
        fit_development_scatter(data+(row(999,split),),b)


def test_joint_score_not_maximum_of_marginal_scores_and_preserves_raw_fields():
    c,scored=fitted();raw={r.case_id:r for r in rows()}
    assert any(abs(r.simultaneous_score-r.mahalanobis.max())>1e-3 for r in scored)
    for r in scored:
        np.testing.assert_array_equal(r.predicted_xy,raw[r.case_id].predicted_xy)
        np.testing.assert_array_equal(r.error_vectors_px,raw[r.case_id].error_vectors_px)
        assert r.nme==raw[r.case_id].nme


def test_evaluation_errors_cannot_change_fitted_center_scatter_or_quantile():
    data=rows();c1,_=fitted(data)
    changed=tuple(row(int(r.case_id),'evaluation',bias=BETA*400) if r.split=='evaluation' else r for r in data)
    c2,_=fitted(changed)
    assert c1.digest==c2.digest


def test_failure_mass_is_not_discarded_or_imputed_as_zero():
    data=tuple(replace(r, failure='prescribed_failure',observed=np.zeros(8,bool),simultaneous_score=None)
               if r.split=='calibration' else r for r in rows())
    c,scored=fitted(data)
    assert c.quantile is None
    assert all(r.simultaneous_score is None for r in scored if r.split=='calibration')


@pytest.mark.parametrize('change',[{'group_id':'0'},{'image_sha256':object_digest({'image':0})},{'lineage_ids':('source0',)}])
def test_joint_heldout_leakage_rejected(change):
    c,_=fitted()
    with pytest.raises(ObservationError,match='leakage'):
        joint_scored_rows((replace(row(90,'evaluation'),**change),),c.bias,c.scatter)


def test_serialization_is_context_bound_and_roundtrips():
    c,_=fitted();o=c.to_dict();loaded=load_localization_calibration(o)
    assert isinstance(loaded,JointCalibration) and c.digest==loaded.digest
    with pytest.raises(ObservationError):load_localization_calibration(o['residual_calibration'])
    with pytest.raises(ObservationError):load_localization_calibration(o['scatter'])


@pytest.mark.parametrize('field',['covariance','group_second_moments'])
def test_scatter_data_cannot_be_tampered_without_recomputing_model(field):
    c,_=fitted();o=c.to_dict();v=o['scatter'][field]
    if field=='covariance':v[0][0]+=.001
    else:v[0][0][0]+=.001
    with pytest.raises(ObservationError):JointCalibration.from_dict(o)


@pytest.mark.parametrize('field',['requires_bias_sha256','requires_scatter_sha256','score'])
def test_joint_fragment_context_cannot_be_swapped(field):
    c,_=fitted();o=c.to_dict();o['residual_calibration'][field]='wrong'
    with pytest.raises(ObservationError):JointCalibration.from_dict(o)


@pytest.mark.parametrize('shrink,floor',[(-.1,.001),(1.1,.001),(.1,0),(.1,-1),(True,.1),(.2,float('nan'))])
def test_invalid_regularization_rejected(shrink,floor):
    with pytest.raises(ObservationError):regularized_scatter(np.zeros((2,16,16)),shrink,floor)


def test_degenerate_training_remains_positive_definite_with_frozen_floor():
    c,_=fitted(tuple(row(i,'development' if i<10 else 'calibration' if i<30 else 'evaluation') for i in range(40)))
    np.testing.assert_allclose(c.scatter.covariance,np.eye(16)*.001**2,atol=1e-18)
    assert c.quantile is not None and c.quantile<1e-10


def test_development_failure_accounting_preserved():
    data=rows();data=(replace(data[0],failure='failed',observed=np.zeros(8,bool),simultaneous_score=None),)+data[1:]
    c,_=fitted(data)
    assert c.bias.attempted_cases==20 and c.bias.fitted_cases==19
    assert sum(c.scatter.group_case_counts)==19
    assert ('0','failed') in c.bias.rejected_cases


def test_joint_scatter_storage_immutable():
    c,_=fitted()
    for array in (c.scatter.covariance,c.scatter.group_second_moments,c.bias.beta):
        with pytest.raises(ValueError):array.setflags(write=True)

@pytest.mark.parametrize('shrink,floor',[('0.1',.001),(.1,'small'),(None,.001),(.1,1j),(.1,1e300)])
def test_non_numeric_or_overflowing_regularizer_has_explicit_contract_error(shrink,floor):
    with pytest.raises(ObservationError):regularized_scatter(np.zeros((2,16,16)),shrink,floor)
