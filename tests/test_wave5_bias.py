from dataclasses import replace
import numpy as np
import pytest

from facemath.observation.bias import (
    BiasCalibration, DevelopmentBias, fit_development_bias, ocular_basis,
    centered_rows, fit_bias_calibration, assess_bias_pair,
)
from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.validation import LocalizationRow


P = np.array([[30,40],[40,40],[60,40],[70,40],[50,60],[40,75],[60,75],[50,100]],float)
BETA = np.array([[.01,-.01]]*4+[[.02,.03],[-.01,.02],[.01,.02],[0,.04]])


def row(i=0, split='development', bias=BETA, group=None, scale=1., angle=0., failure=False):
    t=np.deg2rad(angle);R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    p=P@R.T*scale+[50,60]
    e=bias@ocular_basis(p).T
    cov=np.broadcast_to(np.eye(2)*.25,(8,2,2)).copy()
    err=np.linalg.norm(e,axis=1)
    mahal=err/.5
    return LocalizationRow(str(i),group or str(i),split,object_digest({'image':i}),('source'+str(i),),
        p-e,p,cov,np.ones(8,bool),np.zeros(8,bool) if failure else np.ones(8,bool),e,err,mahal,
        np.ones(8,int),np.zeros(8),40.*scale,float(err.mean()/(40*scale)) if not failure else None,
        float(mahal.max()) if not failure else None,'no_face' if failure else None,
        'synthetic_fixture',object_digest({'provider':'test'}),(object_digest({'ref':i}),))


def training():
    return tuple(row(i,scale=1+i*.1,angle=i*5) for i in range(8))


def test_bias_recovered_across_scale_and_rotation_without_labels_at_prediction():
    b=fit_development_bias(training())
    np.testing.assert_allclose(b.beta,BETA,atol=1e-15)
    r=row(20,'evaluation',scale=1.5,angle=-25)
    np.testing.assert_allclose(b.vectors(r.predicted_xy),r.error_vectors_px,atol=1e-13)


def test_development_fit_rejects_heldout_data():
    with pytest.raises(ObservationError,match='ONLY'):
        fit_development_bias(training()+(row(11,'calibration'),))


def test_fit_weights_groups_not_repeated_images():
    rows=training()+tuple(row(i,bias=BETA*3,group='8') for i in range(10,35))
    b=fit_development_bias(rows)
    np.testing.assert_allclose(b.beta, BETA*11/9,atol=1e-15)


def test_rejected_development_cases_remain_in_accounting():
    b=fit_development_bias(training()+(row(30,failure=True),))
    assert b.attempted_cases==9 and b.fitted_cases==8
    assert b.rejected_cases==( ('30','no_face'), )
    assert '30' in b.development_groups


def test_minimum_groups_is_enforced():
    with pytest.raises(ObservationError,match='insufficient'):
        fit_development_bias(training()[:3])


def test_centering_preserves_observed_coordinates_and_raw_errors():
    b=fit_development_bias(training());r=row(20,'evaluation')
    c=centered_rows((r,),b)[0]
    assert c.simultaneous_score <1e-12
    np.testing.assert_array_equal(r.predicted_xy,c.predicted_xy)
    np.testing.assert_array_equal(r.error_vectors_px,c.error_vectors_px)
    assert r.nme == c.nme
    assert r.simultaneous_score>1


@pytest.mark.parametrize('change', [dict(image_sha256=object_digest({'image':0})),dict(group_id='0'),dict(lineage_ids=('source0',))])
def test_bias_rejects_development_leakage(change):
    with pytest.raises(ObservationError,match='leakage'):
        centered_rows((replace(row(20,'evaluation'),**change),),fit_development_bias(training()))


def test_incomplete_or_failed_scores_cannot_shrink_to_zero():
    r=row(20,'calibration',failure=True)
    assert centered_rows((r,),fit_development_bias(training()))[0].simultaneous_score is None


def test_bias_model_roundtrip_immutable_and_tamper_guard():
    b=fit_development_bias(training());c=DevelopmentBias.from_dict(b.to_dict())
    assert b.digest==c.digest
    with pytest.raises(ValueError):c.beta[0,0]=4
    obj=b.to_dict();obj['beta'][0][0]+=1
    with pytest.raises(ObservationError,match='mean'):
        DevelopmentBias.from_dict(obj)


def test_calibration_roundtrip_frozen_mode():
    rows=training()+tuple(row(i,'calibration',bias=BETA+(.0001*(i-10))) for i in range(10,30))
    c,scored=fit_bias_calibration(rows,study_sha256=object_digest({'study':1}),domain='fixture')
    assert c.quantile is not None
    cc=BiasCalibration.from_dict(c.to_dict());assert cc.digest==c.digest
    assert len(scored)==20


@pytest.mark.parametrize('xy',[P[:4],np.zeros((8,2)),P*np.nan])
def test_invalid_ocular_frames_fail_explicitly(xy):
    with pytest.raises(ObservationError):ocular_basis(xy)


def test_heldout_labels_do_not_change_fitted_bias():
    b=fit_development_bias(training())
    changed=tuple(row(i,'evaluation',bias=BETA*100) for i in range(50,60))
    centered_rows(changed,b)
    assert b.digest==fit_development_bias(training()).digest


def test_evaluation_labels_cannot_tune_bias_or_calibration_order_statistic():
    calibration = tuple(row(i, 'calibration', bias=BETA+.0001*i) for i in range(10,30))
    first, _ = fit_bias_calibration(training()+calibration+tuple(row(i,'evaluation') for i in range(50,60)),
        study_sha256=object_digest({'study':1}),domain='fixture')
    second, _ = fit_bias_calibration(training()+calibration+tuple(row(i,'evaluation',bias=BETA*100) for i in range(50,60)),
        study_sha256=object_digest({'study':1}),domain='fixture')
    assert first.bias.digest == second.bias.digest
    assert first.quantile == second.quantile
    assert first.digest == second.digest


def test_bias_calibration_rejects_a_different_score_recipe():
    c,_=fit_bias_calibration(training()+tuple(row(i,'calibration') for i in range(10,30)),
        study_sha256=object_digest({'study':1}),domain='fixture')
    obj=c.to_dict();obj['score']='some_other_score'
    with pytest.raises(ObservationError,match='unsupported'):
        BiasCalibration.from_dict(obj)


def test_centered_residual_fragment_cannot_masquerade_as_zero_centered():
    from facemath.observation.bias import load_localization_calibration
    c,_=fit_bias_calibration(training()+tuple(row(i,'calibration') for i in range(10,30)),
        study_sha256=object_digest({'study':1}),domain='fixture')
    obj=c.to_dict()
    with pytest.raises(ObservationError,match='unsupported'):
        load_localization_calibration(obj['residual_calibration'])
    assert load_localization_calibration(obj).digest==c.digest
    obj['residual_calibration']['requires_bias_sha256']='0'*64
    with pytest.raises(ObservationError,match='bias context'):
        BiasCalibration.from_dict(obj)
