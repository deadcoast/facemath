import argparse
from dataclasses import replace
import json
import re

import numpy as np
import pytest

from facemath.cli import _replay, build_parser
from facemath.observation.artifacts import write_bundle, verify_bundle
from facemath.observation.bias import load_localization_calibration
from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import ObservationError
from facemath.observation.instrument import FacialInstrument
from facemath.observation.joint import JointCalibration, fit_development_scatter
from facemath.observation.joint_lab import (make_joint_study, _record, fixture_observation,
                                           run_joint_set_controls, DOMAIN, INDICES)
from facemath.observation.joint_report import render_joint_section
from facemath.observation.intrinsic_report import render_intrinsic_section
from facemath.observation.reference import mapped_prediction
from facemath.observation.study import freeze_study, read_json
from facemath.observation.study_io import finalize_session, verify_qualification_bundle
from facemath.observation.study_runner import run_session
from test_wave7_joint_model import rows
from facemath.observation.bias import fit_development_bias


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root=tmp_path_factory.mktemp('joint-study')
    path,executor=make_joint_study(root/'inputs',seed=27093,development=20,calibration=24,evaluation=12)
    frozen=freeze_study(path,root/'frozen')
    run_session(frozen.root,root/'session',executor,maximum_new_cases=3)
    first=executor.calls
    status=run_session(frozen.root,root/'session',executor)
    complete=executor.calls
    run_session(frozen.root,root/'session',executor)
    output=finalize_session(root/'session',root/'qualification')
    calibration=load_localization_calibration(read_json(output/'localization_calibration.json'))
    return root,executor,frozen,status,calibration,(first,complete)


@pytest.fixture(scope='module')
def pair(study):
    root,_,_,_,model,_=study
    a,b=_record(391001),_record(392001,expansion=8)
    oa,ob=(fixture_observation(r['image'],r['prediction'],r['samples']) for r in (a,b))
    config=InstrumentConfig(canvas_size=160,eye_span_px=50,uncertainty_draws=8,enable_flow=False,edge_samples=8)
    result=FacialInstrument(config,localization_calibration=model,domain=DOMAIN,
                           query_groups=('newA','newB'),query_lineages=(('new-source-A',),('new-source-B',))).run(a['image'],b['image'],oa,ob)
    bare=FacialInstrument(config).run(a['image'],b['image'],oa,ob)
    return result,bare,a,b,root,oa,ob


def test_joint_study_runs_existing_acquisition_and_qualification(study):
    root,ex,lock,status,c,calls=study
    assert isinstance(c,JointCalibration)
    assert calls==(3,56) and ex.calls==56 and status['prospectively_excluded']==1
    assert lock.data['joint_error']=={'shrinkage':.02,'floor_eye_spans':.001}
    assert verify_qualification_bundle(root/'qualification')['verified']
    report=read_json(root/'qualification'/'qualification.json')
    assert report['selected_error_model']['error_geometry']=='full16_shared_latent_ellipsoid'
    assert report['selected_error_model']['q_comparability']


def test_qualification_replay_recomputes_new_fit(study,tmp_path):
    root=study[0]
    answer=_replay(argparse.Namespace(bundle=str(root/'qualification'),out=str(tmp_path/'replay')))
    assert answer['replay_verified']


def test_attaching_joint_model_preserves_all_raw_geometry(pair):
    result,bare,*_=pair
    assert len(bare.raw_geometry_arrays())>=135
    for k,v in bare.raw_geometry_arrays().items():
        np.testing.assert_array_equal(v,result.arrays()[k],err_msg=k)
    assert result.intrinsic.joint is not None


def test_pair_uses_one_error_geometry_for_fixed_and_refitted_bounds(pair):
    result,_,_,_,_,oa,_=pair
    cal=result.localization.calibration
    assert isinstance(cal,JointCalibration)
    xy,rawcov,_=mapped_prediction(oa)
    assert not np.allclose(rawcov,cal.source_covariance(oa))
    assert result.intrinsic.joint.to_dict()['raw_repeatability_preserved']
    np.testing.assert_array_equal(result.state.observation_a.xy,oa.xy)


def test_joint_projection_diagonal_blocks_equal_same_source_covariance(pair):
    c=pair[0].localization.calibration;oa=pair[-2]
    joint=c.source_joint_covariance(oa).reshape(8,2,8,2)
    np.testing.assert_allclose(c.source_covariance(oa),np.stack([joint[i,:,i,:] for i in range(8)]))


@pytest.mark.parametrize('angle,scale',[(0,2),(30,1.1),(-55,.8),(180,1)])
def test_joint_observation_covariance_is_covariant_not_axis_fixed(pair,angle,scale):
    obs=pair[-2];c=pair[0].localization.calibration
    t=np.deg2rad(angle);R=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])*scale
    moved=replace(obs,xy=obs.xy@R.T+[200,200],samples=obs.samples@R.T+[200,200],image_shape=(1000,1000))
    transform=np.kron(np.eye(8),R)
    np.testing.assert_allclose(c.source_joint_covariance(moved),transform@c.source_joint_covariance(obs)@transform.T,rtol=1e-11,atol=1e-10)


def test_full16_eigenmodes_reconstruct_exported_error_matrix(pair):
    audit=pair[0].intrinsic.joint
    reconstructed=np.einsum('sik,sk,sjk->sij',audit.mode_vectors,audit.mode_eigenvalues,audit.mode_vectors)
    np.testing.assert_allclose(reconstructed,audit.covariance,rtol=1e-10,atol=1e-15)
    for s in range(2):
        for k in range(16):
            v=audit.mode_vectors[s,:,k]
            assert v[np.argmax(abs(v))]>=0


def test_display_uses_calibrated_marginals_not_tiny_observer_jitter(pair):
    p=pair[0].intrinsic
    html=render_intrinsic_section(p)
    data=json.loads(re.search(r'<script id="intrinsic-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    cov=p.reference_error_covariance
    assert not np.allclose(cov,p.source_covariance)
    for s in range(2):
        for i in range(8):
            expected=p.quantile*np.sqrt(np.linalg.eigvalsh(cov[s,i])[-1])
            assert data['display']['ellipse'][s][i][0]==pytest.approx(expected)


def test_joint_report_has_same_raw_eigensystem_and_no_remote_actions(pair):
    p=pair[0].intrinsic;html=render_joint_section(p)
    data=json.loads(re.search(r'<script id="joint-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    np.testing.assert_array_equal(data['covariance'],p.joint.covariance)
    np.testing.assert_array_equal(data['modes'],p.joint.mode_vectors)
    assert data['joint_shape']==p.shape.to_dict()
    assert data['marginal_shape']==p.joint.marginal_shape.to_dict()
    assert not any(s in html for s in ('fetch(', 'src="http', 'href="http'))
    assert 'not a separately fitted baseline' in html


def test_bare_pair_does_not_invent_joint_data(pair):
    bare=pair[1]
    assert bare.intrinsic.joint is None and render_joint_section(bare.intrinsic)==''
    assert not any(k.startswith('intrinsic_joint') for k in bare.arrays())


def test_joint_bundle_roundtrip_and_numerical_replay(pair,tmp_path):
    r,_,a,b,*_=pair
    path=write_bundle(r,tmp_path/'pair',a['image'],b['image'])
    assert verify_bundle(path)['verified']
    with np.load(path/'arrays.npz',allow_pickle=False) as z:
        for key,value in r.arrays().items():np.testing.assert_array_equal(z[key],value)
    answer=_replay(argparse.Namespace(bundle=str(path),out=str(tmp_path/'replay')))
    assert answer['replay_verified']


@pytest.mark.parametrize('change',[{'shrinkage':True,'floor_eye_spans':.001},
                                   {'shrinkage':.02,'floor_eye_spans':0},
                                   {'shrinkage':.02,'floor_eye_spans':.001,'extra':1}])
def test_bad_frozen_joint_parameters_rejected_before_acquisition(tmp_path,change):
    path,ex=make_joint_study(tmp_path/'inputs',development=5,calibration=10,evaluation=3)
    spec=read_json(path);spec['joint_error']=change;path.write_text(json.dumps(spec))
    with pytest.raises(ObservationError):freeze_study(path,tmp_path/'frozen')
    assert ex.calls==0


def test_joint_parameters_cannot_silently_apply_to_another_mode(tmp_path):
    path,ex=make_joint_study(tmp_path/'inputs',development=5,calibration=10,evaluation=3)
    spec=read_json(path);spec['bias_mode']='none';path.write_text(json.dumps(spec))
    with pytest.raises(ObservationError,match='joint_error'):freeze_study(path,tmp_path/'frozen')


def test_same_development_identifiers_with_changed_measurements_cannot_reuse_bias():
    data=rows()[:20];bias=fit_development_bias(data)
    changed=(replace(data[0],error_vectors_px=data[0].error_vectors_px+1),)+data[1:]
    with pytest.raises(ObservationError,match='bound development bias'):
        fit_development_scatter(changed,bias)


def test_joint_cli_is_integrated():
    args=build_parser().parse_args(['joint-lab','--out','runs/control'])
    assert args.command=='joint-lab' and args.seed==27091


def test_joint_set_control_saves_latent_inputs_not_only_summary(tmp_path):
    results=run_joint_set_controls(tmp_path/'sets',samples=64)
    assert results['sampled_relation_violations']==results['sampled_shape_violations']==0
    assert results['scalar_relation_checks']==64*256
    with np.load(tmp_path/'sets'/'controls.npz',allow_pickle=False) as z:
        for s,label in enumerate(('a','b')):
            actual=z['centers'][s]+(z['unit_ball_'+label]@z['factors'][s].reshape(16,16).T).reshape(64,8,2)
            np.testing.assert_array_equal(actual,z['draws_'+label])
    c=results['translation_cancellation']
    assert c['joint_shape']['upper']<1e-6
    assert c['same_set_marginal_relaxation']['upper']>.5
