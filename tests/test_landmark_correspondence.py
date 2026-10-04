"""Correspondence changes measurements, never stored native vertices."""
from dataclasses import replace

import numpy as np
import pytest

from facemath.observation.contracts import Observation, ObservationError, array_digest
from facemath.observation.correspondence import MP_REFERENCE8_V2
from facemath.observation.anatomy import anatomy_for
from facemath.observation.canonical import canonicalize, InstrumentConfig
from facemath.observation.reference import mapped_prediction, observer_recipe, ReferenceAnnotation
from facemath.observation.study import ObserverDeclaration
from facemath.observation.validation import ValidationCase, evaluate_case, validate_study
from facemath.observation.error_model import fit_localization_calibration


def observation(index=0, correspondence=None):
    image = np.full((96, 96, 3), index, np.uint8)
    xy = np.full((478, 2), 48.)
    xy[[33,133,362,263,1,4,61,291,152]] = [
        [20,30],[30,30],[60,30],[70,30],[45,55],[45,48],[30,65],[60,65],[45,82]]
    samples = np.stack([xy, xy + np.arange(478)[:, None] * .001])
    decl = ObserverDeclaration.mediapipe('f'*64, '1.0.1', augment=False,
                                        correspondence=correspondence)
    obs = Observation(array_digest(image), image.shape[:2], 'mediapipe478.v1',
        tuple(f'mp_{i:03}' for i in range(478)), xy, samples, 'mediapipe',
        decl.provider_fingerprint, attempted_samples=2, sample_labels=('one','two'),
        correspondence=correspondence)
    return image, obs


def case(index=0, correspondence=None):
    _, obs = observation(index, correspondence)
    ref = ReferenceAnnotation(obs.image_sha256, obs.image_shape,
        mapped_prediction(obs)[0] + .1, np.ones(8, bool), 'fixture',
        'external_dataset', 'generated contract fixture; not anatomical evidence')
    return ValidationCase(str(index), str(index), 'calibration', (ref,), obs)


def test_new_mapping_selects_vertex_covariance_and_support_without_mutating_mesh():
    _, old = observation()
    new = replace(old, correspondence=MP_REFERENCE8_V2)
    old_p, old_c, old_m = mapped_prediction(old)
    new_p, new_c, new_m = mapped_prediction(new)
    np.testing.assert_array_equal(new_p[4], old.xy[4])
    np.testing.assert_array_equal(old_p[4], old.xy[1])
    np.testing.assert_array_equal(new_c[4], old.covariance[4])
    for old_array, new_array in ((old.xy,new.xy),(old.samples,new.samples),
                                  (old.covariance,new.covariance)):
        np.testing.assert_array_equal(old_array, new_array)
    np.testing.assert_array_equal(old_p[[0,1,2,3,5,6,7]], new_p[[0,1,2,3,5,6,7]])
    xy = old.xy.copy(); xy[4] = [-1,-1]
    assert not mapped_prediction(replace(new, xy=xy))[2][4]
    assert mapped_prediction(replace(old, xy=xy))[2][4]
    assert anatomy_for(new).anchors['nose_tip'] == (4,)
    assert 4 in anatomy_for(new).region_indices('nose')
    assert anatomy_for(old).anchors['nose_tip'] == (1,)
    assert anatomy_for(old).digest != anatomy_for(new).digest
    assert observer_recipe(old) != observer_recipe(new)


def test_legacy_bytes_and_recipe_remain_stable_and_new_format_is_explicit():
    _, old = observation()
    legacy = old.to_dict()
    assert legacy['format'] == 'facemath.observation.v3'
    assert 'correspondence' not in legacy
    assert Observation.from_dict(legacy).to_dict() == legacy
    new = replace(old, correspondence=MP_REFERENCE8_V2)
    data = new.to_dict()
    assert data['format'] == 'facemath.observation.v17'
    assert data['correspondence'] == MP_REFERENCE8_V2
    assert Observation.from_dict(data).digest == new.digest
    for broken in (dict(data, format='facemath.observation.v3'),
                   dict(data, correspondence=None), dict(data, correspondence='unknown')):
        with pytest.raises(ObservationError, match='correspondence'):
            Observation.from_dict(broken)
    del data['correspondence']
    with pytest.raises(ObservationError, match='correspondence'):
        Observation.from_dict(data)


def test_declaration_and_observation_share_recipe_and_legacy_declaration_roundtrips():
    for identity in (None, MP_REFERENCE8_V2):
        _, obs = observation(correspondence=identity)
        decl = ObserverDeclaration.mediapipe('f'*64,'1.0.1',correspondence=identity)
        obs = replace(obs, attempted_samples=7)
        assert observer_recipe(obs) == decl.recipe_sha256
        assert ObserverDeclaration.from_dict(decl.to_dict()) == decl
        assert ('correspondence' in decl.to_dict()) == (identity is not None)
    assert ObserverDeclaration.mediapipe('f'*64,'1.0.1').correspondence == MP_REFERENCE8_V2
    with pytest.raises(ObservationError, match='correspondence'):
        ObserverDeclaration.from_dict(dict(decl.to_dict(), correspondence=None))


def test_calibration_refuses_mapping_change_in_both_directions():
    for fitted, query in ((None,MP_REFERENCE8_V2),(MP_REFERENCE8_V2,None)):
        rows = tuple(evaluate_case(case(i, fitted)) for i in range(10))
        model = fit_localization_calibration(rows, study_sha256='a'*64, alpha=.1, domain='test')
        _, obs = observation(30, fitted)
        model.require_query(obs, 'query', 'test')
        with pytest.raises(ObservationError, match='recipe mismatch'):
            model.require_query(replace(obs, correspondence=query), 'query', 'test')


def test_mixed_study_and_pair_correspondences_refuse_before_geometry():
    with pytest.raises(ObservationError, match='mixed observer recipes'):
        validate_study((case(0), case(1,MP_REFERENCE8_V2)))
    image, old = observation()
    new = replace(old, correspondence=MP_REFERENCE8_V2)
    with pytest.raises(ObservationError, match='correspondence'):
        canonicalize(image,image,old,new,anatomy_for(old),InstrumentConfig())
    with pytest.raises(ObservationError, match='correspondence'):
        canonicalize(image,image,new,new,anatomy_for(old),InstrumentConfig())


def test_correspondence_cannot_attach_to_unrelated_schema():
    _, old = observation()
    with pytest.raises(ObservationError, match='correspondence'):
        replace(old, schema='semantic21.v1', correspondence=MP_REFERENCE8_V2)


def test_historical_anchor_table_is_immutable():
    from facemath.observation.anatomy import MP_ANCHORS
    with pytest.raises(TypeError):
        MP_ANCHORS['nose_tip'] = 4


@pytest.mark.parametrize('identity', [None, MP_REFERENCE8_V2])
def test_worker_obeys_frozen_correspondence_instead_of_provider_default(tmp_path, monkeypatch, identity):
    from facemath.observation import worker
    from facemath.observation.journal import code_fingerprint, numeric_environment
    from facemath.observation.study import write_json_new, read_json, save_rgb
    import json
    image, obs = observation(correspondence=identity)
    decl = ObserverDeclaration.mediapipe('f'*64,'1.0.1',correspondence=identity)
    obs = replace(obs, attempted_samples=7,
        metadata_json=json.dumps(dict(model_sha256='f'*64,runtime_version='1.0.1')))
    seen = {}
    class Provider:
        version = '1.0.1'
        def __init__(self, *args, **kwargs): seen.update(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def observe(self, image): return obs
    monkeypatch.setattr(worker, 'MediaPipeProvider', Provider)
    save_rgb(tmp_path/'image.png', image)
    write_json_new(tmp_path/'request.json', dict(format='facemath.worker_request.v5',
        engine_code_sha256=code_fingerprint(), numeric_environment=numeric_environment(),
        observer=decl.to_dict(), image=str(tmp_path/'image.png'), model='fixture'))
    assert worker.main([str(tmp_path/'request.json'),str(tmp_path/'response.json')]) == 0
    assert seen['correspondence'] == identity
    assert Observation.from_dict(read_json(tmp_path/'response.json')['observation']).digest == obs.digest


def test_joint_fit_and_held_out_scoring_cannot_cross_correspondence():
    from facemath.observation.joint import fit_joint_calibration, joint_scored_rows
    for fitted, target in ((None, MP_REFERENCE8_V2), (MP_REFERENCE8_V2, None)):
        cases = tuple(replace(case(i, fitted), split='development' if i < 5 else 'calibration')
                      for i in range(15))
        rows = tuple(evaluate_case(c) for c in cases)
        model, _ = fit_joint_calibration(rows, study_sha256='a'*64, domain='test', alpha=.1,
                                      minimum_development_groups=2)
        held_out = evaluate_case(replace(case(40,target),split='evaluation'))
        with pytest.raises(ObservationError, match='recipe'):
            joint_scored_rows((held_out,),model.bias,model.scatter)
