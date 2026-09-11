import json
from dataclasses import replace
import numpy as np
import pytest

from facemath.observation.contracts import ObservationError
from facemath.observation.lab import phantom_points, render_phantom, phantom_observation
from facemath.observation.reference import (
    ReferenceAnnotation, REFERENCE_NAMES, REFERENCE_PROTOCOL_SHA256, consensus,
    mapped_prediction, observer_recipe, import_ibug68, IBUG68_INDICES,
)
from facemath.observation.validation import ValidationCase, evaluate_case, evaluate_study, summarize, validate_study


def case(index=0, split='calibration', delta=(2., -3.), failure=None, group=None, kind='synthetic_fixture'):
    truth = phantom_points()
    image = render_phantom(truth, seed=index+500)
    obs = phantom_observation(image, truth)
    xy, _, _ = mapped_prediction(obs)
    ref = ReferenceAnnotation(obs.image_sha256, obs.image_shape, xy, np.ones(8, bool), 'truth-renderer',
                              kind, 'known schematic render coordinates; not human annotations')
    prediction = replace(obs, xy=truth+delta, samples=(truth+delta)[None])
    return ValidationCase(f'case-{index}', group or f'group-{index}', split, (ref,), None if failure else prediction, failure)


def test_reference_roundtrip_and_immutable_ownership():
    r = case().references[0]
    d = r.to_dict()
    clone = ReferenceAnnotation.from_dict(d)
    assert clone.digest == r.digest
    d['xy'][0][0] += 100
    assert clone.digest == r.digest
    with pytest.raises(ValueError):
        clone.xy[0,0] = 1


@pytest.mark.parametrize('change', [
    {'image_sha256':'a'}, {'xy':np.zeros((7,2))}, {'visible':np.ones(8)},
    {'kind':'prediction'}, {'rater_id':''}, {'source':''},
    {'prediction_exposure':'invalid'}, {'metadata_json':'[]'},
    {'image_shape':(0,50)}, {'xy':np.full((8,2), np.nan)},
])
def test_reference_invalid_inputs_are_not_coerced(change):
    with pytest.raises((ObservationError, ValueError)):
        replace(case().references[0], **change)


def test_blind_reference_cannot_declare_shown_predictions():
    with pytest.raises(ObservationError, match='blind'):
        replace(case().references[0], kind='human_blind', prediction_exposure='shown')


def test_visible_off_image_is_rejected_and_hidden_is_explicit_zero():
    r=case().references[0]; xy=r.xy.copy(); xy[-1]=[-10,20]
    with pytest.raises(ObservationError, match='outside'):
        replace(r,xy=xy)
    mask=r.visible.copy();mask[-1]=False
    with pytest.raises(ObservationError, match='sentinels'):
        replace(r,xy=xy,visible=mask)
    xy[-1]=0
    missing=replace(r,xy=xy,visible=mask)
    assert not consensus((missing,)).visible[-1]


def test_consensus_preserves_disagreement_not_detector_repeatability():
    r=case().references[0]
    second=replace(r,rater_id='second-render-test',xy=r.xy+[2,0])
    c=consensus((r,second),minimum_raters=2)
    np.testing.assert_allclose(c.xy,r.xy+[1,0]);np.testing.assert_allclose(c.disagreement_rms_px,1)
    assert np.all(c.rater_count==2)


def test_duplicate_rater_and_cross_source_rejected():
    r=case().references[0]
    with pytest.raises(ObservationError,match='rater'):
        consensus((r,r))
    with pytest.raises(ObservationError,match='different source'):
        consensus((r,replace(case(1).references[0],rater_id='second')))


def test_detector_error_not_aligned_away():
    c=case(delta=(3.,4.))
    row=evaluate_case(c)
    np.testing.assert_allclose(row.error_norm_px,5)
    np.testing.assert_allclose(row.error_vectors_px,np.tile([3,4],(8,1)))
    assert row.nme==pytest.approx(5/48)
    assert row.simultaneous_score==pytest.approx(50)


def test_annotation_sources_cannot_be_substituted():
    with pytest.raises(ObservationError,match='binding'):
        replace(case(),observation=case(2).observation)


def test_failed_inference_stays_in_denominators():
    good=case(0,delta=(0,0));bad=case(1,failure='scripted no face; not ML')
    report,rows=evaluate_study((good,bad))
    s=report['splits']['calibration']
    assert s['attempted_cases']==2 and s['complete_cases']==1
    assert s['unconditional_auc']==.5 and s['pck_including_failures']==.5
    assert s['successful_case_mean_nme']==0
    assert rows[1].nme is None and not rows[1].observed.any()


def test_partial_prediction_is_failed_case_not_zero_error():
    c=case(delta=(0,0));p=c.observation.xy.copy();p[18]=[-20,0]
    row=evaluate_case(replace(c,observation=replace(c.observation,xy=p,samples=p[None])))
    assert row.failure=='unobservable_predicted_reference_points'
    assert row.nme is None and row.error_norm_px[-1]==0 and not row.observed[-1]


@pytest.mark.parametrize('axis',['group','lineage','duplicate_image','recipe'])
def test_study_rejects_leakage_and_mixed_recipes(axis):
    a,b=case(0),case(1,split='evaluation')
    if axis=='group':b=replace(b,group_id=a.group_id)
    if axis=='lineage':a=replace(a,lineage_ids=('capture01',));b=replace(b,lineage_ids=('capture01',))
    if axis=='duplicate_image':b=replace(a,case_id='different',split='evaluation')
    if axis=='recipe':b=replace(b,observation=replace(b.observation,noise_floor_px=2))
    with pytest.raises(ObservationError):validate_study((a,b))


def test_recipe_binds_attempted_protocol_not_success_count():
    obs=case().observation
    attempted=replace(obs,attempted_samples=7)
    realized=replace(attempted,samples=np.repeat(obs.xy[None],3,axis=0),sample_labels=('a','b','c'))
    assert observer_recipe(attempted)==observer_recipe(realized)
    assert observer_recipe(obs)!=observer_recipe(attempted)


def test_missing_reference_eye_does_not_use_predicted_eye_as_denominator():
    c=case();r=c.references[0];v=r.visible.copy();v[0]=False;p=r.xy.copy();p[0]=0
    with pytest.raises(ObservationError,match='eye'):
        replace(c,references=(replace(r,visible=v,xy=p),))


def test_ibug_import_origin_and_only_reviewed_subset():
    points=np.tile([101.,101.],(68,1))+np.arange(68)[:,None]*.1
    text='version: 1\nn_points: 68\n{\n'+'\n'.join(f'{x} {y}' for x,y in points)+'\n}'
    ref=import_ibug68(text,image_sha256='a'*64,image_shape=(256,256),coordinate_origin=1,source='test-only fixture')
    np.testing.assert_allclose(ref.xy,points[list(IBUG68_INDICES)]-1)
    assert ref.prediction_exposure=='unknown'
    assert json.loads(ref.metadata_json)['mapping_requires_review']


@pytest.mark.parametrize('text', ['version: 1\nn_points: 67\n{\n0 0\n}', '0 0', 'version: 1\nn_points: 68\n{ nan 0 }'])
def test_malformed_pts_does_not_invent_missing_landmarks(text):
    with pytest.raises(ObservationError):
        import_ibug68(text,image_sha256='a'*64,image_shape=(256,256),coordinate_origin=0,source='test')
