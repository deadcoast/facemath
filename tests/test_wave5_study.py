from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, ProviderUnavailable
from facemath.observation.study import (ObserverDeclaration,freeze_study,load_study,read_json,write_json_new,local_file)
from facemath.observation.study_lab import make_fixture_study,FixtureExecutor
from facemath.observation.study_runner import (AttemptOutcome,NativeExecutor,run_session,verify_session,writer_lock)
from facemath.observation.study_io import finalize_session,verify_qualification_bundle,replay_qualification_bundle


@pytest.fixture
def inputs(tmp_path):
    return make_fixture_study(tmp_path/'inputs',development=6,calibration=12,evaluation=6)


def change_spec(path,fn):
    data=read_json(path);fn(data);path.write_text(json.dumps(data));return data


def test_freeze_observes_nothing_and_binds_all_pixels(tmp_path,inputs):
    spec,executor,records=inputs
    study=freeze_study(spec,tmp_path/'frozen')
    assert executor.calls==0 and len(study.included)==24
    assert load_study(study.root).digest==study.digest
    d=study.data;d['domain']='changed';assert study.data['domain']!='changed'
    with pytest.raises(FileExistsError):freeze_study(spec,study.root)


@pytest.mark.parametrize('edit,match',[
 (lambda d:d['cases'][0].update(observation='some.json'),'import predictions'),
 (lambda d:d['cases'][1].update(case_id=d['cases'][0]['case_id']),'duplicate case'),
 (lambda d:d['cases'][1].update(group_id=d['cases'][0]['group_id']),'maximum'),
 (lambda d:d['cases'][6].update(lineage_ids=d['cases'][0]['lineage_ids']),'leakage'),
 (lambda d:d['cases'][0].update(stratum='unregistered'),'stratum'),
 (lambda d:d.update(bias_mode='choose_best_from_test'),'bias_mode'),
 (lambda d:d.update(alpha=0),'alpha'),
 (lambda d:d['cases'][0].update(eligibility='excluded'),'exclusion'),
 (lambda d:d['cases'][0].update(references=d['cases'][1]['references']),'reference'),
 (lambda d:d.update(minimum_development_groups=20),'development'),
 (lambda d:d['cases'][0].update(image='../escape.png'),'traverse'),
])
def test_preflight_rejects_bad_inventory_before_any_inference(tmp_path,inputs,edit,match):
    spec,executor,_=inputs;change_spec(spec,edit)
    with pytest.raises((ObservationError,KeyError),match=match):freeze_study(spec,tmp_path/'frozen')
    assert executor.calls==0 and not (tmp_path/'frozen').exists()


def test_frozen_bytes_cannot_be_silently_replaced(tmp_path,inputs):
    study=freeze_study(inputs[0],tmp_path/'frozen')
    p=study.root/study.cases[0]['image'];p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ObservationError,match='inventory'):load_study(study.root)


def test_no_extra_artifact_in_lock(tmp_path,inputs):
    study=freeze_study(inputs[0],tmp_path/'frozen');(study.root/'extra.txt').write_text('oops')
    with pytest.raises(ObservationError,match='inventory'):load_study(study.root)


def test_partial_resume_is_idempotent_and_exclusions_are_not_calls(tmp_path,inputs):
    spec,executor,_=inputs;study=freeze_study(spec,tmp_path/'frozen')
    first=run_session(study.root,tmp_path/'session',executor,maximum_new_cases=3)
    assert executor.calls==3 and first['committed']==3 and not first['complete']
    second=run_session(study.root,tmp_path/'session',executor)
    calls=executor.calls
    third=run_session(study.root,tmp_path/'session',executor)
    assert second['complete'] and third['complete'] and executor.calls==calls==24
    assert third['prospectively_excluded']==1


def test_interruption_becomes_explicit_unavailable_not_a_retry(tmp_path,inputs):
    spec,executor,_=inputs;study=freeze_study(spec,tmp_path/'frozen');executor.fail_on_call=2
    with pytest.raises(KeyboardInterrupt):run_session(study.root,tmp_path/'session',executor)
    status=verify_session(tmp_path/'session');assert len(status['interrupted_uncommitted'])==1
    executor.fail_on_call=None
    end=run_session(study.root,tmp_path/'session',executor)
    assert end['outcomes']['interrupted']==1 and end['outcomes']['observed']==23
    assert executor.calls==24


def test_bad_output_binding_stops_session_and_cannot_qualify(tmp_path,inputs):
    spec,executor,_=inputs;study=freeze_study(spec,tmp_path/'frozen')
    original=executor.execute
    def wrong(*args):
        out=original(*args)
        return AttemptOutcome('observed',replace(out.observation,image_sha256='0'*64))
    executor.execute=wrong
    with pytest.raises(ObservationError,match='contract'):run_session(study.root,tmp_path/'session',executor)
    status=verify_session(tmp_path/'session');assert status['outcomes']['contract_error']==1
    assert executor.calls==1
    with pytest.raises(ObservationError,match='incomplete|contract'):finalize_session(tmp_path/'session',tmp_path/'bad-output')
    assert not (tmp_path/'bad-output').exists()


def test_startup_failure_does_not_invent_attempts(tmp_path,inputs):
    spec,executor,_=inputs;study=freeze_study(spec,tmp_path/'frozen')
    def fail(*a):raise ProviderUnavailable('missing runtime control')
    executor.preflight=fail
    with pytest.raises(ProviderUnavailable):run_session(study.root,tmp_path/'session',executor)
    status=verify_session(tmp_path/'session');assert status['committed']==0
    assert len(list((tmp_path/'session/preflight').glob('*.json')))==1


def test_receipt_corruption_prevents_resume(tmp_path,inputs):
    spec,executor,_=inputs;study=freeze_study(spec,tmp_path/'frozen')
    run_session(study.root,tmp_path/'session',executor,maximum_new_cases=1)
    p=tmp_path/'session/cases/case_00000/observation.json';p.write_text('{}')
    with pytest.raises(ObservationError,match='inventory'):run_session(study.root,tmp_path/'session',executor)
    assert executor.calls==1


def test_exclusive_writer_lock(tmp_path):
    with writer_lock(tmp_path/'lock'):
        with pytest.raises(ObservationError,match='writer'):
            with writer_lock(tmp_path/'lock'):pass
    with writer_lock(tmp_path/'lock'):pass


def test_worker_declaration_checks_native_fingerprint():
    d=ObserverDeclaration.mediapipe('1'*64,'1.0.1')
    assert ObserverDeclaration.from_dict(d.to_dict()).recipe_sha256==d.recipe_sha256
    with pytest.raises(ObservationError,match='fingerprint'):replace(d,provider_fingerprint='2'*64)


def test_native_preflight_does_not_download_missing_model(tmp_path):
    e=NativeExecutor(tmp_path/'not-here.task')
    with pytest.raises(ProviderUnavailable,match='missing'):
        e.preflight(ObserverDeclaration.mediapipe('1'*64,'1.0.1'))
    assert not (tmp_path/'not-here.task').exists()


def test_full_qualification_preserves_raw_rows_and_replays_exactly(tmp_path,inputs):
    spec,executor,_=inputs;study=freeze_study(spec,tmp_path/'frozen')
    run_session(study.root,tmp_path/'session',executor)
    output=finalize_session(tmp_path/'session',tmp_path/'qualification')
    assert verify_qualification_bundle(output)['verified']
    report=read_json(output/'qualification.json')
    assert report['raw_coordinates_changed'] is False
    assert report['estimator_selected_using_evaluation'] is False
    assert report['live_human_observer_validation'] is False
    assert report['slices'][-1]['has_calibration_representation'] is False
    before=executor.calls
    assert replay_qualification_bundle(output,tmp_path/'replay')['replay_verified']
    assert executor.calls==before


@pytest.mark.parametrize('outcome', ['observed','no_face','multiple_faces','timeout','interrupted','runtime_error','contract_error'])
def test_attempt_contract(outcome):
    if outcome=='observed':
        with pytest.raises(ObservationError):AttemptOutcome(outcome)
    else:
        with pytest.raises(ObservationError):AttemptOutcome(outcome)
        assert AttemptOutcome(outcome,detail='recorded reason').status==outcome
