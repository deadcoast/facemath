"""CLI, actual worker failure paths, coordinate policy and finite-sample tests."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from facemath.cli import execute, build_parser, load_rgb as cli_rgb
from facemath.observation.contracts import ObservationError
from facemath.observation.qualification import binomial_interval
from facemath.observation.study import (
    ObserverDeclaration, freeze_study, read_json, write_json_new, load_rgb, save_rgb,
)
from facemath.observation.study_lab import make_fixture_study
from facemath.observation.study_runner import NativeExecutor, run_session, verify_session
from facemath.observation.validation_io import _image as validation_rgb


def test_source_coordinates_do_not_follow_exif_orientation(tmp_path):
    original = np.zeros((30, 60, 3), np.uint8)
    original[:, :20, 0] = 255
    exif = Image.Exif(); exif[274] = 6  # display wants 90-degree rotation
    path = tmp_path/'raster_é_日本.jpg'
    Image.fromarray(original).save(path, exif=exif)
    decoded = load_rgb(path)
    assert decoded.shape == (30, 60, 3)
    assert np.array_equal(decoded, cli_rgb(path))
    assert np.array_equal(decoded, validation_rgb(path))
    assert decoded[:, :18, 0].mean() > 240


@pytest.mark.parametrize('bad', [0, -1, float('inf'), float('nan'), True, '90'])
def test_timeout_is_frozen_before_observation(tmp_path, bad):
    spec, executor, _ = make_fixture_study(tmp_path/'inputs', development=5, calibration=2, evaluation=2)
    data = read_json(spec); data['timeout_seconds'] = bad
    spec.write_text(json.dumps(data))
    with pytest.raises(ObservationError):
        freeze_study(spec, tmp_path/'frozen')
    assert executor.calls == 0


def test_native_timeout_cannot_change_on_resume(tmp_path):
    # Study inputs are real stored fixture pixels, but the reference kind is
    # explicitly adjusted as a CONTRACT FIXTURE only. No native inference runs.
    spec, executor, _ = make_fixture_study(tmp_path/'inputs', development=5, calibration=2, evaluation=2)
    data = read_json(spec)
    declaration = ObserverDeclaration.mediapipe('1'*64, '1.0.1')
    data['observer'] = declaration.to_dict()
    data['timeout_seconds'] = 12.0
    for case in data['cases']:
        for name in case['references']:
            path = spec.parent/name
            obj = read_json(path)
            # Recreate a valid annotation through its constructor, not tampered hash.
            from facemath.observation.reference import ReferenceAnnotation
            ref = ReferenceAnnotation.from_dict(obj)
            path.write_text(json.dumps(replace(ref, kind='external_dataset').to_dict()))
    spec.write_text(json.dumps(data))
    study = freeze_study(spec, tmp_path/'frozen')
    class BadPolicy:
        def preflight(self, declaration):
            return {'ready': True, 'kind': 'native_mediapipe', 'timeout_seconds': 99.0}
        def execute(self, *args):
            pytest.fail('must not invoke after timeout policy mismatch')
    with pytest.raises(ObservationError, match='timeout'):
        run_session(study.root, tmp_path/'session', BadPolicy())
    assert verify_session(tmp_path/'session')['committed'] == 0


def test_native_worker_actually_executes_failure_path_without_fake_landmarks(tmp_path):
    image = np.full((32, 48, 3), 140, np.uint8)
    save_rgb(tmp_path/'source.png', image)
    out = tmp_path/'attempt'; out.mkdir()
    executor = NativeExecutor(tmp_path/'does-not-exist.task', timeout_seconds=15)
    result = executor.execute(tmp_path/'source.png', ObserverDeclaration.mediapipe('1'*64, '1.0.1'), out)
    assert result.observation is None and result.status in ('contract_error', 'runtime_error')
    assert (out/'stderr.txt').exists() and (out/'stdout.txt').exists()
    process = read_json(out/'process.json')
    assert process['returncode'] != 0 and not process['timed_out']
    request = read_json(out/'request.json')
    assert set(request) == {'format', 'image', 'model', 'observer', 'engine_code_sha256', 'numeric_environment'}
    assert 'references' not in request and 'split' not in request and 'stratum' not in request
    response = read_json(out/'response.json')
    assert response['observation'] is None


def test_simulated_timeout_is_recorded_without_a_second_attempt(tmp_path, monkeypatch):
    out = tmp_path/'attempt';out.mkdir()
    calls = []
    def expired(argv, **kwargs):
        calls.append((argv, kwargs['timeout']))
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
    monkeypatch.setattr('facemath.observation.study_runner.subprocess.run', expired)
    result = NativeExecutor(tmp_path/'model.task', timeout_seconds=1).execute(
        tmp_path/'source.png', ObserverDeclaration.mediapipe('1'*64, '1.0.1'), out)
    assert result.status == 'timeout' and result.observation is None
    assert len(calls) == 1 and calls[0][1] == 1
    assert read_json(out/'process.json')['timed_out']


def test_model_declaration_creation_is_not_inference(tmp_path):
    path=tmp_path/'opaque.task';path.write_bytes(b'test bytes; not a real model')
    args=build_parser().parse_args(['observer-pin','--model',str(path),'--runtime-version','1.0.1',
        '--out',str(tmp_path/'observer.json')])
    result=execute(args)
    assert result['inference_executed'] is False and result['model_validated'] is False
    assert result['model_sha256'] == sha256(path.read_bytes()).hexdigest()
    assert ObserverDeclaration.from_dict(read_json(tmp_path/'observer.json')).runtime_version == '1.0.1'


def test_observer_declaration_can_be_an_explicit_relative_file(tmp_path):
    spec, executor, _=make_fixture_study(tmp_path/'inputs',development=5,calibration=2,evaluation=2)
    data=read_json(spec);write_json_new(spec.parent/'observer.json',data['observer'])
    data['observer']='observer.json';spec.write_text(json.dumps(data))
    study=freeze_study(spec,tmp_path/'frozen')
    assert study.observer.execution_kind=='synthetic_fixture' and executor.calls==0


@pytest.mark.parametrize('hits,total,expected', [(0,0,None),(0,1,(0,.975)),(1,1,(.025,1)),
                                              (0,10,(0,.3084971078187607)),(10,10,(.6915028921812392,1))])
def test_exact_binomial_endpoints(hits,total,expected):
    actual=binomial_interval(hits,total)
    if expected is None: assert actual is None
    else: np.testing.assert_allclose(actual,expected)


@pytest.mark.parametrize('args', [(-1,10),(11,10),(1,0),(True,1),(1.0,10),(0,2,0)])
def test_invalid_binomial_counts(args):
    with pytest.raises(ObservationError): binomial_interval(*args)


def test_prospective_cli_freeze_verify_and_status(tmp_path):
    spec, executor, _=make_fixture_study(tmp_path/'inputs',development=5,calibration=2,evaluation=2)
    def command(*args):
        p=subprocess.run([sys.executable,'-m','facemath',*map(str,args)],cwd=tmp_path,
                         capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr
        return json.loads(p.stdout)
    result=command('study-freeze',spec,'--out',tmp_path/'frozen')
    assert result['inference_executed'] is False
    assert command('verify',tmp_path/'frozen')['kind']=='frozen_study'
    run_session(tmp_path/'frozen',tmp_path/'session',executor,maximum_new_cases=2)
    assert command('study-status',tmp_path/'session')['committed']==2
    assert command('verify',tmp_path/'session')['pending']


def test_session_symlink_is_not_silently_followed(tmp_path):
    spec, executor, _=make_fixture_study(tmp_path/'inputs',development=5,calibration=2,evaluation=2)
    study=freeze_study(spec,tmp_path/'frozen')
    target=tmp_path/'elsewhere';target.mkdir()
    alias=tmp_path/'alias'
    try: alias.symlink_to(target,target_is_directory=True)
    except OSError: pytest.skip('platform cannot create a directory symlink')
    with pytest.raises(ObservationError, match='symlink'):run_session(study.root,alias,executor)
    assert executor.calls==0 and list(target.iterdir())==[]
