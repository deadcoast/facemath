"""Runtime compatibility is a precondition, never a relaxed replay comparator."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest
from threadpoolctl import threadpool_limits

from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.journal import code_fingerprint, numeric_environment
from facemath.observation.numerics import compare_numeric_environments, require_numeric_environment
from facemath.observation.study import freeze_study, read_json, write_json_new
from facemath.observation.study_lab import make_fixture_study
from facemath.observation.study_runner import run_session


def resign(profile):
    profile.pop('content_sha256', None)
    profile['content_sha256'] = object_digest(profile)
    return profile


def test_profile_is_equal_in_a_fresh_process_and_has_no_installation_paths():
    expected = numeric_environment()
    child = subprocess.run([sys.executable, '-c',
        'import json; from facemath.observation.journal import numeric_environment; print(json.dumps(numeric_environment()))'],
        capture_output=True, text=True, check=True)
    actual = json.loads(child.stdout)
    comparison = compare_numeric_environments(expected, actual)
    assert comparison['compatible'], comparison
    assert not comparison['exact_replay_verified']
    assert sys.prefix not in json.dumps(expected)
    assert expected['installation']['packages']['numpy']['binaries']
    assert expected['installation']['packages']['scipy']['binaries']
    assert expected['installation']['numpy_cpu']['features']


def test_returned_profiles_cannot_mutate_cached_identity():
    expected = numeric_environment()
    changed = numeric_environment()
    changed['installation']['packages']['numpy']['binaries'].clear()
    if changed['execution']['blas']:
        changed['execution']['blas'][0]['binary']['sha256'] = '0'*64
    assert numeric_environment() == expected


@pytest.mark.parametrize('kind', ['binary', 'cpu', 'legacy', 'unsealed'])
def test_same_versions_do_not_hide_a_different_or_missing_runtime(kind):
    actual = numeric_environment()
    expected = deepcopy(actual)
    if kind == 'binary':
        binaries = expected['installation']['packages']['numpy']['binaries']
        binaries[next(iter(binaries))]['sha256'] = '0'*64
        resign(expected)
    elif kind == 'cpu':
        expected['execution']['opencv_cpu'] += ' other dispatch'
        resign(expected)
    elif kind == 'legacy':
        expected = {k: expected[k] for k in ('numpy', 'scipy', 'opencv')}
    else:
        expected['content_sha256'] = '0'*64
    assert all(actual[k] == expected[k] for k in ('numpy', 'scipy', 'opencv'))
    assert not compare_numeric_environments(expected, actual)['compatible']
    with pytest.raises(ObservationError, match='requires matching numerical runtime'):
        require_numeric_environment(expected, 'test replay')


def test_actual_thread_pool_limit_is_sampled_each_time():
    with threadpool_limits(limits=1, user_api='blas'):
        one = numeric_environment()
        if not one['execution']['blas']:
            pytest.skip('this runtime exposes no controllable BLAS thread pool')
        with threadpool_limits(limits=2, user_api='blas'):
            two = numeric_environment()
        assert numeric_environment() == one
    assert not compare_numeric_environments(one, two)['compatible']
    assert {p['num_threads'] for p in one['execution']['blas']} == {1}
    assert {p['num_threads'] for p in two['execution']['blas']} == {2}


def test_dispatch_environment_change_is_not_lost_in_a_cached_profile(monkeypatch):
    expected = numeric_environment()
    monkeypatch.setenv('OPENBLAS_CORETYPE', 'different-future-dispatch')
    difference = compare_numeric_environments(expected)
    assert not difference['compatible']
    assert any(d['path'] == '/execution/dispatch_environment/OPENBLAS_CORETYPE' for d in difference['differences'])


def test_resume_and_qualification_refuse_before_more_observer_calls(tmp_path, monkeypatch):
    from facemath.observation.study_io import finalize_session
    spec, executor, _ = make_fixture_study(tmp_path/'inputs', development=5, calibration=2, evaluation=2)
    frozen = freeze_study(spec, tmp_path/'frozen')
    run_session(frozen.root, tmp_path/'session', executor)
    calls = executor.calls
    monkeypatch.setenv('MKL_CBWR', 'changed-numerical-contract')
    with pytest.raises(ObservationError, match='resume requires matching numerical runtime'):
        run_session(frozen.root, tmp_path/'session', executor)
    with pytest.raises(ObservationError, match='qualification requires matching numerical runtime'):
        finalize_session(tmp_path/'session', tmp_path/'qualification')
    assert executor.calls == calls
    assert not (tmp_path/'qualification').exists()
    assert len(list(tmp_path.glob('qualification.failure-*'))) == 1


def test_replay_refuses_before_destination_or_recomputation(tmp_path, monkeypatch):
    from facemath.observation import study_io
    source = tmp_path/'saved'; source.mkdir()
    write_json_new(source/'environment.json', {'engine_code_sha256': code_fingerprint(),
        'numeric_environment': numeric_environment()})
    monkeypatch.setattr(study_io, 'verify_qualification_bundle', lambda root: None)
    monkeypatch.setattr(study_io, 'finalize_session', lambda *args: pytest.fail('preflight must precede recomputation'))
    monkeypatch.setenv('MKL_CBWR', 'different-numerical-contract')
    with pytest.raises(ObservationError, match='qualification replay requires matching numerical runtime'):
        study_io.replay_qualification_bundle(source, tmp_path/'replay')
    assert not (tmp_path/'replay').exists()


def test_native_worker_refuses_before_image_decode_or_model_load(tmp_path, monkeypatch):
    from facemath.observation import worker
    request = {'format': 'facemath.worker_request.v5', 'engine_code_sha256': code_fingerprint(),
               'numeric_environment': numeric_environment()}
    write_json_new(tmp_path/'request.json', request)
    monkeypatch.setenv('MKL_CBWR', 'different-numerical-contract')
    monkeypatch.setattr(worker, 'load_rgb', lambda *args: pytest.fail('must not decode'))
    monkeypatch.setattr(worker, 'MediaPipeProvider', lambda *args: pytest.fail('must not load a model'))
    assert worker.main([str(tmp_path/'request.json'), str(tmp_path/'response.json')]) == 2
    response = read_json(tmp_path/'response.json')
    assert response['status'] == 'contract_error' and response['observation'] is None
    assert 'matching numerical runtime' in response['detail']


def test_cli_returns_comparison_json_and_exit_status_without_overwriting(tmp_path):
    original = tmp_path/'runtime.json'
    command = [sys.executable, '-m', 'facemath', 'runtime-profile']
    result = subprocess.run(command + ['--out', str(original)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    before = original.read_bytes()
    same = subprocess.run(command + ['--compare', str(original)], capture_output=True, text=True)
    assert same.returncode == 0, same.stderr
    assert json.loads(same.stdout)['comparison']['compatible']
    env = dict(os.environ, MKL_CBWR='different-numerical-contract')
    different = subprocess.run(command + ['--compare', str(original), '--out', str(tmp_path/'comparison.json')],
                               env=env, capture_output=True, text=True)
    assert different.returncode == 2
    report = json.loads(different.stdout)
    assert not report['comparison']['compatible'] and not report['inference_executed']
    assert original.read_bytes() == before
    overwrite = subprocess.run(command + ['--out', str(original)], capture_output=True, text=True)
    assert overwrite.returncode == 2 and original.read_bytes() == before
