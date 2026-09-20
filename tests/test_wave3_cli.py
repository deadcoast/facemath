"""End-to-end CLI contracts; detector inference remains a separate live test."""
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from facemath.cli import build_parser, execute, main
from facemath.observation import InstrumentConfig, ObservationError
from facemath.observation.artifacts import verify_bundle
from facemath.observation.canonical import balanced_similarity
from facemath.observation.cohort import Cohort, CohortRow
from facemath.observation.contracts import object_digest
from facemath.observation.lab import phantom_points, render_phantom
from facemath.observation.journal import numeric_environment


def command(*argv):
    return execute(build_parser().parse_args([str(a) for a in argv]))


@pytest.fixture
def demo(tmp_path):
    destination = tmp_path / 'demo'
    command('demo', '--out', destination, '--size', 96, '--draws', 4, '--no-flow')
    return destination


def test_module_entrypoint_reports_version_from_another_directory(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'facemath', '--version'], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == 'facemath 0.8.0'


def test_demo_verify_replay_roundtrip(demo, tmp_path):
    assert command('verify', demo)['verified']
    replay = command('replay', demo, '--out', tmp_path / 'replayed')
    assert replay['replay_verified']
    assert verify_bundle(tmp_path / 'replayed')['verified']
    first = json.loads((demo / 'evidence.json').read_text())
    second = json.loads((tmp_path / 'replayed' / 'evidence.json').read_text())
    assert first['measurement_profile'] == second['measurement_profile']
    assert first['recipe_sha256'] == second['recipe_sha256']


def test_cli_tolerance_reaches_shared_recipe(tmp_path):
    out = tmp_path / 'custom'
    command('demo', '--out', out, '--size', 96, '--draws', 4, '--no-flow', '--tolerance', '.12')
    evidence = json.loads((out / 'evidence.json').read_text())
    assert evidence['configuration']['tolerance_ipd'] == .12
    m = next(x for x in evidence['measurement_profile'] if x['name'] == 'geometry.all.within_tolerance')
    assert '0.12' in m['interpretation']


def test_annotation_import_then_analyze_is_image_bound(tmp_path):
    a = phantom_points()
    b = a.copy()
    b[19, 0] -= 5
    paths = []
    observations = []
    for name, points in [('a', a), ('b', b)]:
        image_path = tmp_path / f'{name}.png'
        cv2.imwrite(str(image_path), cv2.cvtColor(render_phantom(points), cv2.COLOR_RGB2BGR))
        annotation = tmp_path / f'{name}_annotation.json'
        annotation.write_text(json.dumps({'xy': points.tolist()}))
        observation = tmp_path / f'{name}_observation.json'
        result = command('import-annotation', image_path, annotation, '--annotation-id', f'test-{name}',
                         '--sigma', .25, '--out', observation)
        assert result['provider'] == 'explicit_manual_annotation'
        paths.append(image_path)
        observations.append(observation)
    result = command('analyze', *paths, '--observation-a', observations[0], '--observation-b', observations[1],
                     '--out', tmp_path / 'analyzed', '--size', 96, '--draws', 4, '--no-flow')
    assert result['verified']


def test_missing_model_produces_error_without_fake_landmarks(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(['observe', 'nonexistent.png', '--model', str(tmp_path / 'missing.task'), '--out', str(tmp_path / 'out.json')])
    assert error.value.code == 2
    assert 'model not found' in capsys.readouterr().err
    assert not (tmp_path / 'out.json').exists()


def test_existing_run_not_overwritten(demo, capsys):
    original = (demo / 'manifest.json').read_bytes()
    with pytest.raises(SystemExit) as error:
        main(['demo', '--out', str(demo), '--size', '96', '--draws', '4', '--no-flow'])
    assert error.value.code == 2
    assert original == (demo / 'manifest.json').read_bytes()


def test_capabilities_do_not_conflate_coverage_with_implementation(tmp_path):
    result = command('capabilities', '--out', tmp_path / 'capabilities.json')
    text = json.dumps(result)
    assert 'not_fully_audited' in text
    assert 'live_model_not_executed' in text
    assert 'requires_external_telemetry' in text


def test_cli_calibration_reads_verified_metric_and_same_recipe(demo, tmp_path):
    ev = json.loads((demo / 'evidence.json').read_text())
    metric = 'geometry.all.rms'
    current = next(m['value'] for m in ev['measurement_profile'] if m['name'] == metric)
    rows = [CohortRow(f'pair{i}', object_digest(f'left{i}'), object_digest(f'right{i}'),
                      {metric: current + .01 * i}) for i in range(1, 5)]
    cohort = Cohort('synthetic-fixture', ev['recipe_sha256'], rows, kind='synthetic')
    cohort_path = tmp_path / 'cohort.json'
    cohort_path.write_text(json.dumps(cohort.to_dict()))
    result = command('calibrate', demo, '--cohort', cohort_path, '--out', tmp_path / 'rank.json')
    assert result['percentile'] == 100
    assert result['cohort_kind'] == 'synthetic'


def test_numeric_runtime_is_exposed_in_bundle_environment(demo):
    env = json.loads((demo / 'environment.json').read_text())
    assert env['numeric_environment'] == numeric_environment()


@pytest.mark.parametrize('weights', [np.zeros(3), np.array([-1., 1., 1.]), np.array([1., np.nan, 1.]), np.ones(2)])
def test_public_balanced_fit_rejects_invalid_weights(weights):
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    with pytest.raises(ObservationError):
        balanced_similarity(points, points, weights)


def test_flow_enable_is_a_boolean_contract():
    with pytest.raises(ObservationError):
        InstrumentConfig(enable_flow='false')


@pytest.mark.parametrize('digest', ['a', 'z' * 64, 'A' * 64, '0' * 65])
def test_cohort_requires_real_sha256_provenance(digest):
    with pytest.raises(ObservationError):
        CohortRow('pair', digest, object_digest('b'), {'metric': 1.})


def test_lazy_public_api_preserves_original_classes_and_exposes_version():
    import facemath
    from facemath.core import Frame
    from facemath.observation import FacialInstrument, InstrumentFailure
    assert facemath.Frame is Frame
    assert facemath.FacialInstrument is FacialInstrument
    assert facemath.InstrumentFailure is InstrumentFailure
    assert facemath.__version__ == '0.8.0'
    assert 'FacialInstrument' in dir(facemath)
    with pytest.raises(AttributeError):
        getattr(facemath, 'invented_symbol')
