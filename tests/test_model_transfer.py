from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.study import freeze_study, load_study, read_json
from facemath.observation.study_lab import make_fixture_study
from facemath.observation.study_runner import run_session
from facemath.observation.study_io import finalize_session
from facemath.observation.transfer import export_model, load_capsule, require_transfer
from facemath.observation.transfer_io import finalize_transfer, replay_transfer_bundle, verify_transfer_bundle


@pytest.fixture(scope='module')
def donor(tmp_path_factory):
    root = tmp_path_factory.mktemp('donor')
    spec, executor, _ = make_fixture_study(root/'inputs', development=6, calibration=12, evaluation=6)
    data = read_json(spec); data['bias_mode'] = 'development_joint_ocular'
    spec.write_text(json.dumps(data))
    study = freeze_study(spec, root/'study')
    run_session(study.root, root/'session', executor)
    bundle = finalize_session(root/'session', root/'qualification')
    export_model(bundle, root/'model.json')
    return root, read_json(root/'model.json')


def target(tmp_path):
    spec, executor, _ = make_fixture_study(tmp_path/'inputs', seed=91014, development=0, calibration=0, evaluation=10)
    data = read_json(spec); data.update(bias_mode='none', domain='explicit transfer stress domain')
    for case in data['cases']:
        case['group_id'] = 'transfer_' + case['group_id']
    spec.write_text(json.dumps(data))
    return spec, executor


def test_portable_capsule_preserves_complete_model_bytes_and_all_enrollment(donor):
    root, capsule = donor
    model = load_capsule(capsule)
    assert capsule['model_json'].encode() == (root/'qualification/localization_calibration.json').read_bytes()
    assert model.digest == read_json(root/'qualification/qualification.json')['selected_error_model']['calibration_sha256']
    assert len(json.loads(capsule['study_json'])['cases']) == 25
    broken = deepcopy(capsule); broken['model_json'] += ' '
    broken['content_sha256'] = object_digest({k:v for k,v in broken.items() if k!='content_sha256'})
    with pytest.raises(ObservationError, match='artifact binding'):
        load_capsule(broken)


@pytest.mark.parametrize('donor_index', [0, 6, 18, 24])
@pytest.mark.parametrize('token', ['image_sha256', 'original_encoded_sha256', 'group_id', 'lineage_ids', 'original_source'])
def test_full_donor_enrollment_blocks_aliases_even_evaluation_and_excluded(donor, tmp_path, donor_index, token):
    from dataclasses import replace
    root, capsule = donor
    spec, _ = target(tmp_path)
    study = freeze_study(spec, tmp_path/'study', frozen_model=capsule)
    cases = [(c, study.references(c)) for c in study.cases]
    old = json.loads(capsule['study_json'])['cases'][donor_index]
    if token == 'original_source':
        case, refs = cases[0]
        refs = (replace(refs[0], metadata_json=json.dumps({'source_raster_sha256': old['image_sha256']})),)
        cases[0] = case, refs
    else:
        cases[0][0][token] = old[token]
    with pytest.raises(ObservationError, match='full donor enrollment'):
        require_transfer(capsule, study.data, cases)


@pytest.mark.parametrize('edit,match', [
    (lambda d:d['cases'][0].update(split='calibration'), 'evaluation cases ONLY'),
    (lambda d:d.update(alpha=.2), 'contracts'),
    (lambda d:d.update(minimum_raters=2), 'reference-observable'),
    (lambda d:d.update(joint_error={'shrinkage':.02,'floor_eye_spans':.001}), 'joint_error'),
])
def test_transfer_preflight_rejects_training_or_protocol_changes(donor, tmp_path, edit, match):
    _, capsule = donor
    spec, executor = target(tmp_path)
    data = read_json(spec); edit(data); spec.write_text(json.dumps(data))
    with pytest.raises(ObservationError, match=match):
        freeze_study(spec, tmp_path/'study', frozen_model=capsule)
    assert executor.calls == 0


def test_transfer_has_no_fit_keeps_failures_and_raw_arrays_then_replays(donor, tmp_path, monkeypatch):
    _, capsule = donor
    spec, executor = target(tmp_path)
    study = freeze_study(spec, tmp_path/'study', frozen_model=capsule)
    assert executor.calls == 0
    run_session(study.root, tmp_path/'session', executor)
    before = {p.relative_to(tmp_path/'session'): p.read_bytes() for p in (tmp_path/'session').rglob('*') if p.is_file()}
    def no_fit(*args, **kwargs):
        pytest.fail('transfer attempted a fit')
    for module, names in [
        ('facemath.observation.qualification', ['fit_localization_calibration', 'fit_joint_calibration', 'fit_bias_calibration']),
        ('facemath.observation.validation_io', ['fit_localization_calibration']),
        ('facemath.observation.joint', ['fit_development_bias', 'fit_development_scatter', 'fit_localization_calibration']),
    ]:
        for name in names:
            monkeypatch.setattr(module+'.'+name, no_fit)
    with pytest.raises(ObservationError, match='transfer-finalize'):
        finalize_session(tmp_path/'session', tmp_path/'refused-fit')
    bundle = finalize_transfer(tmp_path/'session', tmp_path/'transfer')
    assert verify_transfer_bundle(bundle)['verified']
    report = read_json(bundle/'transfer.json')
    assert not report['fit_performed'] and not report['empirical_admission']
    assert report['enrolled_cases'] == 11 and report['eligible_cases'] == 10
    assert report['outcomes']['no_face'] == 1
    assert report['coverage']['evaluation_groups'] == 10
    assert report['bound_utility']['splits']['evaluation']['landmark_diameter']['total'] == 88
    assert report['raw_localization']['attempted_cases'] == 10
    assert (bundle/'localization_calibration.json').read_bytes() == capsule['model_json'].encode()
    assert not (bundle/'raw_validation/localization_calibration.json').exists()
    assert replay_transfer_bundle(bundle, tmp_path/'replay')['replay_verified']
    assert all((tmp_path/'session'/n).read_bytes() == b for n,b in before.items())
    with np.load(bundle/'raw_validation/arrays.npz') as left, np.load(tmp_path/'replay/raw_validation/arrays.npz') as right:
        assert all(np.array_equal(left[k], right[k]) for k in left.files)
    with pytest.raises(FileExistsError):
        finalize_transfer(tmp_path/'session', bundle)


def test_model_tampering_is_rejected_without_overwriting_results(donor, tmp_path):
    _, capsule = donor
    spec, executor = target(tmp_path)
    study = freeze_study(spec, tmp_path/'study', frozen_model=capsule)
    run_session(study.root, tmp_path/'session', executor)
    bundle = finalize_transfer(tmp_path/'session', tmp_path/'transfer')
    path = bundle/'localization_calibration.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ObservationError, match='inventory'):
        verify_transfer_bundle(bundle)


def test_cli_exposes_explicit_transfer_routes():
    from facemath.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(['dataset','run','data','--frozen-model','model.json','--out','run']).frozen_model == 'model.json'
    assert parser.parse_args(['study-freeze','spec','--frozen-model','model','--out','lock']).frozen_model == 'model'
    assert parser.parse_args(['transfer-finalize','session','--out','result']).command == 'transfer-finalize'
