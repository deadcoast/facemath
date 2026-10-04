from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.enrollment import (export_enrollment, load_receipt, build_registry,
    load_registry, check_enrollment, require_disjoint, require_query_disjoint)
from facemath.observation.study import freeze_study, load_study, read_json, file_inventory
from facemath.observation.study_lab import make_fixture_study
from facemath.observation.study_runner import run_session


def reseal(value):
    value['content_sha256'] = object_digest({k:v for k,v in value.items() if k != 'content_sha256'})
    return value


def input_study(root, seed=71601):
    spec, executor, _ = make_fixture_study(root, seed=seed, development=6, calibration=18, evaluation=10)
    data = read_json(spec)
    # IDs name their source universe; no generic IDs shared between independent fixtures.
    for case in data['cases']:
        case['group_id'] = f'{seed}_' + case['group_id']
    spec.write_text(json.dumps(data))
    return spec, executor


@pytest.fixture(scope='module')
def registered(tmp_path_factory):
    root = tmp_path_factory.mktemp('enrollment')
    spec, executor = input_study(root/'inputs')
    study = freeze_study(spec, root/'study')
    export_enrollment(study.root, root/'before.json')
    status = run_session(study.root, root/'session', executor)
    assert status['outcomes']['no_face'] == 2
    export_enrollment(root/'session/study', root/'after.json')
    assert (root/'before.json').read_bytes() == (root/'after.json').read_bytes()
    receipt = read_json(root/'before.json')
    return study, receipt, build_registry([receipt], 'scripted fixture studies; no real portraits')


@pytest.fixture(scope='module')
def fresh(tmp_path_factory):
    root = tmp_path_factory.mktemp('fresh-enrollment')
    spec, executor = input_study(root/'inputs', seed=71602)
    return spec, executor, freeze_study(spec, root/'study')


def enrolled(study):
    return [(c, study.references(c)) for c in study.cases]


def test_export_retains_failed_and_excluded_full_split_enrollment(registered, tmp_path):
    study, receipt, registry = registered
    rows = load_receipt(receipt)
    assert len(rows) == 35
    assert {r['case_id'] for r in rows} == {c['case_id'] for c in study.cases}
    report = check_enrollment(registry, enrolled(study))
    assert report['enrolled'] == report['overlapping_cases'] == 35
    assert report['eligible'] == 34 and report['prospectively_excluded'] == 1
    assert report['splits'] == {'development': 6, 'calibration': 18, 'evaluation': 11}
    assert all(r['matches'][0]['split'] == r['split'] for r in report['overlaps'])
    target = tmp_path/'receipt.json'
    export_enrollment(study.root, target)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        export_enrollment(study.root, target)
    assert target.read_bytes() == original


@pytest.mark.parametrize('index', [0, 6, 24, 34])
@pytest.mark.parametrize('alias', ['image_sha256', 'original_encoded_sha256', 'group_id',
                                  'lineage_ids', 'source_encoded_sha256', 'source_raster_sha256',
                                  'group_as_lineage', 'lineage_as_group'])
def test_every_split_and_exclusion_blocks_hash_and_cross_role_aliases(registered, fresh, index, alias):
    prior, _, registry = registered
    cases = enrolled(fresh[2])
    old = prior.cases[index]
    case, refs = cases[0]
    if alias.startswith('source_'):
        cases[0] = case, (replace(refs[0], metadata_json=json.dumps({alias: old['image_sha256']})),)
    elif alias == 'image_sha256':
        # Preserve reference binding while simulating a renamed/re-encoded source.
        case[alias] = old[alias]
        cases[0] = case, tuple(replace(r, image_sha256=old[alias]) for r in refs)
    elif alias == 'group_as_lineage':
        case['lineage_ids'] = [old['group_id']]
    elif alias == 'lineage_as_group':
        case['group_id'] = old['lineage_ids'][0]
    else:
        case[alias] = old[alias]
    result = check_enrollment(registry, cases)
    assert result['overlapping_cases'] == 1 and result['enrolled'] == 35
    assert result['overlaps'][0]['matches'][0]['case_id'] == old['case_id']
    with pytest.raises(ObservationError, match='full prior enrollment'):
        require_disjoint(registry, cases)


def test_extension_retains_parent_scope_and_all_cases(registered, fresh, tmp_path):
    _, receipt, registry = registered
    export_enrollment(fresh[2].root, tmp_path/'fresh.json')
    other = read_json(tmp_path/'fresh.json')
    extended = build_registry([other], registry['scope'], prior=registry)
    assert extended['prior'] == registry and len(load_registry(extended)) == 70
    assert not check_enrollment(extended, enrolled(registered[0]))['disjoint']
    assert not check_enrollment(extended, enrolled(fresh[2]))['disjoint']
    assert build_registry([receipt, other], 'explicit two studies') == build_registry([other, receipt], 'explicit two studies')
    with pytest.raises(ObservationError, match='scope'):
        build_registry([other], 'changed scope', prior=registry)
    with pytest.raises(ObservationError, match='duplicate'):
        build_registry([receipt], registry['scope'], prior=registry)
    with pytest.raises(ObservationError, match='new full'):
        build_registry([], registry['scope'], prior=registry)
    altered = deepcopy(extended)
    altered['prior']['entries'] = []
    reseal(altered['prior']); reseal(altered)
    with pytest.raises(ObservationError, match='new full'):
        load_registry(altered)


def test_repeated_sources_across_versions_remain_attributable(registered, tmp_path):
    prior, receipt, _ = registered
    # Change a protocol domain, not the enrolled source bytes; keep both studies.
    spec = read_json(prior.root.parent/'inputs/study.json')
    spec['domain'] += ' distinct declared protocol version'
    path = prior.root.parent/'inputs/repeated-version.json'; path.write_text(json.dumps(spec))
    study = freeze_study(path, tmp_path/'repeated')
    export_enrollment(study.root, tmp_path/'receipt.json')
    registry = build_registry([receipt, read_json(tmp_path/'receipt.json')], 'two versions of one source plan')
    result = check_enrollment(registry, enrolled(prior))
    assert result['registered_cases'] == 70 and result['overlapping_cases'] == 35
    assert all(len(r['matches']) == 2 for r in result['overlaps'])


@pytest.mark.parametrize('edit,match', [
    (lambda r:r.update(study_json=r['study_json']+' '), 'binding'),
    (lambda r:r['references_json'].pop(next(iter(r['references_json']))), 'complete reference'),
    (lambda r:r['references_json'].update(extra='{}'), 'complete reference'),
    (lambda r:r['manifest'].update(study_sha256='0'*64), 'manifest digest'),
])
def test_resealed_receipt_cannot_omit_or_change_bound_data(registered, edit, match):
    receipt = deepcopy(registered[1]); edit(receipt); reseal(receipt)
    with pytest.raises(ObservationError, match=match):
        load_receipt(receipt)


def test_resealed_cropped_enrollment_cannot_use_original_study_pin(registered):
    receipt = deepcopy(registered[1])
    study = json.loads(receipt['study_json']); study['cases'].pop()
    receipt['study_json'] = json.dumps(study); reseal(receipt)
    with pytest.raises(ObservationError, match='binding'):
        load_receipt(receipt)


@pytest.mark.parametrize('edit', [lambda r:r.update(scope=' '), lambda r:r.update(alias_policy='ignore_excluded'),
                                 lambda r:r.update(approved=True)])
def test_registry_never_silently_ignores_unknown_policy_or_fields(registered, edit):
    registry = deepcopy(registered[2]); edit(registry); reseal(registry)
    with pytest.raises(ObservationError):
        load_registry(registry)


def test_registry_freeze_no_observation_and_excluded_target_cannot_hide_alias(registered, fresh, tmp_path):
    spec, executor, _ = fresh
    assert executor.calls == 0
    study = freeze_study(spec, tmp_path/'new', enrollment_registry=registered[2])
    assert executor.calls == 0 and study.data['format'] == 'facemath.prospective_study.v16'
    assert read_json(study.root/'manifest.json')['format'] == 'facemath.study_lock_manifest.v16'
    check = study.data['enrollment_registry']['check']
    assert check['disjoint'] and check['enrolled'] == 35 and check['prospectively_excluded'] == 1
    assert load_study(study.root).digest == study.digest
    data = read_json(spec)
    data['cases'][-1]['group_id'] = registered[0].cases[-1]['group_id']
    # Relative input paths remain beside original assets; the original spec is untouched.
    changed = spec.parent/'overlap.json'; changed.write_text(json.dumps(data))
    with pytest.raises(ObservationError, match='registry overlaps'):
        freeze_study(changed, tmp_path/'refused', enrollment_registry=registered[2])
    assert executor.calls == 0 and not (tmp_path/'refused').exists()


@pytest.mark.parametrize('edit,match', [
    (lambda d:d.pop('enrollment_registry'), 'lost'),
    (lambda d:d['enrollment_registry']['check'].update(enrolled=1), 'recomputation'),
    (lambda d:d['enrollment_registry'].update(content_sha256='0'*64), 'digest'),
    (lambda d:d.update(format='facemath.prospective_study.v5'), 'v16'),
])
def test_resealed_lock_cannot_drop_or_falsify_registry(registered, fresh, tmp_path, edit, match):
    study = freeze_study(fresh[0], tmp_path/'study', enrollment_registry=registered[2])
    data = study.data; edit(data)
    (study.root/'study.lock.json').write_text(json.dumps(data))
    manifest = read_json(study.root/'manifest.json')
    manifest['study_sha256'] = object_digest(data)
    if data['format'].endswith('.v5'):
        manifest['format'] = 'facemath.study_lock_manifest.v5'
    manifest['files'] = file_inventory(study.root, exclude=('manifest.json',))
    (study.root/'manifest.json').write_text(json.dumps(reseal(manifest)))
    with pytest.raises(ObservationError, match=match):
        load_study(study.root)


def test_registry_survives_finalization_replay_and_qualification_snapshot(registered, tmp_path):
    from facemath.observation.study_io import finalize_session, replay_qualification_bundle
    from facemath.observation.deployment import ValidatedQualification
    spec, executor = input_study(tmp_path/'inputs', seed=71603)
    spec_data = read_json(spec); spec_data['bias_mode'] = 'development_joint_ocular'
    spec.write_text(json.dumps(spec_data))
    study = freeze_study(spec, tmp_path/'study', enrollment_registry=registered[2])
    status = run_session(study.root, tmp_path/'session', executor)
    observations = {p.relative_to(tmp_path/'session'):p.read_bytes() for p in (tmp_path/'session').rglob('observation.json')}
    bundle = finalize_session(tmp_path/'session', tmp_path/'qualification')
    assert replay_qualification_bundle(bundle, tmp_path/'replay')['replay_verified']
    assert status['outcomes']['no_face'] == 2
    assert all((tmp_path/'session'/p).read_bytes() == b for p,b in observations.items())
    with np.load(bundle/'arrays.npz') as before, np.load(tmp_path/'replay/arrays.npz') as after:
        assert set(before.files) == set(after.files)
        assert all(np.array_equal(before[k], after[k]) for k in before.files)
    link = ValidatedQualification.from_directory(bundle)
    assert json.loads(link.enrollment_json) == registered[2]
    with pytest.raises(ObservationError, match='withheld'):
        link.require_admitted()
    from facemath.observation.transfer import export_model, load_capsule, require_transfer
    export_model(bundle, tmp_path/'model.json')
    capsule = read_json(tmp_path/'model.json')
    assert capsule['format'] == 'facemath.frozen_model_capsule.v16'
    assert json.loads(capsule['registry_json']) == registered[2]
    load_capsule(capsule)
    # An inherited registry source is not in the new donor's own enrollment.
    cases = enrolled(registered[0])
    for case, _ in cases:
        case['split'] = 'evaluation'
    target = {**spec_data, 'bias_mode': 'none'}
    with pytest.raises(ObservationError, match='registry overlaps'):
        require_transfer(capsule, target, cases)
    broken = deepcopy(capsule); broken['registry_json'] += ' '; reseal(broken)
    with pytest.raises(ObservationError, match='artifact binding'):
        load_capsule(broken)
    broken = deepcopy(capsule); broken['format'] = 'facemath.frozen_model_capsule.v14'; reseal(broken)
    with pytest.raises(ObservationError, match='v16 model capsule'):
        load_capsule(broken)


@pytest.mark.parametrize('alias', ['image', 'group', 'lineage_as_group', 'group_as_lineage'])
def test_prior_enrollment_remains_barred_from_query_context(registered, alias):
    old = registered[0].cases[-1]
    group = SimpleNamespace(group_id='query', lineage_ids=['query-source'],
                            sources=[SimpleNamespace(image_sha256='f'*64)])
    if alias == 'image': group.sources.append(SimpleNamespace(image_sha256=old['image_sha256']))
    if alias == 'group': group.group_id = old['group_id']
    if alias == 'lineage_as_group': group.group_id = old['lineage_ids'][0]
    if alias == 'group_as_lineage': group.lineage_ids = [old['group_id']]
    with pytest.raises(ObservationError, match='query/context'):
        require_query_disjoint(registered[2], (group,))
    # Exercise the admitted-query boundary itself with a lightweight policy stub.
    # This deliberately is not a constructed or scientifically admitted snapshot.
    from facemath.observation.deployment import ValidatedQualification
    calibration = SimpleNamespace(digest='f'*64)
    boundary = SimpleNamespace(require_admitted=lambda: None, calibration=calibration,
        enrollment_json=json.dumps(registered[2]))
    with pytest.raises(ObservationError, match='query/context'):
        ValidatedQualification.require_query(boundary, (group, group), calibration)


def test_cli_exports_builds_and_records_overlap_with_exit_two(registered, tmp_path, capsys):
    from facemath.cli import main
    receipt, registry, result = [tmp_path/n for n in ('receipt.json', 'registry.json', 'check.json')]
    main(['enrollment', 'export', str(registered[0].root), '--out', str(receipt)])
    main(['enrollment', 'build', '--receipt', str(receipt), '--scope', 'explicit fixture', '--out', str(registry)])
    with pytest.raises(SystemExit) as exc:
        main(['enrollment', 'check', str(registry), str(registered[0].root), '--out', str(result)])
    assert exc.value.code == 2 and read_json(result)['overlapping_cases'] == 35
    assert not read_json(result)['inference_executed']


@pytest.mark.parametrize('value', [None, False, [], {}, 'not a registry'])
def test_explicit_invalid_registry_file_is_never_an_absent_option(fresh, registered, tmp_path, value):
    from facemath.cli import main
    from facemath.observation.enrollment import read_registry
    path = tmp_path/'invalid.json'; path.write_text(json.dumps(value))
    with pytest.raises(ObservationError):
        read_registry(path)
    with pytest.raises(SystemExit) as failure:
        main(['study-freeze', str(fresh[0]), '--registry', str(path), '--out', str(tmp_path/'refused')])
    assert failure.value.code == 2 and not (tmp_path/'refused').exists()
    receipt = tmp_path/'receipt.json'; receipt.write_text(json.dumps(registered[1]))
    with pytest.raises(SystemExit) as failure:
        main(['enrollment','build','--receipt',str(receipt),'--scope','invalid prior control',
              '--prior',str(path),'--out',str(tmp_path/'refused-registry.json')])
    assert failure.value.code == 2 and not (tmp_path/'refused-registry.json').exists()


def test_empty_explicit_registry_paths_cannot_disable_checks(fresh, registered, tmp_path):
    from facemath.cli import main
    with pytest.raises(SystemExit) as failure:
        main(['study-freeze',str(fresh[0]),'--registry','','--out',str(tmp_path/'refused')])
    assert failure.value.code == 2 and not (tmp_path/'refused').exists()
    receipt = tmp_path/'receipt.json'; receipt.write_text(json.dumps(registered[1]))
    with pytest.raises(SystemExit) as failure:
        main(['enrollment','build','--receipt',str(receipt),'--scope','empty path control',
              '--prior','','--out',str(tmp_path/'refused-registry.json')])
    assert failure.value.code == 2 and not (tmp_path/'refused-registry.json').exists()
