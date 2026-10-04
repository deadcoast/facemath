from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from facemath.observation.source_cohort import (fetch_cohort, make_recipe, prepare_cohort,
                                         select_members, verify_recipe)
from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.dataset import verify_dataset
from facemath.observation.reference import IBUG68_INDICES, parse_ibug68
from facemath.observation.study import file_inventory, load_rgb, read_json, save_rgb


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest(),
                url='https://example.invalid/' + path.name)


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path/'source'; root.mkdir()
    points = np.full((68, 2), 45.125)
    points[list(IBUG68_INDICES)] = [[30.25,30.5],[40.25,30.5],[55.25,30.5],[65.25,30.5],
                                   [47.25,44.5],[36.25,58.5],[60.25,58.5],[48.25,72.5]]
    pts = ('version: 1\nn_points: 68\n{\n' + '\n'.join(f'{x} {y}' for x,y in points) + '\n}\n').encode()
    members = {}
    for i in range(6):
        path = root/f'image_{i}.png'
        save_rgb(path, np.random.default_rng(i).integers(0, 256, (96,96,3), dtype=np.uint8))
        members[f'data/lfpw/image_{i}.png'] = path.read_bytes()
        members[f'data/lfpw/image_{i}.pts'] = pts
    members['data/lfpw/image_0_mirror.jpg'] = b'not sampled or decoded'
    archive = root/'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name, data in members.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            output.addfile(info, io.BytesIO(data))
    model = root/'model.task'; model.write_bytes(b'test fixture; not an actual observer')
    plan = dict(format='facemath.pts_cohort_plan.v12', name='fixture cohort', source='fixture',
        label_source='PTS adapter fixture', image_terms='generated fixture pixels',
        grouping_limit='fixture only', runtime_version='1.0.1', seed='fixture-before-outcomes',
        selection='sha256_seed_nul_member_blocks_v1', candidate_prefix='data/lfpw',
        split_counts={'development':2,'calibration':1,'evaluation':1},
        coordinate_origin=1, box_margin=.25, archive=pin(archive), model=pin(model),
        protocol=dict(domain='fixture',minimum_raters=1,maximum_cases_per_group=1,
                      minimum_development_groups=2,alpha=.1,bias_mode='development_joint_ocular',strata=['all']))
    plan_path = root/'plan.json'; plan_path.write_text(json.dumps(plan))
    inventory = [dict(name=name, bytes=len(data), sha256=sha256(data).hexdigest(), type='file') for name,data in members.items()]
    return dict(root=root, plan=plan, path=plan_path, archive=archive, model=model, inventory=inventory, points=points)


def test_selection_is_order_independent_and_excludes_mirrors(cohort):
    plan, rows = cohort['plan'], cohort['inventory']
    selected = select_members(plan, rows)
    assert selected == select_members(plan, list(reversed(rows)))
    assert len(selected) == 4 and {r['candidate_count'] for r in selected} == {6}
    assert [r['split'] for r in selected] == ['development','development','calibration','evaluation']
    assert all('_mirror' not in r['image'] for r in selected)
    assert len({r['image'] for r in selected}) == 4


def test_evaluation_only_selection_is_explicit_and_never_allocates_fit_cases(cohort):
    plan = deepcopy(cohort['plan'])
    plan['split_counts'] = dict(development=0, calibration=0, evaluation=4)
    with pytest.raises(ObservationError, match='positive'):
        select_members(plan, cohort['inventory'])
    plan['evaluation_only'] = True
    with pytest.raises(ObservationError, match='fit'):
        select_members(plan, cohort['inventory'])
    plan['protocol']['bias_mode'] = 'none'
    selected = select_members(plan, cohort['inventory'])
    assert len(selected) == 4 and all(r['split'] == 'evaluation' for r in selected)
    assert selected == select_members(plan, cohort['inventory'][::-1])
    plan['split_counts']['calibration'] = 1
    with pytest.raises(ObservationError, match='zero'):
        select_members(plan, cohort['inventory'])


def test_helen_group_selection_never_splits_related_sources(cohort):
    plan = deepcopy(cohort['plan'])
    plan['selection'] = 'sha256_seed_nul_helen_source_blocks_v1'
    inventory = deepcopy(cohort['inventory'])
    for row in inventory:
        row['name'] = row['name'].replace('image_', '1234_')
    with pytest.raises(ObservationError, match='smaller'):
        select_members(plan, inventory)
    for row in inventory:
        # Three related pairs, plus three distinct roots gives four groups.
        for i in range(3, 6):
            row['name'] = row['name'].replace(f'1234_{i}.', f'{4000+i}_1.')
    selected = select_members(plan, inventory)
    assert len({r['source_group'] for r in selected}) == 4
    assert {r['candidate_count'] for r in selected} == {6}
    assert {r['candidate_groups'] for r in selected} == {4}
    assert selected == select_members(plan, inventory[::-1])
    recipe = make_recipe(plan, inventory)
    assert len({r['group_id'] for r in recipe['images']}) == 4
    verify_recipe(recipe)
    recipe['cohort']['selection'][0]['source_group'] += '1'
    with pytest.raises(ObservationError, match='frozen selection'):
        verify_recipe(recipe)


def test_helen_grouping_refuses_unknown_filename_convention(cohort):
    plan = deepcopy(cohort['plan'])
    plan['selection'] = 'sha256_seed_nul_helen_source_blocks_v1'
    with pytest.raises(ObservationError, match='source_face'):
        select_members(plan, cohort['inventory'])


def test_ibug_groups_optional_face_suffixes_and_leading_zero_aliases(cohort):
    plan = deepcopy(cohort['plan']); inventory = deepcopy(cohort['inventory'])
    plan['selection'] = 'sha256_seed_nul_ibug_source_blocks_v1'
    for row in inventory:
        row['name'] = row['name'].replace('image_0.', 'image_000_01.').replace('image_1.', 'image_0_2.')
    inventory[2]['sha256'] = inventory[0]['sha256']  # same photograph, two face labels
    selected = select_members(plan, inventory)
    assert len({r['source_group'] for r in selected}) == 4
    assert {r['candidate_groups'] for r in selected} == {5}
    assert selected == select_members(plan, inventory[::-1])
    alias = deepcopy(inventory); alias[4]['sha256'] = alias[0]['sha256']
    with pytest.raises(ObservationError, match='duplicate original'):
        select_members(plan, alias)
    inventory[0]['name'] = inventory[0]['name'].replace('image_000_01', 'unknown')
    with pytest.raises(ObservationError):
        select_members(plan, inventory)


def test_intake_transfer_preserves_original_pts_precision_and_pixels(cohort, tmp_path):
    dataset = prepare_cohort(cohort['path'], cohort['archive'], cohort['model'], tmp_path/'intake')
    result = verify_dataset(dataset)
    assert result['reference_kind'] == 'external_dataset' and result['eligible'] == 4
    recipe = read_json(dataset/'recipe.json')
    selected = recipe['cohort']['selection'][0]
    original = cohort['root']/Path(selected['image']).name
    conversion = read_json(dataset/'conversions.json')[0]
    x0,y0,x1,y1 = conversion['crop_xyxy_exclusive']
    # Inclusive integer extrema: [30,30] to [66,73], origin correction once.
    assert [x0,y0,x1,y1] == [19,18,76,84]
    np.testing.assert_array_equal(load_rgb(dataset/'inputs/case_00000.png'),load_rgb(original)[18:84,19:76])
    reference = read_json(dataset/'inputs/case_00000.reference.json')
    np.testing.assert_array_equal(reference['xy'],cohort['points'][list(IBUG68_INDICES)] - 1 - [19,18])
    assert reference['kind'] == 'external_dataset' and reference['prediction_exposure'] == 'unknown'
    raw = (dataset/'assets/source_00000.pts').read_text()
    np.testing.assert_array_equal(parse_ibug68(raw),cohort['points'])
    # A transfer needs only the retained dataset, not the large source tar.
    import shutil
    shutil.copytree(dataset,tmp_path/'transfer')
    assert verify_dataset(tmp_path/'transfer')['conversion_recomputed']
    with pytest.raises(FileExistsError):
        prepare_cohort(cohort['path'],cohort['archive'],cohort['model'],tmp_path/'intake')


@pytest.mark.parametrize('mutation,match',[
    (lambda p:p['split_counts'].update(evaluation=100),'smaller'),
    (lambda p:p.update(coordinate_origin=True),'origin'),
    (lambda p:p.update(candidate_prefix='../data'),'basename'),
    (lambda p:p.update(selection='choose_detector_successes'),'algorithm'),
    (lambda p:p['protocol'].update(admission_policy={}),'empirical admission'),
    (lambda p:p['archive'].update(url='http://example.invalid/a'),'HTTPS'),
    (lambda p:p['model'].update(bytes=True),'byte limit'),
])
def test_invalid_prospective_plan(cohort,mutation,match):
    mutation(cohort['plan'])
    with pytest.raises(ObservationError,match=match):select_members(cohort['plan'],cohort['inventory'])


@pytest.mark.parametrize('mode', ['missing_pts','missing_image','ambiguous_image','duplicate_image','path_alias'])
def test_candidate_universe_cannot_silently_drop_or_alias_sources(cohort,mode):
    rows = deepcopy(cohort['inventory'])
    if mode == 'missing_pts': rows.pop(1)
    if mode == 'missing_image': rows.pop(0)
    if mode == 'ambiguous_image': rows.append({**rows[0],'name':'data/lfpw/image_0.jpg'})
    if mode == 'duplicate_image': rows[2]['sha256'] = rows[0]['sha256']
    if mode == 'path_alias': rows.append({**rows[0],'name':rows[0]['name'].upper()})
    with pytest.raises(ObservationError):select_members(cohort['plan'],rows)


@pytest.mark.parametrize('name,kind', [('../escape',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),
    ('data\\escape',tarfile.REGTYPE),('data/CON.txt',tarfile.REGTYPE),
    ('data/link',tarfile.SYMTYPE),('data/link',tarfile.LNKTYPE),('data/fifo',tarfile.FIFOTYPE),
    ('data/extension',tarfile.XHDTYPE),('data/longname',tarfile.GNUTYPE_LONGNAME)])
def test_archive_rejects_unsafe_members_and_preserves_failure(cohort,tmp_path,name,kind):
    with tarfile.open(cohort['archive'],'w:gz') as output:
        info=tarfile.TarInfo(name);info.type=kind;info.linkname='outside'
        output.addfile(info,io.BytesIO(b''))
    cohort['plan']['archive']=pin(cohort['archive']);cohort['path'].write_text(json.dumps(cohort['plan']))
    with pytest.raises(ObservationError):
        prepare_cohort(cohort['path'],cohort['archive'],cohort['model'],tmp_path/'failed')
    assert read_json(tmp_path/'failed/intake.failure.json')['completed'] is False
    assert not (tmp_path/'escape').exists() and not (tmp_path/'failed/dataset').exists()


def test_resealed_selection_and_split_tampering_is_recomputed(cohort,tmp_path):
    dataset=prepare_cohort(cohort['path'],cohort['archive'],cohort['model'],tmp_path/'intake')
    recipe=read_json(dataset/'recipe.json');recipe['images'][0]['split']='evaluation'
    (dataset/'recipe.json').write_text(json.dumps(recipe))
    manifest={'format':'facemath.dataset_bundle.v11','files':file_inventory(dataset,exclude=('manifest.json',))}
    manifest['content_sha256']=object_digest(manifest)
    (dataset/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ObservationError,match='frozen selection'):verify_dataset(dataset)


def test_protocol_and_label_kind_cannot_change_after_selection(cohort):
    recipe=make_recipe(cohort['plan'],cohort['inventory'])
    for key,value in [('label_kind','human_blind'),('coordinate_origin',0),('box_margin',.9)]:
        changed=deepcopy(recipe);changed[key]=value
        with pytest.raises(ObservationError,match='frozen selection'):verify_recipe(changed)


def test_bad_pin_fails_before_intake_directory(cohort,tmp_path):
    cohort['archive'].write_bytes(b'changed archive')
    with pytest.raises(ObservationError,match='pin mismatch'):
        prepare_cohort(cohort['path'],cohort['archive'],cohort['model'],tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_download_failure_retains_actual_partial_bytes(cohort,tmp_path,monkeypatch):
    class Response(io.BytesIO):
        url='https://example.invalid/source'
    monkeypatch.setattr('facemath.observation.source_cohort.urlopen',lambda *a,**kw:Response(b'wrong bytes'))
    with pytest.raises(ObservationError,match='pin mismatch'):fetch_cohort(cohort['path'],tmp_path/'fetch')
    assert (tmp_path/'fetch/source.tar.gz').read_bytes()==b'wrong bytes'
    assert read_json(tmp_path/'fetch/fetch.failure.json')['verified_downloads']==[]


def test_cli_cohort_intake_and_existing_cohort_analysis_coexist(cohort,tmp_path,capsys):
    from facemath.cli import main
    from facemath.observation.cohort import Cohort
    assert Cohort is not None
    main(['dataset','cohort',str(cohort['path']),'--archive',str(cohort['archive']),
          '--model',str(cohort['model']),'--out',str(tmp_path/'cli')])
    result=json.loads(capsys.readouterr().out)
    assert result['verified'] and result['cases']==4


def test_declared_oversized_member_is_refused_before_payload_read(cohort,tmp_path):
    with tarfile.open(cohort['archive'],'w:gz') as output:
        info=tarfile.TarInfo('data/huge');info.size=128*1024**2+1
        output.fileobj.write(info.tobuf(format=tarfile.USTAR_FORMAT))
    cohort['plan']['archive']=pin(cohort['archive']);cohort['path'].write_text(json.dumps(cohort['plan']))
    with pytest.raises(ObservationError,match='member size'):
        prepare_cohort(cohort['path'],cohort['archive'],cohort['model'],tmp_path/'huge')


def window_plan(cohort, offset=0, count=4):
    plan = deepcopy(cohort['plan'])
    plan.update(format='facemath.pts_cohort_plan.v15', selection_offset=offset,
                selection='sha256_seed_nul_ibug_source_blocks_v1', evaluation_only=True,
                split_counts=dict(development=0, calibration=0, evaluation=count))
    plan['protocol']['bias_mode'] = 'none'
    return plan


def test_explicit_correspondence_survives_preparation_and_reverification(cohort, tmp_path):
    from facemath.observation.correspondence import MP_REFERENCE8_V2
    from facemath.observation.study import ObserverDeclaration
    plan = window_plan(cohort)
    plan.update(format='facemath.pts_cohort_plan.v17', correspondence=MP_REFERENCE8_V2)
    cohort['path'].write_text(json.dumps(plan))
    dataset = prepare_cohort(cohort['path'], cohort['archive'], cohort['model'], tmp_path/'v17')
    assert verify_dataset(dataset)['conversion_recomputed']
    recipe = read_json(dataset/'recipe.json')
    assert recipe['correspondence'] == MP_REFERENCE8_V2
    assert recipe['format'] == 'facemath.pts_dataset_recipe.v17'
    decl = ObserverDeclaration.from_dict(read_json(dataset/'study.spec.json')['observer'])
    assert decl.correspondence == MP_REFERENCE8_V2
    recipe['correspondence'] = None
    with pytest.raises(ObservationError, match='frozen selection'):
        verify_recipe(recipe)


@pytest.mark.parametrize('identity', [None, '', 'unknown', 4])
def test_correspondence_plan_refuses_missing_or_unknown_identity(cohort, identity):
    plan = window_plan(cohort)
    plan.update(format='facemath.pts_cohort_plan.v17', correspondence=identity)
    with pytest.raises(ObservationError, match='correspondence'):
        select_members(plan, cohort['inventory'])


def test_correspondence_cannot_be_silently_ignored_by_old_plan(cohort):
    from facemath.observation.correspondence import MP_REFERENCE8_V2
    plan = window_plan(cohort)
    plan['correspondence'] = MP_REFERENCE8_V2
    with pytest.raises(ObservationError, match='v17'):
        make_recipe(plan, cohort['inventory'])


@pytest.mark.parametrize('selector', ['ibug', 'helen'])
def test_windows_partition_ranked_roots_after_grouping(cohort, selector):
    plan = window_plan(cohort, count=5)
    inventory = deepcopy(cohort['inventory'])
    # Six annotations but five roots. Related faces must not straddle windows.
    for row in inventory:
        row['name'] = row['name'].replace('image_1.', 'image_000_02.')
        if selector == 'helen':
            row['name'] = row['name'].replace('image_000_02.', '000_02.')
            for i in (0, 2, 3, 4, 5):
                row['name'] = row['name'].replace(f'image_{i}.', f'{i}_01.')
    if selector == 'helen':
        plan['selection'] = 'sha256_seed_nul_helen_source_blocks_v1'
    all_rows = select_members(plan, inventory)
    first, second = deepcopy(plan), deepcopy(plan)
    first['split_counts']['evaluation'] = 2
    second.update(selection_offset=2)
    second['split_counts']['evaluation'] = 3
    left, right = select_members(first, inventory), select_members(second, inventory[::-1])
    assert left + right == all_rows
    assert not {r['source_group'] for r in left} & {r['source_group'] for r in right}
    assert [r['selection_index'] for r in right] == [2, 3, 4]
    assert {r['candidate_count'] for r in all_rows} == {6}
    assert {r['candidate_groups'] for r in all_rows} == {5}
    recipe = make_recipe(second, inventory)
    assert recipe['cohort']['selection_window'] == dict(unit='photograph_filename_root',
        start_inclusive=2, stop_exclusive=5, candidate_annotations=6,
        candidate_source_groups=5, skipped_before=2, selected=3, remaining_after=0)
    verify_recipe(recipe)


def test_window_split_blocks_and_zero_offset_preserve_legacy_order(cohort):
    plan = deepcopy(cohort['plan'])
    plan['selection'] = 'sha256_seed_nul_ibug_source_blocks_v1'
    old = select_members(plan, cohort['inventory'])
    old_recipe = make_recipe(plan, cohort['inventory'])
    plan.update(format='facemath.pts_cohort_plan.v15', selection_offset=0)
    current = select_members(plan, cohort['inventory'])
    assert [{k:v for k,v in r.items() if k != 'selection_index'} for r in current] == old
    assert old_recipe['format'] == 'facemath.pts_dataset_recipe.v12'
    assert 'selection_window' not in old_recipe['cohort']
    plan['selection_offset'] = 2
    current = select_members(plan, cohort['inventory'])
    assert [r['selection_index'] for r in current] == [2, 3, 4, 5]
    assert [r['split'] for r in current] == ['development','development','calibration','evaluation']


@pytest.mark.parametrize('offset', [-1, True, False, 1.0, '1', None, 20001, [], {}])
def test_window_offsets_are_explicit_bounded_integers(cohort, offset):
    with pytest.raises(ObservationError, match='selection_offset'):
        select_members(window_plan(cohort, offset), cohort['inventory'])


def test_missing_offset_and_legacy_or_ungrouped_windows_are_refused(cohort):
    plan = window_plan(cohort)
    del plan['selection_offset']
    with pytest.raises(ObservationError, match='selection_offset'):
        select_members(plan, cohort['inventory'])
    plan = window_plan(cohort)
    plan['format'] = 'facemath.pts_cohort_plan.v12'
    with pytest.raises(ObservationError, match='v15'):
        select_members(plan, cohort['inventory'])
    plan = window_plan(cohort)
    plan['selection'] = 'sha256_seed_nul_member_blocks_v1'
    with pytest.raises(ObservationError, match='photograph-root'):
        select_members(plan, cohort['inventory'])


@pytest.mark.parametrize('offset', [3, 6, 20000])
def test_exhausted_window_never_truncates_wraps_or_replaces(cohort, offset):
    with pytest.raises(ObservationError, match='smaller.*window'):
        select_members(window_plan(cohort, offset), cohort['inventory'])


def test_skipped_candidates_still_require_intact_pairing(cohort):
    plan = window_plan(cohort, offset=2)
    whole = select_members(window_plan(cohort, count=6), cohort['inventory'])
    broken = [r for r in cohort['inventory'] if r['name'] != whole[0]['annotation']]
    with pytest.raises(ObservationError, match='missing PTS'):
        select_members(plan, broken)


@pytest.mark.parametrize('field', ['selection', 'selection_window', 'selection_offset', 'format'])
def test_resealed_window_tampering_is_recomputed(cohort, tmp_path, field):
    plan = window_plan(cohort, offset=2)
    cohort['path'].write_text(json.dumps(plan))
    dataset = prepare_cohort(cohort['path'], cohort['archive'], cohort['model'], tmp_path/'window')
    assert verify_dataset(dataset)['cases'] == 4
    recipe = read_json(dataset/'recipe.json')
    assert recipe['format'] == 'facemath.pts_dataset_recipe.v15'
    if field == 'selection':
        recipe['cohort']['selection'][0]['selection_index'] = 0
    elif field == 'selection_window':
        recipe['cohort']['selection_window']['skipped_before'] = 0
    elif field == 'selection_offset':
        recipe['cohort']['plan']['selection_offset'] = 0
    else:
        recipe['format'] = 'facemath.pts_dataset_recipe.v12'
    (dataset/'recipe.json').write_text(json.dumps(recipe))
    manifest = {'format':'facemath.dataset_bundle.v11', 'files':file_inventory(dataset, exclude=('manifest.json',))}
    manifest['content_sha256'] = object_digest(manifest)
    (dataset/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ObservationError, match='frozen selection'):
        verify_dataset(dataset)
