from dataclasses import replace
from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from facemath import cli
from facemath.observation.acquisition import ApertureSource,MeasuredGroup,load_group
from facemath.observation.acquisition_lab import APERTURE
from facemath.observation.contracts import ObservationError,object_digest
from facemath.observation.deployment import ValidatedQualification,load_snapshot
from facemath.observation.milestone_lab import run_milestone_laboratory,_control_record
from facemath.observation.artifacts import verify_bundle
from facemath.observation.study import read_json,file_inventory,load_rgb
from facemath.observation.study_io import verify_qualification_bundle
from facemath.observation.readout import inspect_readout,pair_readout

@pytest.fixture(scope='module')
def milestone(tmp_path_factory):
    root=tmp_path_factory.mktemp('wave10')/'lab'
    result=run_milestone_laboratory(root,compact=True)
    return root,result

@pytest.fixture(scope='module')
def accepted(milestone):
    return ValidatedQualification.from_directory(milestone[0]/'positive/qualification')

@pytest.fixture(scope='module')
def rejected(milestone):
    return ValidatedQualification.from_directory(milestone[0]/'negative/qualification')


def test_both_frozen_branches_run_and_raw_geometry_stays_unchanged(milestone):
    root,report=milestone
    assert report['positive_control']['admission']['passed']
    assert not report['negative_control']['admission']['passed']
    assert report['positive_control']['admission']['policy']==report['negative_control']['admission']['policy']
    for name in ('positive_control','negative_control'):
        c=report[name]
        assert c['labels_initially_missing'] and c['ready_after_import']
        assert c['observer_calls']==[4,38,38] and c['enrollment']['prospectively_excluded']==1
        assert c['duplicate_import_idempotent'] and c['frozen_origin_present']
    pair=report['admitted_pair']
    # Original 135, four structural, ten regional, and two diagnostic masks.
    # Every new observed component must participate in qualification invariance.
    assert pair['raw_arrays_unchanged'] and pair['raw_arrays_checked']==135+4+10+2
    with np.load(root/'admitted_pair/arrays.npz') as arrays:
        assert all('structure_'+name in arrays.files for name in
                   ('nominal', 'nominal_valid', 'expression_a', 'expression_b'))
        assert sum(name.startswith('regional_') for name in arrays.files)==10
        assert {'heldout_fit_supported', 'sensitivity_draw_supported'} <= set(arrays.files)
    assert pair['known_reference_shape_enclosed'] and pair['verified']
    assert report['withheld_query']['observer_calls']==0
    journal=(root/'withheld_query.events.jsonl').read_text()
    assert 'admission.require_recomputed' in journal and 'observation.acquire' not in journal


def test_snapshot_reload_recomputes_same_admission(accepted,milestone):
    root,_=milestone
    other=load_snapshot(root/'admitted_pair/qualification_snapshot.zip')
    assert other.summary==accepted.summary
    assert other.summary['raw_qualification_recomputed'] and not other.summary['observer_reexecuted']
    assert other.summary['scope']=='synthetic control only'
    accepted.require_admitted()


def test_withheld_qualification_cannot_authorize_pair(rejected):
    with pytest.raises(ObservationError,match='withheld'):rejected.require_admitted()


def test_summary_view_does_not_create_a_new_score(milestone):
    root,_=milestone;p=root/'admitted_pair'
    data=inspect_readout(p)
    assert data['empirical_admission']['status']=='admitted_synthetic_control'
    assert data['localization_calibrated'] and data['source_measured']
    assert data['registered_all_rms']['value'] is not None
    assert data['shape_distance']['observed_rms']>0
    assert sum(f['prespecified_relations'] for f in data['families'])==256
    assert read_json(p/'summary.json')==pair_readout(read_json(p/'evidence.json'))
    assert read_json(p/'evidence.json')['consensus_similarity_percent'] is None


def test_pair_replay_uses_same_complete_qualification(milestone,tmp_path):
    root,_=milestone
    args=cli.build_parser().parse_args(['replay',str(root/'admitted_pair'),'--out',str(tmp_path/'replay')])
    result=cli.execute(args)
    assert result['replay_verified']
    before=read_json(root/'admitted_pair/evidence.json')
    after=read_json(tmp_path/'replay/evidence.json')
    assert before['empirical_admission']==after['empirical_admission']
    assert before['intrinsic_profile']==after['intrinsic_profile']


def test_rehashed_derived_approval_rejected_by_raw_recomputation(milestone,tmp_path):
    source=milestone[0]/'positive/qualification';target=tmp_path/'forged'
    shutil.copytree(source,target)
    report=read_json(target/'qualification.json');report.pop('content_sha256')
    report['empirical_admission']['categories'][0]['coverage_lower']=.99999
    report['empirical_admission'].pop('content_sha256')
    report['empirical_admission']['content_sha256']=object_digest(report['empirical_admission'])
    report['content_sha256']=object_digest(report)
    (target/'qualification.json').write_text(json.dumps(report))
    manifest=read_json(target/'manifest.json');manifest.pop('content_sha256')
    manifest['files']=file_inventory(target,exclude=('manifest.json',))
    manifest['content_sha256']=object_digest(manifest)
    (target/'manifest.json').write_text(json.dumps(manifest))
    # Self-signed byte consistency alone passes. Raw replay must disagree.
    assert verify_qualification_bundle(target)['verified']
    with pytest.raises(ObservationError,match='numerical replay differs'):
        ValidatedQualification.from_directory(target)


@pytest.mark.parametrize('split',['development','calibration','evaluation'])
def test_query_context_cannot_reuse_any_enrolled_split(accepted,milestone,split):
    root,_=milestone;ga=load_group(root/'query_group_a')
    case=next(c for c in accepted.study['cases'] if c['split']==split)
    image=load_rgb(root/'positive/qualification/acquisition/study'/case['image'])
    # New identifiers cannot hide identical source pixels, even as unselected context.
    group=MeasuredGroup(ga.policy,'new-name',(ApertureSource(image,('new-unreported-lineage',),APERTURE),))
    with pytest.raises(ObservationError,match='image leaks'):accepted.require_query((ga,group),accepted.calibration)


@pytest.mark.parametrize('kind',['group','lineage','excluded_context'])
def test_full_enrollment_leakage_not_just_calibration(accepted,milestone,kind):
    root,_=milestone;ga=load_group(root/'query_group_a');r=_control_record(999001,'clear')
    case=next(c for c in accepted.study['cases'] if c['eligibility']=='excluded')
    groupid=case['group_id'] if kind=='group' else 'fresh-other'
    lineage=case['lineage_ids'] if kind=='lineage' else ['fresh-lineage']
    image=load_rgb(root/'positive/qualification/acquisition/study'/case['image']) if kind=='excluded_context' else r['image']
    group=MeasuredGroup(ga.policy,groupid,(ApertureSource(image,tuple(lineage),APERTURE),))
    with pytest.raises(ObservationError,match='leaks'):accepted.require_query((ga,group),accepted.calibration)


def test_qualification_does_not_select_only_successful_categories(rejected):
    rows=rejected.report['empirical_admission']['categories']
    assert rows[0]['passes'] and not rows[1]['passes']
    assert not rejected.report['empirical_admission']['passed']


def test_query_model_substitution_rejected(accepted,milestone):
    root,_=milestone;ga=load_group(root/'query_group_a');gb=load_group(root/'query_group_b')
    class Other: digest='0'*64
    with pytest.raises(ObservationError,match='calibration differs'):accepted.require_query((ga,gb),Other())


def test_cli_preserves_actual_failure_journal_before_native_constructor(milestone,tmp_path,monkeypatch,capsys):
    root,_=milestone;out=tmp_path/'refused'
    monkeypatch.setattr(cli,'MediaPipeProvider',lambda *args:pytest.fail('native constructor must not execute'))
    argv=['analyze',str(root/'admitted_pair/source_a.png'),str(root/'admitted_pair/source_b.png'),
        '--qualification',str(root/'negative/qualification'),'--model','missing.task','--out',str(out)]
    with pytest.raises(SystemExit) as err:cli.main(argv)
    assert err.value.code==2 and not out.exists()
    journal=Path(str(out)+'.failure.events.jsonl')
    assert journal.is_file() and 'admission.require_recomputed' in journal.read_text()
    before=journal.read_bytes()
    with pytest.raises(SystemExit):cli.main(argv)
    assert journal.read_bytes()==before
    assert 'not overwritten' in capsys.readouterr().err


def test_conflicting_research_and_qualification_flags_rejected(milestone,tmp_path):
    root,_=milestone;p=root/'admitted_pair'
    args=cli.build_parser().parse_args(['analyze',str(p/'source_a.png'),str(p/'source_b.png'),
        '--qualification',str(root/'positive/qualification'),'--localization-calibration','not-read.json','--out',str(tmp_path/'out')])
    with pytest.raises(ObservationError,match='not both'):cli.execute(args)


def test_overview_terminal_discloses_fraction_semantics(milestone,capsys):
    root,_=milestone
    cli.main(['overview',str(root/'admitted_pair')])
    text=capsys.readouterr().out
    assert 'FACEMATH ◆ EVIDENCE' in text and 'not resemblance probabilities' in text
    assert 'admitted_synthetic_control' in text
