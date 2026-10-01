from dataclasses import replace
from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.milestone_lab import make_milestone_workspace_spec
from facemath.observation.reference import ReferenceAnnotation
from facemath.observation.study import read_json, write_json_new, load_study, file_inventory
from facemath.observation.workspace import (prepare_workspace,import_reference,workspace_status,
    freeze_workspace,verify_workspace_origin)

@pytest.fixture
def prepared(tmp_path):
    path,executor,labels=make_milestone_workspace_spec(tmp_path/'inputs',compact=True)
    obj=read_json(path)
    # Enough sources for the freezer, one held-out group/category is deliberately
    # underpowered. Workspace readiness is not acceptance.
    chosen=obj['sources'][:6]+[next(s for s in obj['sources'] if s['split']=='calibration'),
        next(s for s in obj['sources'] if s['split']=='evaluation'),obj['sources'][-1]]
    obj['sources']=chosen;path.write_text(json.dumps(obj))
    root=prepare_workspace(path,tmp_path/'workspace')
    labels=[p for p in labels if p[0] in {s['case_id'] for s in chosen}]
    return root,labels,path


def fill(root,labels):
    for cid,path in labels:import_reference(root,path,cid)


def test_blank_workbench_and_readiness_never_claim_observation(prepared):
    root,labels,_=prepared
    status=workspace_status(root)
    assert not status['ready_to_freeze_labels'] and not status['native_inference_executed']
    assert status['retained_annotation_revisions']==0
    obj=read_json(root/'workspace.json')
    assert obj['predictions_shown'] is False
    assert list((root/'labels').iterdir())==[]
    assert all((root/name).is_file() for c in obj['cases'] for name in c['workbenches'].values())
    assert obj['cases'][-1]['workbenches']=={}


def test_latest_revision_freezes_without_destroying_first(prepared,tmp_path):
    root,labels,_=prepared;fill(root,labels)
    cid,path=labels[0];old=ReferenceAnnotation.from_dict(read_json(path))
    xy=old.xy.copy();xy[4,0]+=.25
    revised=replace(old,xy=xy)
    new=tmp_path/'corrected.json';write_json_new(new,revised.to_dict())
    first=import_reference(root,path,cid)
    assert first['idempotent'] and not first['imported']
    revision=import_reference(root,new,cid)
    assert revision['receipt']['supersedes_reference_sha256']==old.digest
    status=workspace_status(root)
    assert status['ready_to_freeze_labels'] and status['retained_annotation_revisions']==len(labels)+1
    frozen=freeze_workspace(root,tmp_path/'frozen')
    origin=frozen.data['workspace_origin']
    assert len(origin['reference_revisions'])==len(labels)+1
    assert origin['reference_revisions'][0]['xy']==old.xy.tolist()
    assert revised.digest in origin['selected_reference_sha256']
    assert old.digest not in origin['selected_reference_sha256']
    assert not frozen.data['admission_plan']['algebraically_feasible']
    load_study(frozen.root)


def test_case_cannot_import_other_source_or_unknown_rater(prepared,tmp_path):
    root,labels,_=prepared
    with pytest.raises(ObservationError,match='another source'):import_reference(root,labels[1][1],labels[0][0])
    old=ReferenceAnnotation.from_dict(read_json(labels[0][1]))
    other=tmp_path/'other.json';write_json_new(other,replace(old,rater_id='invented-rater').to_dict())
    with pytest.raises(ObservationError,match='rater'):import_reference(root,other,labels[0][0])


def test_excluded_source_cannot_acquire_labels(prepared):
    root,labels,_=prepared
    cid=read_json(root/'workspace.json')['cases'][-1]['case_id']
    with pytest.raises(ObservationError,match='included'):import_reference(root,labels[0][1],cid)


def test_orphan_label_never_promoted(prepared):
    root,labels,_=prepared
    shutil.copyfile(labels[0][1],root/'labels/revision_000000.reference.json')
    with pytest.raises(ObservationError,match='uncommitted'):workspace_status(root)


def test_changed_source_bytes_fail_before_freeze(prepared):
    root,_,_=prepared
    case=read_json(root/'workspace.json')['cases'][0]
    with (root/case['image']).open('ab') as out:out.write(b'tamper')
    with pytest.raises(ObservationError,match='inventory'):workspace_status(root)


def test_rehashed_false_supersession_is_semantically_rejected(prepared):
    root,labels,_=prepared
    import_reference(root,labels[0][1],labels[0][0])
    path=root/'labels/revision_000000.receipt.json';data=read_json(path)
    data.pop('content_sha256');data['supersedes_reference_sha256']='a'*64
    data['content_sha256']=object_digest(data);path.write_text(json.dumps(data))
    with pytest.raises(ObservationError,match='supersession'):workspace_status(root)


def test_unobservable_reference_anchor_not_counted_as_readiness(prepared,tmp_path):
    root,labels,_=prepared;fill(root,labels)
    cid,path=labels[0];old=ReferenceAnnotation.from_dict(read_json(path))
    xy=old.xy.copy();xy[0]=0;visible=old.visible.copy();visible[0]=False
    path=tmp_path/'missing.json';write_json_new(path,replace(old,xy=xy,visible=visible).to_dict())
    import_reference(root,path,cid)
    status=workspace_status(root)
    assert not status['ready_to_freeze_labels']
    assert any('four eye corners' in s for s in status['blockers'])
    with pytest.raises(ObservationError):freeze_workspace(root,tmp_path/'frozen')


@pytest.mark.parametrize('bad', ['references','observation','failure','prediction','unexpected'])
def test_source_manifest_cannot_inject_observer_or_labels(prepared,tmp_path,bad):
    _,_,path=prepared;data=read_json(path);data['sources'][0][bad]='invented'
    path.write_text(json.dumps(data))
    with pytest.raises(ObservationError):prepare_workspace(path,tmp_path/'new')


@pytest.mark.parametrize('bad', ['../outside.png','/absolute.png','..\\outside.png'])
def test_source_path_escape_rejected(prepared,tmp_path,bad):
    _,_,path=prepared;data=read_json(path);data['sources'][0]['image']=bad
    path.write_text(json.dumps(data))
    with pytest.raises(ObservationError):prepare_workspace(path,tmp_path/'new')


def test_group_dependency_cannot_cross_splits(prepared,tmp_path):
    _,_,path=prepared;data=read_json(path)
    data['sources'][-2]['lineage_ids']=data['sources'][0]['lineage_ids']
    path.write_text(json.dumps(data))
    with pytest.raises(ObservationError,match='crosses'):prepare_workspace(path,tmp_path/'new')


def test_freezer_rechecks_history_against_actual_reference(prepared,tmp_path):
    root,labels,_=prepared;fill(root,labels)
    frozen=freeze_workspace(root,tmp_path/'frozen');data=frozen.data
    data['workspace_origin']['selected_reference_sha256'][0]='f'*64
    with pytest.raises(ObservationError,match='selected-label'):verify_workspace_origin(data,frozen.root)


def test_workspace_is_create_only(prepared):
    root,_,path=prepared
    with pytest.raises(FileExistsError):prepare_workspace(path,root)
