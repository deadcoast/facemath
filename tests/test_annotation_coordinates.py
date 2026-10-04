"""Capture provenance and an actual automated browser download through intake.

The fixture is a retained indexed-raster browser export, not human reference data.
Pointer geometry itself is exercised by scripts/browser_qa_annotation.py.
"""
import json
from pathlib import Path
import re
import shutil

import numpy as np

from facemath.observation.milestone_lab import make_milestone_workspace_spec
from facemath.observation.reference import ReferenceAnnotation
from facemath.observation.study import load_study, read_json, write_json_new
from facemath.observation.workspace import prepare_workspace, import_reference, freeze_workspace

FIXTURE = Path(__file__).parent / 'fixtures/annotation-v20'


def test_synthetic_workbench_emits_importable_reference_kind(tmp_path):
    spec_path, _, labels = make_milestone_workspace_spec(tmp_path/'inputs', compact=True)
    spec=read_json(spec_path);spec['sources']=spec['sources'][:1]
    selected=tmp_path/'inputs/selected.json';write_json_new(selected,spec)
    workspace=prepare_workspace(selected,tmp_path/'workspace')
    html=(workspace/'workbenches/source_00000_rater_000.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S)[1])
    label=read_json(labels[0][1]);label['kind']=payload['kind']
    exported=tmp_path/'exported.json';write_json_new(exported,label)
    assert import_reference(workspace,exported,labels[0][0])['imported']
    assert payload['kind']=='synthetic_fixture'
    assert payload['coordinate_capture']['coordinates']=='zero_based_source_pixel_centers_xy'


def test_browser_download_survives_import_and_freeze_without_coordinate_rewrite(tmp_path):
    original=(FIXTURE/'desktop.reference.json').read_bytes()
    ref=ReferenceAnnotation.from_dict(json.loads(original))
    np.testing.assert_array_equal(ref.xy,[[0,0],[47,0],[0,31],[47,31],[24.5,16.5],[15,21],[32,21],[24,28]])
    spec_path, _, labels = make_milestone_workspace_spec(tmp_path/'inputs',compact=True)
    spec=read_json(spec_path)
    spec['rater_ids'].append(ref.rater_id)
    shutil.copyfile(FIXTURE/'indexed-raster.png',tmp_path/'inputs/browser-raster.png')
    spec['sources'][0]['image']='browser-raster.png'
    spec['sources'][0].pop('acquisition_aperture')
    adapted=tmp_path/'inputs/browser.spec.json';write_json_new(adapted,spec)
    workspace=prepare_workspace(adapted,tmp_path/'workspace')
    receipt=import_reference(workspace,FIXTURE/'desktop.reference.json',labels[0][0])
    for cid,label in labels[1:]:import_reference(workspace,label,cid)
    frozen=freeze_workspace(workspace,tmp_path/'frozen')
    loaded=load_study(frozen.root)
    revisions=loaded.data['workspace_origin']['reference_revisions']
    restored=next(ReferenceAnnotation.from_dict(r) for r in revisions if r['rater_id']==ref.rater_id)
    assert restored.digest==ref.digest==receipt['receipt']['reference_sha256']
    np.testing.assert_array_equal(restored.xy,ref.xy)
    assert restored.to_dict()['metadata']==ref.to_dict()['metadata']
    assert (FIXTURE/'desktop.reference.json').read_bytes()==original
