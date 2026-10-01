import json
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata

import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, object_digest
from facemath.observation.native_check import native_smoke,verify_smoke
from facemath.observation.readout import pair_readout,render_pair_readout,admission_panel,terminal_readout
from facemath.observation.study import save_rgb,file_inventory,read_json
from facemath.observation.study_runner import AttemptOutcome


def missing(name):raise importlib.metadata.PackageNotFoundError(name)


def test_actual_missing_prerequisites_not_an_inference(tmp_path,monkeypatch):
    import facemath.observation.native_check as mod
    monkeypatch.setattr(mod.importlib.metadata,'version',missing)
    monkeypatch.setattr(mod.NativeExecutor,'execute',lambda *args:pytest.fail('cannot invoke observer'))
    source=tmp_path/'source.png';save_rgb(source,np.full((16,16,3),127,np.uint8))
    root=tmp_path/'smoke'
    report=native_smoke(source,tmp_path/'absent.task',root)
    assert report['status']=='not_attempted' and not report['observer_attempted']
    assert len(report['blockers'])==2 and not (root/'attempt').exists()
    assert verify_smoke(root)['verified'] and not report['human_accuracy_validated']
    journal=(root/'events.jsonl').read_text()
    assert 'native_smoke.preflight' in journal and 'native_smoke.execute_existing_worker' not in journal
    with pytest.raises(FileExistsError):native_smoke(source,tmp_path/'absent',root)


@pytest.mark.parametrize('outcome',['no_face','multiple_faces','timeout','runtime_error','interrupted','contract_error'])
def test_native_execution_outcome_contract_with_fixture_not_live(tmp_path,monkeypatch,outcome):
    import facemath.observation.native_check as mod
    monkeypatch.setattr(mod.importlib.metadata,'version',lambda _: 'api-fixture-not-live')
    monkeypatch.setattr(mod.NativeExecutor,'preflight',lambda *args:{'ready':True,'fixture':True})
    def execute(self,image,decl,directory):
        (directory/'stdout.txt').write_text('fixture worker output')
        return AttemptOutcome(outcome,detail='API FIXTURE; no actual model executed')
    monkeypatch.setattr(mod.NativeExecutor,'execute',execute)
    source=tmp_path/'source.png';save_rgb(source,np.full((16,16,3),127,np.uint8))
    model=tmp_path/'model.task';model.write_bytes(b'fake API fixture bytes')
    root=tmp_path/'run';r=native_smoke(source,model,root)
    assert r['observer_attempted'] and r['status']==outcome and not r['human_accuracy_validated']
    assert verify_smoke(root)['verified']


def test_pinned_model_mismatch_stops_before_execution(tmp_path,monkeypatch):
    import facemath.observation.native_check as mod
    monkeypatch.setattr(mod.importlib.metadata,'version',lambda _:'fixture')
    monkeypatch.setattr(mod.NativeExecutor,'execute',lambda *args:pytest.fail('no observer'))
    source=tmp_path/'source.png';save_rgb(source,np.full((16,16,3),127,np.uint8))
    model=tmp_path/'model.task';model.write_bytes(b'known wrong pin')
    report=native_smoke(source,model,tmp_path/'run',expected_sha256='0'*64)
    assert report['status']=='not_attempted' and any('digest' in s for s in report['blockers'])


def test_consistently_rehashed_accuracy_claim_rejected(tmp_path,monkeypatch):
    import facemath.observation.native_check as mod
    monkeypatch.setattr(mod.importlib.metadata,'version',missing)
    source=tmp_path/'source.png';save_rgb(source,np.full((16,16,3),127,np.uint8))
    root=tmp_path/'run';native_smoke(source,tmp_path/'absent',root)
    report=read_json(root/'smoke.json');report['human_accuracy_validated']=True
    (root/'smoke.json').write_text(json.dumps(report))
    manifest=read_json(root/'manifest.json');manifest.pop('content_sha256')
    manifest['files']=file_inventory(root,exclude=('manifest.json',));manifest['content_sha256']=object_digest(manifest)
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ObservationError,match='accuracy'):verify_smoke(root)


def evidence():
    return {'recipe_sha256':'a'*64,'measurement_profile':[],
        'intrinsic_profile':{'shape_distance':{'observed_rms':.1,'available':False},'families':[
        {'family':'distance','prespecified_relations':28,'within':7,'outside':7,'unresolved_or_unavailable':14,
         'within_tolerance_fraction_bounds':[.25,.75]}]},'warnings':[],
        'empirical_admission':{'status':'not_requested'}}


def test_readout_percentages_have_declared_denominators_not_similarity():
    e=evidence();before=json.dumps(e,sort_keys=True)
    data=pair_readout(e)
    assert data['families'][0]['within_tolerance_percent_bounds']==[25.,75.]
    assert 'NOT a probability' in data['percentage_meaning']
    assert json.dumps(e,sort_keys=True)==before
    html=render_pair_readout(e)
    assert '25.00–75.00%' in html and 'Missing relations remain' in html and 'reference-shape range unavailable' in html


def test_unavailable_shape_not_rendered_as_zero():
    e=evidence();e['intrinsic_profile']=None
    assert pair_readout(e)['shape_distance'] is None
    assert 'Unavailable' in render_pair_readout(e)


def test_research_panel_never_claims_empirical_approval():
    html=admission_panel(None)
    assert 'Research analysis' in html and 'not an empirical deployment approval' in html


def test_unsafe_category_html_escaped():
    payload={'status':'withheld','policy':{'evidence_kind':'synthetic_control'},'categories':[
        {'category':'<script>alert(1)</script>','groups':1,'delivered':1,'covered':0,'delivery_lower':.1,'coverage_lower':0.,'passes':False,'reasons':['<bad>']}]}
    html=admission_panel(payload)
    assert '<script>alert' not in html and '&lt;script&gt;' in html and 'SYNTHETIC CONTROL' in html


def test_nominal_uncalibrated_family_does_not_invent_zero_or_percentage():
    e=evidence();f=e['intrinsic_profile']['families'][0]
    f.update(within_tolerance_fraction_bounds=None,within=None,outside=None,unresolved_or_unavailable=None)
    data=pair_readout(e)
    assert data['families'][0]['within_tolerance_percent_bounds'] is None
    assert 'Unavailable — uncalibrated' in render_pair_readout(e)
    data['report']='report.html'
    text=terminal_readout(data)
    assert 'bounds unavailable' in text and '0.00%' not in text
