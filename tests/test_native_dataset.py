from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from facemath.observation.contracts import ObservationError, array_digest, object_digest
from facemath.observation.dataset import fetch_dataset, prepare_dataset, verify_dataset, run_pipeline
from facemath.observation.reference import IBUG68_INDICES, ReferenceAnnotation
from facemath.observation.study import read_json, save_rgb, load_rgb, freeze_study, file_inventory


@pytest.fixture
def recipe(tmp_path):
    root = tmp_path / 'sources'; root.mkdir()
    images, assets = [], []
    xml = ET.Element('dataset'); inventory = ET.SubElement(xml, 'images')
    xy = np.array([[30,30],[40,30],[55,30],[65,30],[47,44],[36,58],[60,58],[48,72]])
    for i, split in enumerate(('development','development','calibration','evaluation')):
        name = f'{i}.png'
        image = np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3).copy()
        image[0,0,0] = i
        save_rgb(root/name, image)
        images.append(dict(path=name,group_id=str(i),lineage_ids=[str(i)],split=split))
        entry = ET.SubElement(inventory,'image',file=name)
        box = ET.SubElement(entry,'box',left='20',top='20',width='55',height='60')
        for k in range(68):
            point = xy[IBUG68_INDICES.index(k)] if k in IBUG68_INDICES else (45,45)
            ET.SubElement(box,'part',name=f'{k:02}',x=str(point[0]),y=str(point[1]))
    (root/'labels.xml').write_bytes(ET.tostring(xml))
    (root/'model.task').write_bytes(b'not a real model; preparation must not invoke it')
    for path in sorted(root.iterdir()):
        assets.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest(),
                           url='https://example.invalid/'+path.name))
    spec = dict(format='facemath.dlib_dataset_recipe.v11',name='unit fixture',source='unit-test-fixture',
        label_source='machine-reference adapter fixture, not inference',image_terms='generated test pixels',
        grouping_limit='fixture source groups',runtime_version='1.0.1',label_kind='model_generated',
        coordinate_origin=0,box_margin=.25,annotations=['labels.xml'],model_asset='model.task',assets=assets,images=images,
        protocol=dict(domain='test only',minimum_raters=1,maximum_cases_per_group=1,minimum_development_groups=2,
                      alpha=.1,bias_mode='development_joint_ocular',strata=['all']))
    path=root/'recipe.json'; path.write_text(json.dumps(spec))
    return path


def edit(recipe, fn):
    value=read_json(recipe);fn(value);recipe.write_text(json.dumps(value))


def update_xml(recipe, fn):
    path=recipe.parent/'labels.xml';tree=ET.parse(path);fn(tree.getroot());tree.write(path)
    def pin(value):
        asset=next(a for a in value['assets'] if a['path']=='labels.xml')
        asset.update(bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest())
    edit(recipe,pin)


def test_crop_translation_and_portable_reverification(recipe,tmp_path):
    root=prepare_dataset(recipe,tmp_path/'dataset')
    assert verify_dataset(root)['cases']==4
    ref=ReferenceAnnotation.from_dict(read_json(root/'inputs/case_00000.reference.json'))
    assert ref.kind=='model_generated' and ref.prediction_exposure=='unknown'
    np.testing.assert_array_equal(ref.xy[0],[24,25])
    np.testing.assert_array_equal(load_rgb(root/'inputs/case_00000.png'),load_rgb(recipe.parent/'0.png')[5:95,6:89])
    moved=tmp_path/'moved';shutil.copytree(root,moved)
    assert verify_dataset(moved)['conversion_recomputed']
    assert read_json(moved/'study.spec.json')['cases'][0]['split']=='development'
    with pytest.raises(FileExistsError):prepare_dataset(recipe,root)


@pytest.mark.parametrize('change,match',[
    (lambda d:d.update(coordinate_origin=True),'origin'),
    (lambda d:d.update(label_kind='human_blind'),'model_generated'),
    (lambda d:d['assets'][0].update(path='../escape'),'basename'),
    (lambda d:d['assets'][0].update(path='C:escape'),'basename'),
    (lambda d:d['assets'][0].update(path='CON.txt'),'basename'),
    (lambda d:d['assets'][0].update(path='recipe.json'),'unique'),
    (lambda d:d['assets'][0].update(bytes=True),'size'),
    (lambda d:d['assets'][0].update(sha256='0'*64),'pin mismatch'),
    (lambda d:d['assets'][0].update(url='file:///local'),'HTTPS'),
    (lambda d:d['images'][2].update(group_id='0'),'leakage'),
    (lambda d:d['images'][2].update(lineage_ids=['0']),'leakage'),
    (lambda d:d['images'].pop(),'enrollment'),
    (lambda d:d['protocol'].update(admission_policy={}), 'empirical admission'),
])
def test_preparation_rejects_invalid_pins_protocols_and_leakage(recipe,tmp_path,change,match):
    edit(recipe,change)
    with pytest.raises(ObservationError,match=match):prepare_dataset(recipe,tmp_path/'dataset')
    assert not (tmp_path/'dataset').exists()


def test_duplicate_parts_cannot_silently_replace_labels(recipe,tmp_path):
    def duplicate(xml):
        box=xml.find('./images/image/box');box.append(deepcopy(box.find('part')))
    update_xml(recipe,duplicate)
    with pytest.raises(ObservationError,match='68 unique'):prepare_dataset(recipe,tmp_path/'out')


def test_out_of_crop_anchors_become_explicit_prospective_exclusion(recipe,tmp_path):
    update_xml(recipe,lambda xml:xml.find('./images/image/box/part[@name="36"]').set('x','-10'))
    root=prepare_dataset(recipe,tmp_path/'dataset')
    spec=read_json(root/'study.spec.json')
    assert spec['cases'][0]['eligibility']=='excluded'
    assert spec['cases'][0]['exclusion_reason'].startswith('prospective_reference_support')
    assert verify_dataset(root)['eligible']==3


def test_recomputed_conversion_rejects_resealed_fabricated_coordinates(recipe,tmp_path):
    root=prepare_dataset(recipe,tmp_path/'dataset')
    path=root/'inputs/case_00000.reference.json'
    value=read_json(path);value['xy'][4][0]+=1;path.write_text(json.dumps(value))
    manifest=read_json(root/'manifest.json');manifest['files']=file_inventory(root,exclude=('manifest.json',))
    manifest.pop('content_sha256');manifest['content_sha256']=object_digest(manifest)
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ObservationError,match='source coordinates'):verify_dataset(root)


def test_machine_labels_cannot_acquire_human_reference_admission(recipe,tmp_path):
    root=prepare_dataset(recipe,tmp_path/'dataset')
    spec=read_json(root/'study.spec.json');spec['admission_policy']={'attempt':'spoof'}
    path=root/'study.spec.json';path.write_text(json.dumps(spec))
    # Even changing the exposure declaration does not turn a teacher into a human.
    for item in spec['cases']:
        p=root/item['references'][0];ref=read_json(p);ref['prediction_exposure']='none_declared';p.write_text(json.dumps(ref))
    with pytest.raises(ObservationError,match='independent'):freeze_study(path,tmp_path/'study')


def test_failed_pipeline_keeps_copied_sources_actual_journal_and_no_success_manifest(recipe,tmp_path):
    root=prepare_dataset(recipe,tmp_path/'dataset')
    # Identical crop pixels in this fixture prevent freezing before the fake model is reached.
    with pytest.raises(ObservationError) as caught:run_pipeline(root,tmp_path/'run')
    failed=Path(caught.value.failure_path)
    assert read_json(failed/'failure.json')['completed'] is False
    assert (failed/'dataset/assets/labels.xml').read_bytes()==(recipe.parent/'labels.xml').read_bytes()
    assert not (failed/'manifest.json').exists()
    events=[json.loads(s) for s in (failed/'failure.events.jsonl').read_text().splitlines()]
    assert events[-1]['operation']=='execution.failed'


def test_download_verifies_bytes_and_retains_failed_artifact(recipe,tmp_path,monkeypatch):
    class Response:
        url='https://example.invalid/source'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,limit):return b'incorrect upstream bytes'
    monkeypatch.setattr('facemath.observation.dataset.urlopen',lambda *a,**kw:Response())
    with pytest.raises(ObservationError,match='pin mismatch'):fetch_dataset(recipe,tmp_path/'fetch')
    failure=read_json(tmp_path/'fetch/fetch.failure.json')
    assert not failure['completed'] and failure['verified_downloads']==[]
    assert (tmp_path/'fetch/0.png').read_bytes()==b'incorrect upstream bytes'


def test_explicit_one_based_input_has_identical_stored_raster_results(recipe,tmp_path):
    zero=prepare_dataset(recipe,tmp_path/'zero')
    def shift(xml):
        for box in xml.findall('./images/image/box'):
            for key in ('left','top'):box.set(key,str(int(box.attrib[key])+1))
            for part in box.findall('part'):
                for key in ('x','y'):part.set(key,str(int(part.attrib[key])+1))
    update_xml(recipe,shift)
    edit(recipe,lambda d:d.update(coordinate_origin=1))
    one=prepare_dataset(recipe,tmp_path/'one')
    for i in range(4):
        np.testing.assert_array_equal(load_rgb(zero/f'inputs/case_{i:05d}.png'),load_rgb(one/f'inputs/case_{i:05d}.png'))
        assert read_json(zero/f'inputs/case_{i:05d}.reference.json')['xy']==read_json(one/f'inputs/case_{i:05d}.reference.json')['xy']
