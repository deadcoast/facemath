import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import threading
from urllib.request import Request, urlopen
import zipfile

import cv2
import numpy as np
from PIL import Image
import pytest

from facemath.observation.contracts import array_digest
from facemath.workspace import Workspace, WorkspaceServer
from facemath.workspace_input import select_input, source_preview, png, selection_page
from test_workspace import Pending


def upload(pixels=None, crop=None):
    if pixels is None:
        yy, xx = np.indices((40, 60))
        pixels = np.stack([xx, yy, xx+yy], axis=-1).astype(np.uint8)
    return {'name': 'indexed.png', 'data': base64.b64encode(png(pixels)).decode(), 'crop': crop}


@pytest.fixture
def work(tmp_path):
    model = tmp_path/'model.task'; model.write_bytes(b'explicit API fixture, not native weights')
    instance = Workspace(tmp_path/'app', model)
    yield instance
    instance.close()


def test_crop_pixels_coordinates_and_hashes_are_exact():
    item = upload(crop=[7, 11, 39, 35])
    s = select_input(item)
    np.testing.assert_array_equal(s.selected, s.pixels[11:35, 7:39])
    encoded = png(s.selected)
    record = s.record('a', encoded)
    assert record['crop'] == [7, 11, 39, 35]
    assert record['analysis']['width'] == 32 and record['analysis']['height'] == 24
    np.testing.assert_array_equal(record['analysis']['to_original'], [[1,0,7],[0,1,11],[0,0,1]])
    for x, y in ((0,0),(31,23),(12,8)):
        original = np.array(record['analysis']['to_original']) @ [x,y,1]
        np.testing.assert_array_equal(s.selected[y,x], s.pixels[original[1],original[0]])
    assert record['sha256'] == sha256(s.original).hexdigest()
    assert record['analysis']['decoded_rgb_sha256'] == array_digest(cv2.cvtColor(s.selected, cv2.COLOR_BGR2RGB))
    decoded = cv2.imdecode(np.frombuffer(encoded,np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded,s.selected)


@pytest.mark.parametrize('crop',[None,[0,0,60,40]])
def test_full_image_keeps_original_upload_bytes(crop):
    item = upload(crop=crop); selected = select_input(item)
    assert not selected.cropped
    assert selected.original == base64.b64decode(item['data'])
    assert selected.record('b',selected.original)['analysis']['file'] == 'input_b.image'


@pytest.mark.parametrize('crop',[[0,0,61,40],[-1,0,40,40],[0,-1,40,40],[0,0,40,41],
    [0,0,7,8],[0,0,8,7],[20,10,10,30],[False,0,40,40],[0.,0,40,40],
    [0,0,40],{'left':0},'0,0,40,40'])
def test_invalid_crops_never_create_a_run(work, crop):
    with pytest.raises(ValueError):
        work.submit({'a':upload(crop=crop),'b':upload()})
    assert work.history() == []


def test_preview_and_selection_ignore_exif_display_rotation():
    rgb = np.zeros((30,60,3),np.uint8); rgb[:,:20,0] = 255
    exif = Image.Exif(); exif[274] = 6
    data = BytesIO(); Image.fromarray(rgb).save(data, format='JPEG', exif=exif)
    item = {'name':'rotated.jpg','data':base64.b64encode(data.getvalue()).decode(),'crop':[0,0,20,30]}
    preview = source_preview(item); selected = select_input(item)
    assert (preview['width'], preview['height']) == (60,30)
    displayed = cv2.imdecode(np.frombuffer(base64.b64decode(preview['preview']),np.uint8),cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(displayed,selected.pixels)
    assert selected.selected.shape == (30,20,3) and selected.selected[:,:18,2].mean() > 240


def test_preview_resize_does_not_resize_analysis_input():
    item = upload(np.zeros((1200,1800,3),np.uint8),[100,200,1100,1000])
    preview = source_preview(item); selected = select_input(item)
    displayed = cv2.imdecode(np.frombuffer(base64.b64decode(preview['preview']),np.uint8),cv2.IMREAD_COLOR)
    assert max(displayed.shape[:2]) == 1024
    assert selected.selected.shape == (800,1000,3)
    assert (preview['width'],preview['height']) == (1800,1200)


def test_native_command_receives_lossless_crop_and_originals_remain(work, monkeypatch):
    process = Pending(); calls=[]
    def launch(command, **kwargs):
        calls.append(command); return process
    monkeypatch.setattr(subprocess,'Popen',launch)
    a, b = upload(crop=[7,11,39,35]), upload()
    record = work.submit({'a':a,'b':b})
    root = Path(record['directory'])
    assert calls[0][4:6] == [str(root/'analysis_a.png'),str(root/'input_b.image')]
    assert (root/'input_a.image').read_bytes() == base64.b64decode(a['data'])
    np.testing.assert_array_equal(cv2.imread(str(root/'analysis_a.png')),select_input(a).selected)
    assert json.loads((root/'request.json').read_text())['inputs'] == record['inputs']
    html = (root/'selection.html').read_text(encoding='utf-8')
    assert 'columns [7, 39)' in html and 'rows [11, 35)' in html and 'x="6.5" y="10.5"' in html
    work.cancel(record['id']); work.thread.join(3)
    assert (root/'analysis_a.png').exists() and (root/'input_a.image').exists()


def test_success_download_retains_input_selection_without_editing_result(work, monkeypatch):
    from facemath.observation import artifacts
    process = Pending(); monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:process)
    # Explicit lifecycle fixture, never represented as native inference.
    monkeypatch.setattr(artifacts,'verify_bundle',lambda root:{'verified':True})
    record = work.submit({'a':upload(crop=[0,0,20,20]),'b':upload()})
    root = Path(record['directory']); (root/'result').mkdir()
    (root/'result/report.html').write_bytes(b'explicit publication fixture')
    process.returncode=0; process.event.set(); work.thread.join(3)
    assert work.status(record['id'])['status'] == 'complete'
    with zipfile.ZipFile(root/'comparison.zip') as z:
        for name in ('selection.html','preview_a.png','preview_b.png','analysis_a.png','input_a.image','input_b.image','request.json'):
            assert z.read(name) == (root/name).read_bytes()
        assert z.read('result/report.html') == b'explicit publication fixture'


def test_preview_endpoint_has_no_run_side_effect(work):
    server = WorkspaceServer(work); thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        request = Request(server.url+'api/preview',data=json.dumps({'image':upload()}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            preview = json.load(response)
        assert (preview['width'],preview['height']) == (60,40)
        assert work.history() == []
    finally:
        server.shutdown(); server.server_close(); thread.join(3)


def test_selection_html_escapes_source_names():
    s = select_input(upload()); record = s.record('a',s.original); record['name'] = '<img onerror="bad">'
    html = selection_page([record,record])
    assert '<img onerror=' not in html and '&lt;img onerror=' in html
