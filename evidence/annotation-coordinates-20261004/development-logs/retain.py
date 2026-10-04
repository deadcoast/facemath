from pathlib import Path
import json,shutil
import cv2
from facemath.observation.annotation_ui import create_annotation_workspace
from facemath.observation.journal import code_fingerprint
from facemath.observation.study import read_json,write_json_new

root=Path(__file__).resolve().parents[2];run=Path(__file__).parent
out=root/'evidence/annotation-coordinates-20261004'
image=cv2.cvtColor(cv2.imread(str(out/'indexed-raster.png')),cv2.COLOR_BGR2RGB)
create_annotation_workspace(image,out/'workbench.html',source='Indexed 48x32 raster; automated coordinate fixture',kind='synthetic_fixture')
assert (out/'workbench.html').read_bytes()==(out/'corrected-02/workbench.html').read_bytes()
runtime=out/'runtime';runtime.mkdir()
prior=root/'evidence/core-features-20261004/intake-01/dataset'
for source,target in [(prior/'assets/face_landmarker.task',runtime/'face_landmarker.task'),(prior/'inputs/case_00000.png',runtime/'portrait.png')]:
    shutil.copyfile(source,target)
image=cv2.cvtColor(cv2.imread(str(runtime/'portrait.png')),cv2.COLOR_BGR2RGB)
create_annotation_workspace(image,out/'portrait-workbench.html',source='Retained HELEN evaluation portrait, already enrolled; blank reference input, not a new study')
pair=read_json(out/'pair-regression-02/comparison.json')
old=read_json(root/'evidence/core-features-20261004/pair-01/comparison.json')
assert pair['raw_array_sha256']==old['raw_array_sha256']
write_json_new(out/'pair-parity.json',{'code_sha256':code_fingerprint(),'raw_arrays_equal_to_parent':len(pair['raw_array_sha256']),
    'new_native_inference':False,'fit_performed':False,'scope':'regression using retained native observations; no new cohort',
    'comparison':'pair-regression-02/comparison.json','encoding':'cp1252:strict','raw_array_sha256':pair['raw_array_sha256']})
logs=out/'development-logs';logs.mkdir()
for source in run.iterdir():
    if source.is_file() and source.suffix in ('.py','.ps1','.log','.json'):
        shutil.copyfile(source,logs/source.name)
write_json_new(out/'failure-history.json',{
  'parent_coordinate_failure':'parent/coordinate-failure.json',
  'parent_workspace_kind_failure':'workspace-parent/failure.json',
  'browser_01':'Wheel anchor control originally used a fractional requested pointer position; corrected to the actual integer WheelEvent location, keeping the same 0.002-pixel assertion.',
  'browser_02':'Storage fault injection returned a function to Playwright, which invoked it. Wrapped installation in an explicit function; no application change.',
  'browser_03_and_04':'Narrow edge-center clicks rejected browser float rounding before quantization; raw values retained in edge-rounding-failure.json. Fixed production capture to quantize before domain check.',
  'pair_regression_01':'Windows CP1252 UnicodeEncodeError in overview, retained in pair-regression-01/terminal.log. Fixed production stream-aware text rendering.',
  'setup_edit_helper':'Initial edit helper used Windows default text decoding and stopped before writing. Corrected helper to explicit UTF-8.',
  'final_browser':'browser-05/browser.json', 'prior_artifacts_overwritten':False})
print(json.dumps({'code_sha256':code_fingerprint(),'native_raw_arrays_unchanged':158,'browser_variants':6},indent=2))
