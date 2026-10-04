from pathlib import Path
from hashlib import file_digest
import json, shutil, sys
import cv2
import numpy as np

root=Path(__file__).resolve().parents[2]
out=root/'evidence/annotation-coordinates-20261004'
out.mkdir(parents=True,exist_ok=False)
archive=root/'packages/facemath-0.19.0-core-20261004.zip'
with archive.open('rb') as stream: digest=file_digest(stream,'sha256').hexdigest()
assert digest=='387acf24ae6714b610a83075be4746444cb8b4bb4ba4264c3d5a55e0398416c4'
release=root/'runs/core-features-20261004/release-01'
parent=release/'extracted/facemath-0.19.0'
sys.path.insert(0,str(root/'scripts'))
from verify_release import verify
check=verify(parent)
sys.path.insert(0,str(parent/'src'))
from facemath.observation.annotation_ui import create_annotation_workspace
from facemath.observation.journal import code_fingerprint
assert code_fingerprint()=='6ec1ff8b6b1df885899542bf829ddc30938b77b2335a1349ea4cde7016d26dc3'
records=json.loads((release/'commands.json').read_text())
baseline=next(r for r in records if r['name']=='archive-tests')
assert baseline['returncode']==0
retained=out/'parent';retained.mkdir()
for name in ('archive-tests.log','archive.xml'):shutil.copyfile(release/name,retained/name)
record={'archive':str(archive),'sha256':digest,'extracted':str(parent),'byte_check':check,
 'baseline':baseline,'baseline_reused':True,
 'reason':'The immediately preceding release executed the inherited full suite from these unchanged extracted bytes. Preserve that actual run rather than repeat it before any source edit.'}
(retained/'parent.json').write_text(json.dumps(record,indent=2))
y,x=np.indices((32,48));image=np.stack(((x*17)%256,(y*23)%256,((x+y)*31)%256),axis=-1).astype(np.uint8)
cv2.imwrite(str(out/'indexed-raster.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
create_annotation_workspace(image,out/'parent/workbench.html',source='Indexed 48x32 raster; automated coordinate fixture',kind='synthetic_fixture')
print(json.dumps(record,indent=2))
