from hashlib import file_digest
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('scripts').resolve()))
from verify_release import verify

def digest(path):
    with path.open('rb') as f:return file_digest(f,'sha256').hexdigest()

records=[]
for run,version,kind in [('native-pipeline-20261002','0.11.0','native'),('native-portability-20261002','0.11.1','portability')]:
    base=Path('runs')/run/'release-check'
    receipt=json.loads((base/'package-bytes.json').read_text())
    canonical=Path(f'packages/facemath-{version}-{kind}-20261002.zip')
    expected=receipt['archive']['archive_sha256']
    assert digest(canonical)==expected
    stage=base/'stage'
    result=verify(stage)
    assert result['verified'] and result['files']==receipt['archive']['files']
    duplicate=base/f'facemath-{version}.zip'
    assert digest(duplicate)==expected
    records.append(dict(base=str(base.resolve()),targets=[str(stage.resolve()),str(duplicate.resolve())],
        retained_archive=str(canonical),sha256=expected,stage_verified=result))
with Path('runs/bound-utility-20261002/old-copy-proof.json').open('x') as f:json.dump(records,f,indent=2)
print(json.dumps(records,indent=2))
