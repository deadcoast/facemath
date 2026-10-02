import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path('scripts').resolve()))
from verify_release import verify

rows=[]
for run,version in [('native-pipeline-20261002','0.11.0'),('native-portability-20261002','0.11.1')]:
    base=Path('runs')/run/'release-check'
    assert json.loads((base/'validation.json').read_text())['completed']
    target=base/'extracted'/f'facemath-{version}'
    result=verify(target)
    assert result['verified']
    rows.append(dict(base=str((base/'extracted').resolve()),target=str(target.resolve()),verification=result,
                     canonical_archive_check='old-copy-proof.json'))
with Path('runs/bound-utility-20261002/old-extraction-proof.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows,indent=2))
