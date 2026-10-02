import json
from pathlib import Path
import shutil
from facemath.observation.study import file_inventory
from facemath.observation.transfer_io import verify_transfer_bundle
run=Path(__file__).resolve().parent
source=run/'release/source-native/evaluation'
target=run/'portable-windows-evaluation'
shutil.copytree(source,target)
assert file_inventory(source)==file_inventory(target)
result=dict(source=str(source),transfer=str(target),byte_identical=True,files=len(file_inventory(target)),
            verification=verify_transfer_bundle(target),
            reason='Original secure Windows temporary-directory owner ACL was not readable to WSL user; distinct copy inherits transfer-directory permissions, original unchanged')
with (run/'portable-copy.result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
