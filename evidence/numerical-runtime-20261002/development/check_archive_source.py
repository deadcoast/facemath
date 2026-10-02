import json
from pathlib import Path
import sys
import zipfile

run=Path(__file__).resolve().parent
root=run.parents[1]
sys.path.insert(0,str(root/'scripts'))
from package_release import inventory

archive=run/'release-check/facemath-0.11.2.zip'
source=inventory(root,[])
with zipfile.ZipFile(archive) as z:
    for name,p in source.items():
        assert z.read('facemath-0.11.2/'+name)==p.read_bytes(),name
with (run/'archive-source-parity.result.json').open('x') as f:
    json.dump(dict(verified=True,files=len(source),paths=list(source),
        scope='all current source/tests/scripts/docs/examples/workflows and selected root files equal sealed archive bytes'),f,indent=2)
print(f'{len(source)} current source, test, script, documentation and configuration files equal the sealed archive')
