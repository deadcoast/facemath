"""Extract the known transfer archive, verify bytes, and retain exact replay."""
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

run=Path(__file__).resolve().parent
archive=run/'windows-native-transfer.zip'
receipt=json.loads((run/'windows-native-transfer.json').read_text())
assert sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
target=Path('/tmp/facemath-windows-native-transfer-20261002')
target.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        path=PurePosixPath(info.filename)
        assert info.orig_filename==info.filename and not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename
    assert len(z.infolist())==receipt['files']
    z.extractall(target)
commands=[['verify',str(target)],['replay',str(target),'--out',str(target.parent/'facemath-windows-native-replayed-20261002')]]
results=[]
for args in commands:
    p=subprocess.run([sys.executable,'-m','facemath',*args],capture_output=True,text=True)
    results.append(dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    print(p.stdout[-2000:]);print(p.stderr[-2000:])
    if p.returncode:break
with (run/'cross-windows-to-linux-transferred.results.json').open('x') as f:json.dump(results,f,indent=2)
replayed=target.parent/'facemath-windows-native-replayed-20261002'
if replayed.exists():shutil.copytree(replayed,run/'cross-windows-to-linux-transferred')
sys.exit(results[-1]['returncode'])
