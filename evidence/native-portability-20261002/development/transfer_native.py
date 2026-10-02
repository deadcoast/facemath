"""Serialize a completed native bundle without changing its files."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile

run=Path(__file__).resolve().parent
source=run/'release-check/source-native'
out=run/'windows-native-transfer.zip'
files=sorted(p for p in source.rglob('*') if p.is_file())
assert all(not p.is_symlink() for p in files)
with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(source).as_posix())
with (run/'windows-native-transfer.json').open('x') as stream:
    json.dump(dict(source=str(source),files=len(files),bytes=out.stat().st_size,
        sha256=sha256(out.read_bytes()).hexdigest(),
        purpose='transport original Windows bundle bytes to Linux-local storage without inherited filesystem ACLs'),stream,indent=2)
