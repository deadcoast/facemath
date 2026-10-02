"""Prepare an immutable Windows qualification snapshot for the Linux control."""
from hashlib import sha256
import json
from pathlib import Path
from facemath.observation.deployment import pack_snapshot
from facemath.observation.study_io import verify_qualification_bundle

run = Path(__file__).resolve().parent
source = run/'release-check/source-native/qualification'
verification = verify_qualification_bundle(source)
data = pack_snapshot(source)
target = run/'windows-qualification.zip'
with target.open('xb') as stream:
    stream.write(data)
record = dict(source=str(source), verification=verification, archive=str(target), bytes=len(data),
    sha256=sha256(data).hexdigest(), source_manifest_sha256=sha256((source/'manifest.json').read_bytes()).hexdigest())
with (run/'windows-transfer.json').open('x') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record, indent=2))
