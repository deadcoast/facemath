from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import sys

run = Path(__file__).resolve().parent
root = run.parents[1]
sys.path.insert(0, str(root/'scripts'))
from verify_release import verify

parent = root/'packages/facemath-0.13.0-utility-20261002.zip'
extracted = root/'runs/bound-utility-20261002/release/extracted/facemath-0.13.0'
commands = json.loads((root/'runs/bound-utility-20261002/release/commands.json').read_text())
baseline = next(c for c in commands if c['name'] == 'archive-tests')
assert baseline['returncode'] == 0
log = root/'runs/bound-utility-20261002/release/archive-tests.log'
assert '1045 passed, 1 skipped' in log.read_text()
with parent.open('rb') as stream: digest = file_digest(stream, 'sha256').hexdigest()
assert digest == 'd8ac19a877f0a3d74a8d4ab0255ea1fa2c61a7c24fbf02e95176ef236a752a57'
record = dict(checked_utc=datetime.now(timezone.utc).isoformat(), parent=str(parent),
    sha256=digest, bytes=parent.stat().st_size, extracted=verify(extracted),
    baseline=baseline, baseline_log=str(log),
    reuse='Actual extracted-parent test execution retained from prior turn, completed before any 0.14 source edit; not rerun',
    result='1045 passed, 1 skipped; Windows directory symlink unavailable')
(run/'parent-baseline.json').write_text(json.dumps(record, indent=2))
(run/'parent-tests.log').write_bytes(log.read_bytes())
print(json.dumps(record))

def inventory(path):
    out = {}
    for p in path.rglob('*'):
        if p.is_file():
            with p.open('rb') as stream: h = file_digest(stream, 'sha256').hexdigest()
            out[p.relative_to(path).as_posix()] = [p.stat().st_size, h]
    return out

pairs = [
    ('runs/bound-utility-20261002/release/source-native', 'evidence/bound-utility-20261002/preseal/native-pipeline'),
    ('runs/bound-utility-20261002/release/wheel-check/native-pipeline', 'evidence/bound-utility-20261002/postseal/installed-native'),
    ('runs/bound-utility-20261002/release/archive-native', 'evidence/bound-utility-20261002/postseal/archive-native'),
    ('runs/bound-utility-20261002/linux-native', 'evidence/bound-utility-20261002/development/linux-native'),
]
checked = []
for source, retained in pairs:
    a, b = inventory(root/source), inventory(root/retained)
    assert a and a == b, (source, retained)
    checked.append(dict(source=source, retained=retained, files=len(a), bytes=sum(v[0] for v in a.values()), exact=True))
(run/'duplicate-check.json').write_text(json.dumps(checked, indent=2))
print(json.dumps(checked))
