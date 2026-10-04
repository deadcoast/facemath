"""Retain the rejected first build and bypass control as distinct diagnostics."""
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import zipfile
from facemath.observation.study import write_json_new, read_json

run = Path(__file__).resolve().parent
root = run.parents[1]
target = root/'evidence/enrollment-registry-20261003/superseded-01'
target.mkdir(exist_ok=False)
archive = run/'release/facemath-0.16.0.zip'
with archive.open('rb') as stream: digest = file_digest(stream,'sha256').hexdigest()
assert digest == '3143bb38e006780e89e569e04a90f3bfcd39bf43e6bdf7778f5c1df24203c274'
copied = []
with zipfile.ZipFile(archive) as source:
    for name in source.namelist():
        relative = name.split('/',1)[1]
        if relative.startswith('src/facemath/') and relative.endswith('.py'):
            dest = target/'source'/relative
            assert dest.resolve().is_relative_to((target/'source').resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as stream: stream.write(source.read(name))
            copied.append(relative)
assert len(copied) == 87
for name in ('null-control','null-bypass-lock'):
    shutil.copytree(run/name,target/name)
for name in ('null-registry.json','null-reproduction.json','null-reproduction.log','null-bypass.result.json'):
    shutil.copyfile(run/name,target/name)
for name in ('validation.json','commands.json','source.xml','archive.xml','package-bytes.json'):
    shutil.copyfile(run/'release'/name,target/name)
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), disposition='superseded; not the final release',
    reason='Explicit JSON null registry bypass demonstrated despite passing inherited/focused suites',
    source_code_sha256='dd1fbf7dd9d1d4cc5ee0ef10653f1a316a2d10bf70247000f29e1b74e38ba57d',
    archive_sha256=digest,archive_preserved_at=archive.relative_to(root).as_posix(),
    source_modules_retained=copied,original_validation_completed=read_json(run/'release/validation.json')['completed'],
    bypass=read_json(run/'null-bypass.result.json'),
    final_attempt='release-03; includes strict null and empty-path guards; all final tiers rerun')
write_json_new(target/'STATUS.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ('source_modules_retained','bypass')},indent=2))
