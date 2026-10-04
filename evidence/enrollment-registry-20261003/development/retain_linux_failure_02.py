"""Preserve the actual intermediate Linux fixture interrupted by source drift."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

root = Path('/mnt/c/Users/ryanf/github/_active/facemath')
source = Path('/tmp/facemath-enrollment-registry-20261003-02/tests/wave100/lab')
target = root/'evidence/enrollment-registry-20261003/superseded-02/linux-fixture'
assert source.is_dir() and target.resolve().is_relative_to(root.resolve())
shutil.copytree(source,target)
record = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(source), target=str(target),
    outcome='1139 passed, 17 fixture-setup errors; not a passing qualification',
    error='study code differs; freeze a new study rather than mutating its protocol',
    cause='CLI empty-path guard was completed while this intermediate suite was running; source fingerprint changed',
    retained='actual positive and partial negative fixture study/acquisition artifacts, without repair',
    resolution='fresh final source freeze, new release-03 and isolated Linux-03 environment; originals unchanged')
with (target.parent/'STATUS.json').open('x') as stream: json.dump(record,stream,indent=2)
print(json.dumps(record,indent=2))
