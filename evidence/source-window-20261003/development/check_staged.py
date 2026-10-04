"""Validate one logical local release commit and its LFS archive pointer."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

run = Path(__file__).resolve().parent
root = run.parents[1]

def git(*args, input=None):
    return subprocess.check_output(['git',*args], cwd=root, input=input)

assert git('rev-parse','HEAD').decode().strip() == '381e15af01427ae78cf01717d9f1c2a7fe8067ed'
archive = 'packages/facemath-0.15.0-source-window-20261003.zip'
pin = git('show', ':'+archive).decode()
expected = json.loads((run/'release-audit.result.json').read_text())
assert pin == ('version https://git-lfs.github.com/spec/v1\n'
               f'oid sha256:{expected["archive_sha256"]}\nsize {expected["archive_bytes"]}\n')
allowed = {'.gitattributes','CHANGELOG.md','README.md','docs/NATIVE_PIPELINE_OPERATIONS.md',
    'docs/WAVE10_ROADMAP.md','docs/SOURCE_WINDOW_DESIGN.md','pyproject.toml','uv.lock',
    'src/facemath/__init__.py','src/facemath/observation/dataset.py',
    'src/facemath/observation/source_cohort.py','tests/test_source_cohort.py',
    'examples/helen-transfer-window.json','CONTINUATION_REVIEW.json',archive,
    archive.replace('.zip','.release.json'),archive.replace('.zip','.validation.json')}
names = [v.decode() for v in git('diff','--cached','--name-only','-z').split(b'\0') if v]
assert names and all(n in allowed or n.startswith('evidence/source-window-20261003/') for n in names)
selected = set(names)
objects = {}
for row in git('ls-files','-s','-z').split(b'\0'):
    if not row:
        continue
    info, name = row.split(b'\t',1)
    if name.decode() not in selected:
        continue
    mode, oid, stage = info.split()
    assert stage == b'0' and mode == b'100644'
    objects[oid.decode()] = name.decode()
sizes = git('cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)',
            input=('\n'.join(objects)+'\n').encode()).decode().splitlines()
maximum = max(int(row.split()[2]) for row in sizes)
assert maximum < 100 * 1024**2
record = dict(parent='381e15af01427ae78cf01717d9f1c2a7fe8067ed', staged_files=len(names),
    unique_blobs=len(objects), largest_regular_blob_bytes=maximum, archive_lfs_pointer=pin,
    archive_sha256=sha256((root/archive).read_bytes()).hexdigest(),
    only_planned_release_files=True, remote_publication_performed=False)
with (run/'staged-proof.json').open('x') as f:
    json.dump(record,f,indent=2)
print(json.dumps(record,indent=2))
