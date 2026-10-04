"""Check the logical local release and exact LFS archive pointer before commit."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

run = Path(__file__).resolve().parent
root = run.parents[1]

def git(*args, input=None):
    return subprocess.check_output(['git',*args],cwd=root,input=input)

parent = '66a87e5d0fbf1f0aafcb77ad8a853c82229c0d02'
assert git('rev-parse','HEAD').decode().strip() == parent
archive = 'packages/facemath-0.16.0-enrollment-registry-20261003.zip'
pin = git('show', ':'+archive).decode()
expected = json.loads((run/'release-audit.result.json').read_text())
assert pin == ('version https://git-lfs.github.com/spec/v1\n'
               f'oid sha256:{expected["archive_sha256"]}\nsize {expected["archive_bytes"]}\n')
allowed = {'.gitattributes','CHANGELOG.md','README.md','docs/NATIVE_PIPELINE_OPERATIONS.md',
    'docs/WAVE10_ROADMAP.md','docs/ENROLLMENT_REGISTRY_DESIGN.md','pyproject.toml','uv.lock',
    'src/facemath/__init__.py','src/facemath/cli.py','src/facemath/observation/dataset.py',
    'src/facemath/observation/deployment.py','src/facemath/observation/enrollment.py',
    'src/facemath/observation/study.py','src/facemath/observation/transfer.py','src/facemath/observation/transfer_io.py',
    'scripts/check_wheel.py','scripts/validate_current.py','tests/test_enrollment_registry.py',
    'examples/helen-registry-window.json','CONTINUATION_REVIEW.json',archive,
    archive.replace('.zip','.release.json'),archive.replace('.zip','.validation.json')}
names = [s.decode() for s in git('diff','--cached','--name-only','-z').split(b'\0') if s]
assert names and all(n in allowed or n.startswith('evidence/enrollment-registry-20261003/') for n in names)
selected = set(names); objects = {}
for row in git('ls-files','-s','-z').split(b'\0'):
    if not row: continue
    info,name = row.split(b'\t',1)
    if name.decode() not in selected: continue
    mode,oid,stage = info.split()
    assert stage == b'0' and mode == b'100644'
    objects[oid.decode()] = name.decode()
sizes = git('cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)',
            input=('\n'.join(objects)+'\n').encode()).decode().splitlines()
maximum = max(int(row.split()[2]) for row in sizes)
assert maximum < 100 * 1024**2
record = dict(parent=parent,staged_files=len(names),unique_blobs=len(objects),largest_regular_blob_bytes=maximum,
    archive_lfs_pointer=pin,archive_sha256=sha256((root/archive).read_bytes()).hexdigest(),
    only_planned_release_files=True,remote_publication_performed=False)
with (run/'staged-proof.json').open('x') as stream: json.dump(record,stream,indent=2)
print(json.dumps(record,indent=2))
