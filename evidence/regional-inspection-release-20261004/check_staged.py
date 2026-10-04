import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=root).decode().split('\0')
selected = []
for entry in entries:
    if not entry:
        continue
    meta, name = entry.split('\t', 1)
    if name.startswith(('evidence/regional-inspection-20261004/', 'evidence/regional-inspection-release-20261004/')) or name == 'REGIONAL_INSPECTION.json':
        selected.append((meta.split()[1], name))
proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for oid, name in selected:
    proc.stdin.write((oid+'\n').encode()); proc.stdin.flush()
    header = proc.stdout.readline().decode().split()
    assert header[1] == 'blob', (name, header)
    data = proc.stdout.read(int(header[2])); assert proc.stdout.read(1) == b'\n'
    assert data == (root/name).read_bytes(), name
proc.stdin.close(); assert proc.wait() == 0
record = json.loads((root/'REGIONAL_INSPECTION.json').read_text())
archive = record['archive']['path']
pointer = subprocess.check_output(['git', 'show', ':'+archive], cwd=root).decode()
with (root/archive).open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
assert 'oid sha256:'+digest in pointer
assert 'size '+str((root/archive).stat().st_size) in pointer
assert digest == record['archive']['sha256']
result = {'verified': True, 'staged_evidence_files_equal_worktree': len(selected),
          'archive_lfs_pointer_sha256': digest, 'archive_bytes': (root/archive).stat().st_size,
          'scope': 'All staged new regional evidence and release record; this check record is created afterward and not counted.'}
destination = root/'evidence/regional-inspection-release-20261004/staged-byte-check.json'
with destination.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
