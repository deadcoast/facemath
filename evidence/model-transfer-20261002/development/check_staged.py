"""Check byte-preserved evidence, LFS pointers and staged blob sizes."""
from hashlib import sha1, sha256
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
def git(*args, data=None):
    return subprocess.run(['git',*args],cwd=root,input=data,capture_output=True,check=True).stdout
assert git('rev-parse','--show-object-format').strip()==b'sha1'
changed=set(git('diff','--cached','--name-only','-z').decode().split('\0'))-{''}
index={}
for entry in git('ls-files','--stage','-z').decode().split('\0'):
    if entry:
        meta,name=entry.split('\t',1)
        mode,oid,stage=meta.split()
        if name in changed:
            assert stage=='0'
            index[name]=oid
oids=sorted(set(index.values()))
records=git('cat-file','--batch-check',data=('\n'.join(oids)+'\n').encode()).decode().splitlines()
maximum=max(int(line.split()[2]) for line in records)
assert maximum<100*1024*1024
checked=0
for name,oid in index.items():
    if name.startswith(('src/facemath/','evidence/model-transfer-20261002/')):
        data=(root/name).read_bytes()
        assert sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==oid,name
        checked+=1
archive='packages/facemath-0.14.0-transfer-20261002.zip'
pointer=git('show',':'+archive).decode()
digest=sha256((root/archive).read_bytes()).hexdigest()
assert 'version https://git-lfs.github.com/spec/v1\n' in pointer
assert f'oid sha256:{digest}\n' in pointer
assert f'size {(root/archive).stat().st_size}\n' in pointer
result=dict(staged_files=len(index),byte_preserved_source_and_evidence_files=checked,
            maximum_regular_blob_bytes=maximum,archive_sha256=digest,lfs_pointer=pointer)
with (Path(__file__).resolve().parent/'staged-proof.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))

