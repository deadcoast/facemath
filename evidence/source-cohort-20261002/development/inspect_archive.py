"""Read and hash every member; write only original annotation XML into a new directory."""
from hashlib import sha256, file_digest
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
source = root/'ibug_300W_large_face_landmark_dataset.tar.gz'
receipt = json.loads((root/'source-download.json').read_text())
assert source.stat().st_size == receipt['bytes'] == int(receipt['headers']['Content-Length'])
with source.open('rb') as stream:
    assert file_digest(stream, 'sha256').hexdigest() == receipt['sha256']
metadata = root/'metadata'; metadata.mkdir(exist_ok=False)
records = []; total = 0
with tarfile.open(source, 'r|gz') as archive:
    for member in archive:
        if len(records) >= 20000 or member.size > 128*1024**2:
            raise ValueError('source exceeds prospective inventory limits')
        if not member.isfile() and not member.isdir():
            raise ValueError('unexpected link or special source member: '+member.name)
        record = dict(name=member.name, bytes=member.size, type='file' if member.isfile() else 'directory')
        if member.isfile():
            total += member.size
            if total > 8*1024**3:
                raise ValueError('source exceeds expanded-byte limit')
            stream = archive.extractfile(member)
            digest = sha256(); blocks = [] if member.name.endswith('.xml') else None
            while block := stream.read(1024**2):
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
            record['sha256'] = digest.hexdigest()
            if blocks is not None:
                name = member.name.split('/')[-1]
                if not name or any(c in name for c in '\\:\x00'):
                    raise ValueError('unsafe annotation basename')
                with (metadata/name).open('xb') as output:
                    output.write(b''.join(blocks))
        records.append(record)
result = dict(archive_sha256=receipt['sha256'], members=records, expanded_bytes=total,
    inference_executed=False, selection_performed=False)
with (root/'source-inventory.json').open('x', encoding='utf-8') as output:
    json.dump(result, output, indent=2)
print(json.dumps(dict(members=len(records), expanded_bytes=total,
    annotations=[r for r in records if r['name'].endswith('.xml')],
    first_members=records[:12]), indent=2))
