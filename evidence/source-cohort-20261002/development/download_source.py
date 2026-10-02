"""Retain the publicly published dlib source archive before selecting a cohort."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parent
url = 'https://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz'
target = root/'ibug_300W_large_face_landmark_dataset.tar.gz'
limit = 2 * 1024**3
record = dict(url=url, source_index='https://dlib.net/files/data/',
    annotation_source='https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/',
    terms='Research-only annotations; no commercial products. Original photographs retain their owners rights.',
    started_utc=datetime.now(timezone.utc).isoformat(), inference_executed=False)
digest = sha256(); count = 0
try:
    with urlopen(url, timeout=90) as response, target.open('xb') as stream:
        record.update(final_url=response.url, headers=dict(response.headers))
        if int(response.headers.get('Content-Length', 0)) > limit:
            raise ValueError('archive exceeds the prospective 2 GiB intake cap')
        while block := response.read(1024*1024):
            if count + len(block) > limit:
                raise ValueError('archive exceeds the prospective 2 GiB intake cap')
            stream.write(block); digest.update(block); count += len(block)
            if count % (128*1024**2) == 0:
                print(count, flush=True)
    record.update(completed=True, bytes=count, sha256=digest.hexdigest())
except Exception as exc:
    record.update(completed=False, received_bytes=count, received_sha256=digest.hexdigest(),
                  exception=type(exc).__name__, detail=str(exc))
    raise
finally:
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with (root/'source-download.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))
