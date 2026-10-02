from hashlib import sha256
from pathlib import Path
import json
import tarfile
import urllib.request

root = Path(__file__).resolve().parent
url = 'https://github.com/astral-sh/uv/releases/download/0.12.22/uv-x86_64-unknown-linux-gnu.tar.gz'
expected = 'b9980552309f09c15172b8be828555e375097f16deb459795ce7bfd200380f0b'
with urllib.request.urlopen(url, timeout=90) as response:
    data = response.read(19916279)
assert len(data) == 19916278 and sha256(data).hexdigest() == expected
archive = root / 'uv-linux.tar.gz'
with archive.open('xb') as stream:
    stream.write(data)
out = root / 'linux-tools'
out.mkdir()
with tarfile.open(archive) as tar:
    member = tar.getmember('uv-x86_64-unknown-linux-gnu/uv')
    assert member.isfile()
    with (out / 'uv').open('xb') as stream:
        stream.write(tar.extractfile(member).read())
(root / 'uv-receipt.json').write_text(json.dumps(dict(url=url, bytes=len(data), sha256=expected), indent=2))
print(expected)
