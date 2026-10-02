from pathlib import Path
from urllib.request import urlopen
from hashlib import sha256
from datetime import datetime, timezone
import json
import xml.etree.ElementTree as ET

root = Path(__file__).parent / 'upstream'
root.mkdir(exist_ok=False)
revision = '46fa4a281a35b16aed6577250d7dabbda0e74a2b'
base = f'https://raw.githubusercontent.com/davisking/dlib/{revision}/examples/faces/'
records = []
def fetch(url, name):
    with urlopen(url, timeout=90) as response:
        data = response.read()
        record = dict(url=url, final_url=response.url, status=response.status,
                      name=name, bytes=len(data), sha256=sha256(data).hexdigest(),
                      retrieved_utc=datetime.now(timezone.utc).isoformat())
    with (root/name).open('xb') as f: f.write(data)
    records.append(record)
    (root/'receipts.json').write_text(json.dumps(records, indent=2))
    print(name, len(data), record['sha256'], flush=True)
    return data

for name in ('training_with_face_landmarks.xml', 'testing_with_face_landmarks.xml'):
    xml = fetch(base+name, name)
    for element in ET.fromstring(xml).findall('./images/image'):
        image = element.attrib['file']
        if not (root/image).exists(): fetch(base+image, image)
fetch(f'https://raw.githubusercontent.com/davisking/dlib/{revision}/dlib/LICENSE.txt', 'dlib-LICENSE.txt')
fetch('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')
