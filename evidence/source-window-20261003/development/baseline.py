from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
parent = root / 'packages/facemath-0.14.0-transfer-20261002.zip'
sys.path.insert(0, str(root / 'scripts'))
from verify_release import verify
with parent.open('rb') as f:
    digest = file_digest(f, 'sha256').hexdigest()
assert digest == '054ad6bf5bccd9d2b8dd1b688f42a5a476e014e1ff192851e99d1cac3b38cb9a'
receipt = {'parent': str(parent), 'sha256': digest, 'bytes': parent.stat().st_size,
           'archive_verification': verify(parent)}
with zipfile.ZipFile(parent) as z:
    z.extractall(out / 'parent')
extracted = out / 'parent/facemath-0.14.0'
receipt['extracted_verification'] = verify(extracted)
env = dict(os.environ, PYTHONPATH=str(extracted/'src'), OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
native = extracted/'evidence/model-transfer-20261002/dataset'
recipe = json.loads((native/'recipe.json').read_text())
spec = json.loads((native/'study.spec.json').read_text())
env['FACEMATH_MODEL'] = str(native/'assets'/recipe['model_asset'])
env['FACEMATH_FACE'] = str(native/next(c['image'] for c in spec['cases'] if c['eligibility']=='included'))
command = [sys.executable, '-m', 'pytest', '-q', '-ra', f'--junitxml={out / "parent.xml"}',
           f'--basetemp={out / "parent-temp"}']
receipt.update(command=command, cwd=str(extracted), started_utc=datetime.now(timezone.utc).isoformat(),
               environment={k:env[k] for k in ('PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','FACEMATH_MODEL','FACEMATH_FACE')})
(out/'parent-baseline-start.json').write_text(json.dumps(receipt, indent=2))
start = time.monotonic()
with (out/'parent-tests.log').open('x', encoding='utf-8') as log:
    proc = subprocess.run(command, cwd=extracted, env=env, stdout=log, stderr=subprocess.STDOUT)
receipt.update(returncode=proc.returncode, seconds=time.monotonic()-start,
               finished_utc=datetime.now(timezone.utc).isoformat())
(out/'parent-baseline.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
print((out/'parent-tests.log').read_text()[-4000:])
sys.exit(proc.returncode)
