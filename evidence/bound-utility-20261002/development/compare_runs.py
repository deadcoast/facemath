"""Read-only comparison of retained native observations; never alters a run."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

from facemath.observation.contracts import Observation
from facemath.observation.dataset import verify_pipeline

left, right, output = map(Path, sys.argv[1:])
verification = [verify_pipeline(p) for p in (left, right)]
assert (left/'dataset/study.spec.json').read_bytes() == (right/'dataset/study.spec.json').read_bytes()
rows = []
for case in sorted((left/'session/cases').iterdir()):
    other = right/'session/cases'/case.name
    a = json.loads((case/'receipt.json').read_text())
    b = json.loads((other/'receipt.json').read_text())
    row = dict(case=case.name, statuses=[a['status'],b['status']], same_status=a['status']==b['status'])
    if a['status'] == b['status'] == 'observed':
        raw = [(p/'observation.json').read_bytes() for p in (case,other)]
        obs = [Observation.from_dict(json.loads(d)) for d in raw]
        row.update(observation_sha256=[sha256(d).hexdigest() for d in raw],
                   observation_bytes_equal=raw[0]==raw[1], arrays={})
        for name in ('xy','samples'):
            x,y=(getattr(o,name) for o in obs)
            same_layout=x.shape==y.shape and x.dtype==y.dtype
            row['arrays'][name] = dict(same_layout=same_layout,
                bytes_equal=same_layout and x.tobytes()==y.tobytes(),
                maximum_absolute_delta_px=float(np.max(np.abs(x-y))) if same_layout else None)
    rows.append(row)
result = dict(format='facemath.native_run_comparison.v1', runs=[str(left),str(right)],
    verification=verification, same_frozen_spec_bytes=True, cases=rows,
    identical_observations=sum(r.get('observation_bytes_equal',False) for r in rows),
    total=len(rows), limits=['observed repeatability only; not universal determinism or anatomical accuracy',
                            'different run journals and environments remain distinct'])
with output.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ('cases','verification')},indent=2))
