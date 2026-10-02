"""Read-only raw observation invariance audit across acquisition and qualification."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
from facemath.observation.contracts import Observation

root,output=map(Path,sys.argv[1:])
rows=[]
for case in sorted((root/'session/cases').iterdir()):
    receipt=json.loads((case/'receipt.json').read_text())
    if receipt['status']!='observed':
        rows.append(dict(case=case.name,status=receipt['status'],raw_arrays=0));continue
    paths=[case/'observation.json',root/'qualification/acquisition/cases'/case.name/'observation.json',root/'replay/acquisition/cases'/case.name/'observation.json']
    raw=[p.read_bytes() for p in paths];assert raw[0]==raw[1]==raw[2]
    observations=[Observation.from_dict(json.loads(data)) for data in raw]
    for field in ('xy','samples'):
        arrays=[getattr(o,field) for o in observations]
        assert all(a.shape==arrays[0].shape and a.dtype==arrays[0].dtype and a.tobytes()==arrays[0].tobytes() for a in arrays[1:])
    rows.append(dict(case=case.name,status='observed',observation_sha256=sha256(raw[0]).hexdigest(),raw_arrays=2))
result=dict(pipeline=str(root),cases=rows,raw_arrays_checked=sum(r['raw_arrays'] for r in rows),observed=sum(r['status']=='observed' for r in rows),raw_observation_bytes_unchanged=True,scope='actual acquisition, saved qualification and exact replay copies; no fitted centers substituted for observations')
with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
