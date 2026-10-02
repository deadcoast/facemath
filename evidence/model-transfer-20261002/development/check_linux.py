import json
from pathlib import Path
import numpy as np
from facemath.observation.contracts import Observation

run=Path(__file__).resolve().parent
left=run/'release/source-native';right=run/'linux-native'
rows=[]
for p in sorted((left/'session/cases').glob('*/observation.json')):
    a=Observation.from_dict(json.loads(p.read_text()))
    b=Observation.from_dict(json.loads((right/'session/cases'/p.parent.name/p.name).read_text()))
    rows.append(dict(case=p.parent.name,arrays={k:dict(bytes_equal=getattr(a,k).tobytes()==getattr(b,k).tobytes(),
        maximum_delta=float(np.max(np.abs(getattr(a,k)-getattr(b,k))))) for k in ('xy','samples')}))
differences=[]
def walk(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in a:walk(a[k],b[k],path+'/'+k)
    elif isinstance(a,list) and isinstance(b,list):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
    elif a!=b:
        differences.append(dict(path=path,left=a,right=b,delta=abs(a-b) if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None))
walk(json.loads((left/'evaluation/transfer.json').read_text()),json.loads((right/'evaluation/transfer.json').read_text()))
result=dict(cases=rows,report_differences=differences)
with (run/'linux-comparison.result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(observed=len(rows),exact_nominal=sum(r['arrays']['xy']['bytes_equal'] for r in rows),
    exact_samples=sum(r['arrays']['samples']['bytes_equal'] for r in rows),report_numeric_differences=sum(r['delta'] is not None for r in differences),
    max_numeric_delta=max(r['delta'] for r in differences if r['delta'] is not None),
    other=[r['path'] for r in differences if r['delta'] is None]),indent=2))
