"""Quantify a rejected exact replay without changing its acceptance rule."""
import json
from pathlib import Path
import sys
import numpy as np

left,right,out=map(Path,sys.argv[1:])
differences=[]
def walk(a,b,path):
    if type(a) is not type(b):
        differences.append(dict(path=path,kind='type',left=a,right=b))
    elif isinstance(a,dict):
        if set(a)!=set(b):differences.append(dict(path=path,kind='keys'))
        for k in sorted(set(a)&set(b)):walk(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list):
        if len(a)!=len(b):differences.append(dict(path=path,kind='length'))
        for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
    elif a!=b:
        row=dict(path=path,kind='number' if isinstance(a,(float,int)) else 'value',left=a,right=b)
        if row['kind']=='number':row['absolute_delta']=abs(a-b)
        differences.append(row)
for name in ('qualification.json','localization_calibration.json'):
    walk(json.loads((left/name).read_text()),json.loads((right/name).read_text()),name)
arrays=[]
with np.load(left/'arrays.npz',allow_pickle=False) as a,np.load(right/'arrays.npz',allow_pickle=False) as b:
    assert set(a.files)==set(b.files)
    for name in a.files:
        x,y=a[name],b[name]
        same=x.shape==y.shape and x.dtype==y.dtype
        arrays.append(dict(name=name,same_layout=same,bytes_equal=same and x.tobytes()==y.tobytes(),
            maximum_absolute_delta=float(np.max(np.abs(x-y))) if same and x.size and x.dtype.kind in 'fiu' else None))
result=dict(kind='rejected_exact_replay_difference',left=str(left),right=str(right),
    json_differences=differences,arrays=arrays,
    exact_replay_accepted=False,tolerance_changed=False,
    meaning='diagnostic only; recomputed output is not substituted for the retained result')
with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(json_differences=len(differences),examples=differences[:12],
    changed_arrays=[a for a in arrays if not a['bytes_equal']]),indent=2))
