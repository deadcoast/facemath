import json
from pathlib import Path
from facemath.observation.study import file_inventory
run=Path(__file__).resolve().parent
root=run.parents[1]
ev=root/'evidence/model-transfer-20261002'
pairs=[(run/'release/source-native',ev/'preseal/native-pipeline'),
       (run/'release/wheel-check/native-pipeline',ev/'postseal/installed-native'),
       (run/'release/archive-native',ev/'postseal/archive-native'),
       (run/'linux-native',ev/'development/linux-native')]
result=[]
for source,target in pairs:
    a,b=file_inventory(source),file_inventory(target)
    assert a==b,(str(source),str(target))
    result.append(dict(source=str(source),retained=str(target),exact_bytes=True,files=len(a),
                       bytes=sum(v['bytes'] for v in a.values())))
with (run/'retained-copy.result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
