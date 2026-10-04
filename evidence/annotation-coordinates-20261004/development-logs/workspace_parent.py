from pathlib import Path
import sys,json,re
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'runs/core-features-20261004/release-01/extracted/facemath-0.19.0/src'))
from facemath.observation.milestone_lab import make_milestone_workspace_spec
from facemath.observation.workspace import prepare_workspace,import_reference
from facemath.observation.study import read_json,write_json_new
out=root/'evidence/annotation-coordinates-20261004/workspace-parent'
path,_,labels=make_milestone_workspace_spec(out/'inputs',compact=True)
data=read_json(path);data['sources']=data['sources'][:1]
one=out/'inputs/single.spec.json';write_json_new(one,data)
workspace=prepare_workspace(one,out/'workspace')
bench=workspace/'workbenches/source_00000_rater_000.html'
html=bench.read_text(encoding='utf-8')
payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',html,re.S)[1])
# Reproduce the emitted kind through the actual template's export contract;
# coordinates are known synthetic fixture labels, not a human annotation.
reference=read_json(labels[0][1]);reference['kind']=payload['kind']
reference['metadata']={'reproduction':'parent workbench emitted kind with scripted fixture coordinates; NOT human labels'}
export=out/'emitted-kind.reference.json';write_json_new(export,reference)
try:import_reference(workspace,export,labels[0][0])
except Exception as exc:failure=str(exc)
else:raise AssertionError('Expected parent kind mismatch')
result={'observer_kind':data['study']['observer']['execution_kind'],'emitted_kind':payload['kind'],
        'import_failure':failure,'scope':'synthetic control; no human annotations and no inference'}
write_json_new(out/'failure.json',result);print(json.dumps(result,indent=2))
