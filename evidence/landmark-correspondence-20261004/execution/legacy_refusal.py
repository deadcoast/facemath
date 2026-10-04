from pathlib import Path
import json
from facemath.observation.study import read_json
from facemath.observation.transfer import require_transfer, load_capsule
from facemath.observation.contracts import ObservationError
root=Path.cwd(); out=root/'evidence/landmark-correspondence-20261004'
old=read_json(root/'evidence/enrollment-registry-20261003/helen-model.json')
model=load_capsule(old)
spec=read_json(root/'runs/landmark-repair-20261004/prospective-study/study.lock.json')
spec['bias_mode']='none';spec.pop('joint_error',None)
try:
    require_transfer(old,spec,[])
except ObservationError as exc:
    assert 'observer/rater/alpha contracts' in str(exc),str(exc)
    result={'legacy_capsule_validated':True,'legacy_model_sha256':model.digest,'refused_before_inference':True,'reason':str(exc),'new_observer_recipe_sha256':spec['observer_recipe_sha256'],'old_observer_recipe_sha256':model.bias.observer_recipe_sha256,'native_calls':0,'scope':'known legacy native capsule rejected at recipe comparison, before enrollment or inference'}
else: raise AssertionError('incompatible legacy model accepted')
with (out/'legacy-model-refusal.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
