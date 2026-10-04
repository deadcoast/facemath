from pathlib import Path
import json
import numpy as np
from facemath.observation.contracts import Observation
from facemath.observation.reference import mapped_prediction, observer_recipe
from facemath.observation.anatomy import MP_ANCHORS
root=Path.cwd(); out=root/'runs/landmark-repair-20261004'
p=root/'evidence/bound-utility-20261002/preseal/native-pipeline/session/cases/case_00000/observation.json'
obs=Observation.from_dict(json.loads(p.read_text()))
recipe_before=observer_recipe(obs); pred_before=mapped_prediction(obs)[0]
MP_ANCHORS['nose_tip']=4
recipe_after=observer_recipe(obs); pred_after=mapped_prediction(obs)[0]
r={'failure_reproduced': recipe_before==recipe_after and not np.array_equal(pred_before,pred_after), 'recipe_before':recipe_before,'recipe_after':recipe_after,'nose_before':pred_before[4].tolist(),'nose_after':pred_after[4].tolist(),'scope':'in-memory mutation in isolated diagnostic process; no file or observation bytes changed'}
(out/'pre-fix-reproduction.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
