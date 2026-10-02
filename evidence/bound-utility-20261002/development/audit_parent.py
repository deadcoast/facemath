"""Recompute old retained qualification without replacing any old artifact."""
import json
from pathlib import Path
import numpy as np
from facemath.observation.qualification import qualify
from facemath.observation.study import load_study, read_json
from facemath.observation.study_runner import verify_session
from facemath.observation.study_io import verify_qualification_bundle
from facemath.observation.journal import code_fingerprint

root = Path('evidence/source-cohort-20261002/preseal/native-pipeline/qualification')
assert verify_qualification_bundle(root)['verified']
study = load_study(root/'acquisition/study')
old = read_json(root/'qualification.json')
report, arrays, model = qualify(study, read_json(root/'raw_validation/validation.json'), verify_session(root/'acquisition'))
assert json.loads(json.dumps(model.to_dict())) == read_json(root/'localization_calibration.json')
with np.load(root/'arrays.npz', allow_pickle=False) as original:
    assert set(original.files) == set(arrays)
    for key, array in arrays.items():
        assert array.dtype == original[key].dtype and array.shape == original[key].shape
        assert array.tobytes() == original[key].tobytes(), key
utility = report.pop('bound_utility')
report.pop('content_sha256'); old.pop('content_sha256')
assert json.loads(json.dumps(report)) == old
result = dict(parent=str(root), engine_code_sha256=code_fingerprint(), fitted_model_unchanged=True,
              qualification_arrays_byte_identical=sorted(arrays), old_report_fields_identical=True,
              retrospective_utility_diagnostic=utility,
              limit='Old LFPW evaluation is not retuned; post-hoc width diagnostic, no new acceptance claim')
with Path('runs/bound-utility-20261002/parent-invariance.result.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='retrospective_utility_diagnostic'},indent=2))
