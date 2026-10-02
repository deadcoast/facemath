from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import shutil
import numpy as np

root=Path.cwd(); run=root/'runs/native-pipeline-20261002'; post=root/'evidence/native-pipeline-20261002/postseal'
for name in ('native-pair-01','native-pair-replay','native-pair-archive-replay'):
    shutil.copytree(run/name,post/name)
for stem in ('native-pair-analyze','native-pair-verify','native-pair-replay','native-pair-archive-replay','assemble-final'):
    for suffix in ('.json','.log'):
        with (post/(stem+suffix)).open('xb') as stream:stream.write((run/(stem+suffix)).read_bytes())
with (post/'retain_supplement.py').open('xb') as stream:stream.write(Path(__file__).read_bytes())
arrays={}
with np.load(post/'native-pair-01/arrays.npz',allow_pickle=False) as original:
    for name in ('native-pair-replay','native-pair-archive-replay'):
        with np.load(post/name/'arrays.npz',allow_pickle=False) as replay:
            arrays[name]={k:(original[k].shape==replay[k].shape and original[k].dtype.str==replay[k].dtype.str
                            and original[k].tobytes()==replay[k].tobytes()) for k in original.files}
assert all(all(values.values()) for values in arrays.values())
base=root/'evidence/native-pipeline-20261002/pilot/session/cases'
repetitions={}
for name,path in [('source',run/'release-check/source-native/session/cases'),
                  ('installed_wheel',post/'installed-native/session/cases'),
                  ('extracted_archive',post/'archive-native/session/cases')]:
    records=list(path.glob('*/observation.json'))
    repetitions[name]={p.parent.name:sha256(p.read_bytes()).hexdigest()==sha256((base/p.parent.name/p.name).read_bytes()).hexdigest() for p in records}
assert all(len(values)==43 and all(values.values()) for values in repetitions.values())
result={'format':'facemath.postseal_native_pair_and_byte_repetition.v11',
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'pair_cases':['case_00027','case_00034'],
    'selection':'first case of the first two frozen evaluation source groups; research pair without localization calibration',
    'array_byte_identity':arrays,'native_observation_file_sha256_identity':repetitions,
    'meaning':'These executed repetitions only; no universal native determinism or human-reference accuracy claim.'}
with (post/'NATIVE_PAIR_VALIDATION.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print('Native observation bytes identical: 43/43 in all four runs. Pair arrays identical: 151/151 in source and archive replays.')
