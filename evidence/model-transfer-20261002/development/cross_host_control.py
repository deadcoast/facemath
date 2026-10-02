"""Cross-host exact replay is refused before recomputation; model import is allowed."""
import json
import sys
from pathlib import Path
from facemath.observation.contracts import ObservationError
from facemath.observation.transfer_io import replay_transfer_bundle

root=Path('/mnt/c/Users/ryanf/github/_active/facemath')
run=root/'runs/model-transfer-20261002'
target=run/'cross-host-refused'
try:
    replay_transfer_bundle(Path(sys.argv[1]) if len(sys.argv)>1 else run/'release/source-native/evaluation',target)
except ObservationError as exc:
    assert 'matching numerical runtime' in str(exc),str(exc)
    assert not target.exists()
    result=dict(expected_refusal=True,reason=str(exc),recomputation_started=False,
        limit='A saved fitted model can be evaluated under a separately retained new runtime; exact replay of an old run requires its numerical runtime')
else:
    raise AssertionError('unexpected cross-host exact replay admission')
with (run/'cross-host.result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
