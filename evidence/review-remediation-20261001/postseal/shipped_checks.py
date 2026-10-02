import json
from pathlib import Path
import sys

import facemath
from facemath.cli import build_parser,execute
from facemath.observation.journal import environment

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert Path(facemath.__file__).resolve().is_relative_to(root/'src')
print(json.dumps({'import':facemath.__file__,'environment':environment()},indent=2),flush=True)
lab=root/'evidence/review-remediation-20261001/preseal/laboratory'
for kind,relative in [('qualification','positive/qualification'),('pair','admitted_pair')]:
    for command in (['verify',str(lab/relative)],['replay',str(lab/relative),'--out',str(out/kind)]):
        result=execute(build_parser().parse_args(command))
        print(json.dumps({'command':command,'result':result},indent=2),flush=True)
