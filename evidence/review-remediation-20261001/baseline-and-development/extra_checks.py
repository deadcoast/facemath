import importlib.metadata
import json
from pathlib import Path
import re
import sys

import numpy as np
from facemath.observation import FacialInstrument,InstrumentConfig
from facemath.observation.artifacts import write_bundle,verify_bundle
from facemath.observation.journal import environment
from facemath.observation.lab import phantom_points,phantom_observation,render_phantom

root=Path(__file__).resolve().parents[2]
paths=[p for directory in ('src/facemath','tests','scripts') for p in (root/directory).rglob('*.py')]
for path in paths:compile(path.read_bytes(),str(path),'exec')
print(json.dumps({'environment':environment(),'actual_interpreter_parse_count':len(paths),
                  'metadata_version':importlib.metadata.version('facemath')},indent=2))
records=[]
for missing in ((),(12,),(10,11,12,13)):
    points=phantom_points();other=points.copy()
    for i in missing:other[i]=[500+i,500]
    image=render_phantom(points)
    result=FacialInstrument(InstrumentConfig(canvas_size=128,eye_span_px=40,uncertainty_draws=8,enable_flow=False)).run(
        image,image,phantom_observation(image,points),phantom_observation(image,other))
    out=Path(__file__).parent/f'denominator-{sys.version_info.minor}-{len(missing)}'
    write_bundle(result,out,image,image)
    html=(out/'report.html').read_text(encoding='utf-8')
    evidence=json.loads((out/'evidence.json').read_text(encoding='utf-8'))
    for suffix in ('rms','change'):
        name=f'registration.heldout_nose.{suffix}'
        row=next(row for row in re.findall(r'<tr.*?</tr>',html,re.S) if name in row)
        assert f'{4-len(missing)} / 4' in row,row
        measurement=next(m for m in evidence['measurement_profile'] if m['name']==name)
        assert (measurement['support'],measurement['total'])==(4-len(missing),4)
    with np.load(out/'arrays.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['observation_b_xy'],other)
    records.append({'missing':missing,'json_and_html_denominator':4,'support':4-len(missing),
                    'original_observations_unchanged':True,'bundle':verify_bundle(out)})
print(json.dumps(records,indent=2))
