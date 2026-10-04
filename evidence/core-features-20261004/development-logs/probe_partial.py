from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from facemath.observation.contracts import Observation
from facemath.observation.anatomy import anatomy_for
from facemath.observation.canonical import canonicalize, InstrumentConfig
from facemath.observation.instrument import FacialInstrument, InstrumentFailure
from facemath.observation.study import load_rgb

root=Path(__file__).resolve().parents[2]
bundle=root/'evidence/structural-comparison-20261004/pair-01/nominal'
stored=json.loads((bundle/'observations.json').read_text())
observation=Observation.from_dict(stored['a'])
image=load_rgb(bundle/'source_a.png')
anatomy=anatomy_for(observation)
ids=sorted({i for name in anatomy.fit_anchor_names if name not in ('left_eye','right_eye','nose_tip') for i in anatomy.anchors[name]})
xy=observation.xy.copy(); xy[ids]=[-100,-100]
samples=np.array(observation.samples); samples[:,ids]=[-100,-100]
partial=replace(observation,xy=xy,samples=samples,source_notes=observation.source_notes+('synthetic_clipping_control_not_a_native_observation',))
config=InstrumentConfig(canvas_size=96,eye_span_px=32,uncertainty_draws=4,enable_flow=False,enable_edges=False)
state=canonicalize(image,image,partial,partial,anatomy,config)
result={'scope':'Synthetic clipping control using retained native geometry; not a new native observation',
        'removed_indices':ids,'primary_fit_anchors':state.fit_names,'primary_fit_completed':True}
try:
    compared=FacialInstrument(config).run(image,image,partial,partial)
    result.update(completed=True,measurements=[m.to_dict() for m in compared.profile.measurements if m.name.startswith('registration.')])
except InstrumentFailure as exc:
    result.update(completed=False,error=str(exc),journal=exc.journal_jsonl)
print(json.dumps(result,indent=2))
