"""Same pixels, changed scripted error: a source-only router cannot distinguish it."""
from pathlib import Path
import argparse,json
import numpy as np
from facemath.observation.acquisition_lab import measured_record,demonstration_policy,APERTURE
from facemath.observation.acquisition import ApertureSource,MeasuredGroup,write_group
from facemath.observation.stratified_lab import observation
from facemath.observation.study import write_json_new
from facemath.observation.joint_lab import INDICES

def run(root):
    root.mkdir(parents=True,exist_ok=False)
    a=measured_record(290010,'clear',hidden_shift=False)
    b=measured_record(290010,'clear',hidden_shift=True)
    policy=demonstration_policy()
    ga=MeasuredGroup(policy,'identical-pixel-source',(ApertureSource(a['image'],('counterfactual-source',),APERTURE),))
    gb=MeasuredGroup(policy,'identical-pixel-source',(ApertureSource(b['image'],('counterfactual-source',),APERTURE),))
    oa,ob=observation(a),observation(b)
    assert np.array_equal(a['image'],b['image']) and ga.digest==gb.digest
    assert not np.array_equal(oa.xy,ob.xy)
    write_group(ga,root/'source')
    write_json_new(root/'observer_original.json',oa.to_dict());write_json_new(root/'observer_shifted.json',ob.to_dict())
    delta=ob.xy[INDICES]-oa.xy[INDICES]
    np.savez_compressed(root/'control.npz',original=oa.xy,shifted=ob.xy,reference_indices=INDICES,
                        reference8_delta=delta,features_original=ga.features,features_shifted=gb.features)
    result={'kind':'SYNTHETIC_COUNTERFACTUAL_NOT_REAL_MODEL','seed':290010,'pixels_identical':True,
            'source_group_sha256':ga.digest,'source_measurements_and_route_identical':True,
            'category':ga.category,'reference8_mean_changed_coordinate_px':float(np.linalg.norm(delta,axis=1).mean()),
            'reference8_max_changed_coordinate_px':float(np.linalg.norm(delta,axis=1).max()),
            'meaning':'identical observable input cannot distinguish a scripted hidden change in localization error; no shift detection claimed'}
    write_json_new(root/'control.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(run(Path(a.out)),indent=2))
