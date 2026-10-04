"""Exercise production paired-edge output on retained native observations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from check_structural_pair import main as run_pair
from facemath.observation.study import read_json, write_json_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline',type=Path); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); run_pair()
    root=args.out.resolve()
    summaries=[read_json(root/kind/'summary.json')['edge_comparison'] for kind in ('nominal','calibrated','replayed')]
    assert summaries[0]==summaries[1]==summaries[2]
    assert read_json(root/'calibrated.log')['comparison']['edge_comparison']==summaries[1]
    rows=summaries[1]['rows']; assert len(rows)==8
    with np.load(root/'calibrated/arrays.npz') as arrays:
        for row in rows:
            if row['array_prefix'] is None:
                assert row['supported']==0 and row['unresolved']==row['total']
                continue
            prefix=row['array_prefix']; accepted=arrays[prefix+'accepted']
            a,b=arrays[prefix+'a_best_offset_px'],arrays[prefix+'b_best_offset_px']
            delta=arrays[prefix+'delta_b_minus_a_px']
            np.testing.assert_array_equal(delta[accepted],(b-a)[accepted])
            np.testing.assert_array_equal(delta[~accepted],0.)
            for name in ('points','normals','offsets'):
                np.testing.assert_array_equal(arrays[prefix+'a_'+name],arrays[prefix+'b_'+name])
            np.testing.assert_array_equal(accepted,(arrays[prefix+'rejection_a']==0)&(arrays[prefix+'rejection_b']==0))
            assert int(accepted.sum())==row['supported'] and len(accepted)==row['total']
            for side in ('a','b'): assert sum(row['rejections'][side].values())==row['total']
            if accepted.any():
                np.testing.assert_allclose(row['canonical_pixels']['rms_offset'],np.sqrt(np.mean(delta[accepted]**2)),atol=1e-12)
            else: assert row['canonical_pixels']['rms_offset'] is None
    comparison=read_json(root/'comparison.json')
    result={'format':'facemath.edge_pair_check.v21','raw_arrays_checked':comparison['raw_arrays_checked'],
        'same_before_after_calibration_and_replay':True,'shared_ray_coordinates_verified':True,
        'rows':rows,'new_native_inference':False,'fit_performed':False,
        'scope':'Retained native pair, same frozen first two sources; image gradients are not independent anatomical labels'}
    write_json_new(root/'edge-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__': main()
