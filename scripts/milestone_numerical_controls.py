"""Consolidate independently executable earlier analytic controls, unmodified."""
from pathlib import Path
import argparse,json
from facemath.observation.lab import laboratory
from facemath.observation.stratified_lab import run_mixed_radius_controls
from facemath.observation.acquisition_lab import run_pixel_controls
from facemath.observation.study import write_json_new

def run(out:Path):
    out.mkdir(parents=True,exist_ok=False)
    analytic=laboratory();write_json_new(out/'analytic.json',analytic)
    if not all(analytic['checks'].values()):raise AssertionError('inherited analytic controls failed')
    mixed=run_mixed_radius_controls(out/'mixed_radius',samples=1024)
    if mixed['sampled_relation_violations'] or mixed['sampled_shape_violations']:
        raise AssertionError('inherited analytic enclosures violated by retained sampled sets')
    pixels=run_pixel_controls(out/'pixel_controls')
    summary={'format':'facemath.consolidated_controls.v10','analytic_checks':analytic['checks'],
        'mixed_radius':mixed,'pixel_controls':pixels,
        'scope':'analytic and synthetic source/shape controls, not human observation validation'}
    write_json_new(out/'controls.json',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.out),indent=2))
