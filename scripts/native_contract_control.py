"""Positive wrapper API fixture. NO native model or runtime is executed."""
from pathlib import Path
from unittest.mock import patch
from hashlib import sha256
import argparse,json
import numpy as np
from facemath.observation.contracts import Observation,array_digest,canonical_json
from facemath.observation.study import save_rgb,load_rgb
from facemath.observation.study_runner import AttemptOutcome
from facemath.observation.native_check import native_smoke,verify_smoke


def run(out:Path):
    out.mkdir(parents=True,exist_ok=False)
    image=np.full((64,64,3),127,np.uint8);save_rgb(out/'fixture.png',image)
    model=out/'NOT-A-MODEL.task';model.write_bytes(b'API fixture bytes; no inference')
    def execute(self,path,decl,directory):
        source=load_rgb(path);xy=np.column_stack((np.arange(478)%22+10,np.arange(478)//22+10)).astype(float)
        obs=Observation(array_digest(source),source.shape[:2],'mediapipe478.v1',tuple(f'mp_{i:03}' for i in range(478)),
            xy,np.repeat(xy[None],7,0),decl.provider,decl.provider_fingerprint,decl.noise_floor_px,7,
            tuple(f'fixture_{i}' for i in range(7)),('API_FIXTURE_NOT_NATIVE',),
            canonical_json({'model_sha256':decl.model_sha256,'runtime_version':decl.runtime_version,'api_fixture':True}))
        (directory/'fixture-marker.txt').write_text('Synthetic returned Observation; no subprocess or detector executed')
        return AttemptOutcome('observed',obs)
    with patch('facemath.observation.native_check.importlib.metadata.version',return_value='API-FIXTURE-NOT-NATIVE'), \
         patch('facemath.observation.native_check.NativeExecutor.preflight',return_value={'ready':True,'api_fixture':True}), \
         patch('facemath.observation.native_check.NativeExecutor.execute',execute):
        report=native_smoke(out/'fixture.png',model,out/'simulated-outcome')
    assert report['landmarks']==478 and report['samples']==7 and not report['human_accuracy_validated']
    assert report['observation_sha256'] and verify_smoke(out/'simulated-outcome')['verified']
    return {'kind':'API_FIXTURE_NOT_NATIVE','native_inference_executed':False,'wrapper_positive_branch_verified':True,
        'source_binding_verified':True,'478_coordinates_and_7_samples_preserved':True,'human_accuracy_validated':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.out),indent=2))
