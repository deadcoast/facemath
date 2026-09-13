from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import cv2
from .core import Frame, FacePair, LandmarkSet
from .wave2 import FacialGeometryEngine
from .video import sample_frames
from .sequence import analyze_sequence
from .report import build_report


def load_image(path: str) -> Frame:
    bgr=cv2.imread(path,cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return Frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source=path)


def load_landmarks(path: str | None) -> LandmarkSet | None:
    if path is None: return None
    obj=json.loads(Path(path).read_text())
    conf=obj.get('confidence')
    return LandmarkSet(np.asarray(obj['xy'],float),tuple(obj.get('names',[])) or None,None if conf is None else np.asarray(conf,float))


def _clean(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if hasattr(v,'value') and isinstance(v.value,str): return v.value
    if isinstance(v,dict): return {str(k):_clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_clean(x) for x in v]
    return v


def main(argv: list[str] | None = None) -> None:
    ap=argparse.ArgumentParser(prog='facemath')
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('compare',help='Wave-2 integrated pair analysis')
    p.add_argument('image_a'); p.add_argument('image_b')
    p.add_argument('--landmarks-a'); p.add_argument('--landmarks-b')
    p.add_argument('--out',default='facemath_evidence.json')
    v=sub.add_parser('sequence',help='Analyze temporal/morph stability across a video')
    v.add_argument('video'); v.add_argument('--frames',type=int,default=48); v.add_argument('--out',default='facemath_sequence.json')
    args=ap.parse_args(argv)
    if args.cmd=='compare':
        pair=FacePair(load_image(args.image_a),load_image(args.image_b),load_landmarks(args.landmarks_a),load_landmarks(args.landmarks_b))
        bundle=FacialGeometryEngine().run(pair)
        report=build_report(bundle)
        Path(args.out).write_text(json.dumps(_clean(asdict(report)),indent=2),encoding='utf-8')
        print(args.out)
    elif args.cmd=='sequence':
        frames=sample_frames(args.video,args.frames)
        ev=analyze_sequence(frames)
        Path(args.out).write_text(json.dumps(_clean(asdict(ev)),indent=2),encoding='utf-8')
        print(args.out)

if __name__ == '__main__':
    main()
