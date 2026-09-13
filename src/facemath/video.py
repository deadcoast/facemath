from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .core import Frame


def read_video(path: str | Path) -> tuple[list[Frame], float]:
    cap=cv2.VideoCapture(str(path))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    frames=[]; i=0
    while True:
        ok,bgr=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        frames.append(Frame(rgb,i/fps if fps else None,str(path)))
        i+=1
    cap.release()
    return frames,fps


def sample_frames(path: str | Path, count: int = 12) -> list[Frame]:
    cap=cv2.VideoCapture(str(path))
    fps=float(cap.get(cv2.CAP_PROP_FPS)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n<=0: return []
    ids=np.linspace(0,n-1,count).round().astype(int)
    out=[]
    for idx in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,bgr=cap.read()
        if ok:
            out.append(Frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),idx/fps if fps else None,str(path)))
    cap.release(); return out
