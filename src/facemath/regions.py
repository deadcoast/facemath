from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import cv2
import numpy as np

from .core import LandmarkSet
from .state import Region


@dataclass(frozen=True)
class RegionSchema:
    """Landmark-name driven anatomical hierarchy; schema is detector-agnostic."""
    groups: Mapping[str, tuple[str,...]]
    parents: Mapping[str,str|None]
    weights: Mapping[str,float]


DEFAULT_SCHEMA=RegionSchema(
    groups={
        'left_eye':('left_eye_outer','left_eye_inner','left_eye_top','left_eye_bottom'),
        'right_eye':('right_eye_inner','right_eye_outer','right_eye_top','right_eye_bottom'),
        'nose':('nose_bridge','nose_tip','left_nostril','right_nostril'),
        'mouth':('mouth_left','mouth_right','upper_lip','lower_lip'),
        'central':('left_eye_inner','right_eye_inner','nose_bridge','nose_tip','mouth_left','mouth_right','upper_lip','lower_lip','chin'),
        'jaw':('jaw_left','chin','jaw_right'),
        'face':('temple_left','temple_right','jaw_left','chin','jaw_right'),
    },
    parents={'left_eye':'central','right_eye':'central','nose':'central','mouth':'central','central':'face','jaw':'face','face':None},
    weights={'left_eye':1.4,'right_eye':1.4,'nose':1.35,'mouth':1.2,'central':1.5,'jaw':0.9,'face':1.0},
)


def _indices(ls:LandmarkSet,names:tuple[str,...])->list[int]:
    if ls.names is None: return []
    lut={n:i for i,n in enumerate(ls.names)}
    return [lut[n] for n in names if n in lut]


def polygon_mask(shape_hw:tuple[int,int],points:np.ndarray,padding_px:float=0)->np.ndarray:
    h,w=shape_hw; mask=np.zeros((h,w),np.uint8)
    p=np.asarray(points,float)
    if len(p)<3: return mask.astype(bool)
    hull=cv2.convexHull(np.round(p).astype(np.int32))
    cv2.fillConvexPoly(mask,hull,255)
    if padding_px>0:
        k=max(1,int(round(padding_px)))
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*k+1,2*k+1))
        mask=cv2.dilate(mask,kernel)
    return mask.astype(bool)


def ellipse_face_mask(shape_hw:tuple[int,int],landmarks:LandmarkSet,expand:float=1.15)->np.ndarray:
    p=landmarks.xy
    if len(p)<3: return np.ones(shape_hw,bool)
    center=p.mean(0)
    q=p-center
    cov=np.cov(q.T)+np.eye(2)*1e-6
    vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
    axes=np.maximum(8,np.sqrt(vals)*2.6*expand)
    h,w=shape_hw; m=np.zeros((h,w),np.uint8)
    angle=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]))
    cv2.ellipse(m,tuple(np.round(center).astype(int)),tuple(np.round(axes).astype(int)),float(angle),0,360,255,-1)
    return m.astype(bool)


def build_regions(shape_hw:tuple[int,int],landmarks:LandmarkSet,schema:RegionSchema=DEFAULT_SCHEMA,padding_ratio:float=.025)->dict[str,Region]:
    scale=max(shape_hw); pad=padding_ratio*scale
    out={}
    for name,names in schema.groups.items():
        idx=_indices(landmarks,names)
        if len(idx)>=3:
            m=polygon_mask(shape_hw,landmarks.xy[idx],pad)
            out[name]=Region(name,m,float(schema.weights.get(name,1.0)),schema.parents.get(name))
    return out
