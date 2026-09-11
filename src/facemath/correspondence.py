from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .state import CorrespondenceField


def dense_correspondence_farneback(a:np.ndarray,b:np.ndarray,mask:np.ndarray|None=None,fb_threshold:float=1.5)->CorrespondenceField:
    """Dense A->B correspondence with forward/backward consistency confidence."""
    aa=np.asarray(a,np.float32); bb=np.asarray(b,np.float32)
    if aa.max() <= 1.0: aa = aa * 255.0
    if bb.max() <= 1.0: bb = bb * 255.0
    f=cv2.calcOpticalFlowFarneback(aa,bb,None,.5,4,31,4,7,1.5,0)
    r=cv2.calcOpticalFlowFarneback(bb,aa,None,.5,4,31,4,7,1.5,0)
    h,w=aa.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    qx=xx+f[...,0].astype(np.float32); qy=yy+f[...,1].astype(np.float32)
    rb_x=cv2.remap(r[...,0],qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
    rb_y=cv2.remap(r[...,1],qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
    fb=np.hypot(f[...,0]+rb_x,f[...,1]+rb_y)
    conf=np.exp(-(fb/max(fb_threshold,1e-6))**2)
    grad_a=np.hypot(*np.gradient(aa)); grad_b=np.hypot(*np.gradient(bb))
    texture=np.clip((gaussian_filter(grad_a,1)+gaussian_filter(grad_b,1))/(np.percentile(np.r_[grad_a.ravel(),grad_b.ravel()],90)+1e-6),0,1)
    conf*=.25+.75*texture
    valid=np.isfinite(fb)&(fb<=fb_threshold*2.5)&(qx>=0)&(qx<w-1)&(qy>=0)&(qy<h-1)
    if mask is not None: valid &= np.asarray(mask,bool)
    conf=np.where(valid,conf,0)
    return CorrespondenceField(f,conf,valid,'farneback_forward_backward')


def weighted_flow_statistics(field:CorrespondenceField,mask:np.ndarray|None=None)->dict[str,float]:
    valid=field.valid.copy()
    if mask is not None: valid &= np.asarray(mask,bool)
    if not np.any(valid):
        return {'support_fraction':0.0,'mean_magnitude':float('nan'),'median_magnitude':float('nan'),'p90_magnitude':float('nan'),'mean_confidence':0.0}
    mag=np.linalg.norm(field.uv,axis=2); w=field.confidence*valid
    vals=mag[valid]; ww=w[valid]
    mean=float(np.sum(vals*ww)/max(np.sum(ww),1e-12))
    return {'support_fraction':float(np.mean(valid)),'mean_magnitude':mean,'median_magnitude':float(np.median(vals)),'p90_magnitude':float(np.percentile(vals,90)),'mean_confidence':float(np.mean(field.confidence[valid]))}


def region_correspondence_summary(field:CorrespondenceField,regions:dict)->dict[str,dict[str,float]]:
    return {name:weighted_flow_statistics(field,reg.mask) for name,reg in regions.items()}
