from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .core import Frame, LandmarkSet
from .geometry import apply_homography, similarity_transform
from .state import RegistrationDiagnostics, SupportLevel


def weighted_similarity_transform(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray,float,float]:
    """Weighted Umeyama-style 2D similarity transform a -> b."""
    from .observation.canonical import proper_similarity
    m, wrmse = proper_similarity(a, b, weights)
    scale = float(np.sqrt(np.linalg.det(m[:2, :2])))
    return m, scale, wrmse


def ransac_similarity_transform(a: np.ndarray,b: np.ndarray,threshold_px: float=3.0,max_trials:int=512,seed:int=0) -> tuple[np.ndarray,np.ndarray,float]:
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2: raise ValueError('at least two correspondences required')
    rng=np.random.default_rng(seed)
    best=None
    for _ in range(max_trials):
        idx=rng.choice(len(a),size=2,replace=False)
        try: m,_,_=similarity_transform(a[idx],b[idx])
        except Exception: continue
        e=np.linalg.norm(apply_homography(a,m)-b,axis=1)
        inliers=e<=threshold_px
        score=(int(inliers.sum()),-float(np.median(e[inliers])) if inliers.any() else -np.inf)
        if best is None or score>best[0]: best=(score,inliers,m)
    if best is None: raise RuntimeError('RANSAC could not fit transform')
    inliers=best[1]
    if inliers.sum()>=2:
        m,_,rmse=similarity_transform(a[inliers],b[inliers])
    else:
        m=best[2]; e=np.linalg.norm(apply_homography(a,m)-b,axis=1); rmse=float(np.sqrt(np.mean(e*e)))
    return m,inliers,rmse


def warp_frame(frame: Frame, m: np.ndarray, shape_hw: tuple[int,int]) -> Frame:
    h,w=shape_hw
    rgb=cv2.warpAffine(frame.rgb,m[:2].astype(np.float32),(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return Frame(rgb,frame.timestamp,frame.source)


def refine_ecc(a_gray: np.ndarray,b_gray: np.ndarray,initial: np.ndarray,mask: np.ndarray | None=None,iterations:int=100,eps:float=1e-6) -> tuple[np.ndarray,float|None]:
    """Refine an affine warp by ECC. Falls back to initial on non-convergence."""
    warp=np.asarray(initial[:2],np.float32).copy()
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,iterations,eps)
    imask=None if mask is None else (np.asarray(mask,bool).astype(np.uint8)*255)
    try:
        cc,warp=cv2.findTransformECC(np.asarray(b_gray,np.float32),np.asarray(a_gray,np.float32),warp,cv2.MOTION_AFFINE,criteria,imask,5)
        m=np.eye(3); m[:2]=warp
        return m,float(cc)
    except cv2.error:
        return np.asarray(initial,float),None


def overlap_fraction(shape_hw:tuple[int,int],m:np.ndarray) -> float:
    h,w=shape_hw
    ones=np.ones((h,w),np.uint8)*255
    warped=cv2.warpAffine(ones,m[:2].astype(np.float32),(w,h),flags=cv2.INTER_NEAREST,borderValue=0)
    return float(np.mean(warped>0))


def diagnose_registration(a: LandmarkSet|None,b:LandmarkSet|None,m:np.ndarray,shape_hw:tuple[int,int],inliers:np.ndarray|None=None,ecc:float|None=None) -> RegistrationDiagnostics:
    rmse=wrmse=None
    inlier_fraction=None
    warnings=[]
    if a is not None and b is not None:
        pred=apply_homography(a.xy,m); e=np.linalg.norm(pred-b.xy,axis=1)
        rmse=float(np.sqrt(np.mean(e*e)))
        if a.confidence is not None or b.confidence is not None:
            wa=np.ones(len(e)) if a.confidence is None else np.asarray(a.confidence,float)
            wb=np.ones(len(e)) if b.confidence is None else np.asarray(b.confidence,float)
            w=np.clip(np.minimum(wa,wb),0,1); w=w/max(w.sum(),1e-12)
            wrmse=float(np.sqrt(np.sum(w*e*e)))
        if inliers is not None: inlier_fraction=float(np.mean(inliers))
    overlap=overlap_fraction(shape_hw,m)
    support=SupportLevel.VALID
    scale=max(shape_hw)
    if overlap<.75:
        warnings.append('low_raster_overlap'); support=SupportLevel.DEGRADED
    if rmse is not None and rmse>0.03*scale:
        warnings.append('high_landmark_registration_error'); support=SupportLevel.DEGRADED
    if overlap<.5 or (inlier_fraction is not None and inlier_fraction<.5):
        support=SupportLevel.INSUFFICIENT
    return RegistrationDiagnostics('landmark_similarity',np.asarray(m,float),rmse,wrmse,inlier_fraction,overlap,ecc,support,tuple(warnings))
