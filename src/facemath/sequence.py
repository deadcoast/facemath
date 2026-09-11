from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Frame
from .signal import structure_tensor
from .temporal import robust_pca_pcp, slow_feature_analysis


@dataclass(frozen=True)
class SequenceEvidence:
    timestamps: np.ndarray
    frame_deltas: np.ndarray
    structure_deltas: np.ndarray
    temporal_smoothness: float
    structural_smoothness: float
    slow_feature_values: np.ndarray | None
    low_rank_fraction: float | None
    admissible_as_facial_similarity: bool = False
    interpretation: str = "Two-image crossfades are inherently low rank; these statistics do not establish resemblance. Prefer facemath inspect-video."


def analyze_sequence(
    frames: list[Frame], mask: np.ndarray | None = None, max_frames: int = 64
) -> SequenceEvidence:
    if len(frames) < 3:
        raise ValueError("at least three frames required")
    if len(frames) > max_frames:
        ids = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in ids]
    xs = np.stack([f.gray for f in frames])
    if mask is not None:
        m = np.asarray(mask, bool)
        flat = xs[:, m]
    else:
        flat = xs.reshape(len(xs), -1)
    deltas = np.sqrt(np.mean(np.diff(flat, axis=0) ** 2, axis=1))
    # multichannel local structure tensors summarized per frame
    feats = []
    for x in xs:
        st = structure_tensor(x)
        if mask is not None:
            st = st[m]
        feats.append(
            np.r_[np.mean(st.reshape(-1, 4), axis=0), np.std(st.reshape(-1, 4), axis=0)]
        )
    feats = np.asarray(feats)
    sd = np.sqrt(np.mean(np.diff(feats, axis=0) ** 2, axis=1))
    smooth = float(np.sqrt(np.mean(np.diff(deltas) ** 2))) if len(deltas) > 1 else 0.0
    ssmooth = float(np.sqrt(np.mean(np.diff(sd) ** 2))) if len(sd) > 1 else 0.0
    slow_vals = None
    try:
        _, slow_vals = slow_feature_analysis(feats, min(3, feats.shape[1]))
    except Exception:
        pass
    low_rank_fraction = None
    # RPCA on downsampled vectors for bounded cost
    small = flat[
        :, :: max(1, int(np.ceil(flat.shape[1] / 4096)))
    ].T  # honor supplied mask
    try:
        l, s = robust_pca_pcp(small, max_iter=120)
        low_rank_fraction = float(np.linalg.norm(l) / max(np.linalg.norm(small), 1e-12))
    except Exception:
        pass
    ts = np.array(
        [f.timestamp if f.timestamp is not None else i for i, f in enumerate(frames)],
        float,
    )
    return SequenceEvidence(
        ts, deltas, sd, smooth, ssmooth, slow_vals, low_rank_fraction
    )
