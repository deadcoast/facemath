"""Time-aware video diagnostics; crossfade rank is never resemblance evidence.

For X[:,t] = alpha[t] A + (1-alpha[t]) B, rank(X)<=2 and
rank(X - temporal_mean)<=1 for ANY A and B. This null model is executable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import ObservationError, array_digest


def crossfade(a: np.ndarray, b: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    a, b, alpha = np.asarray(a, float), np.asarray(b, float), np.asarray(alphas, float)
    if a.shape != b.shape or alpha.ndim != 1 or not np.isfinite(alpha).all():
        raise ObservationError("crossfade needs equal endpoints and finite [T] alpha")
    if np.any((alpha < 0) | (alpha > 1)):
        raise ObservationError("alpha outside [0,1]")
    shape = (len(alpha),) + (1,) * a.ndim
    return alpha.reshape(shape) * a + (1 - alpha.reshape(shape)) * b


def sequence_diagnostics(
    frames: np.ndarray, timestamps: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, Any]:
    x = np.asarray(frames, float)
    t = np.asarray(timestamps, float)
    if x.ndim < 3 or len(x) < 3 or t.shape != (len(x),):
        raise ObservationError(
            "sequence must be [T,H,W,...], T>=3, with matching times"
        )
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ObservationError(
            "frames finite; timestamps finite and strictly increasing"
        )
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != x.shape[1:3] or not np.any(mask):
            raise ObservationError(
                "sequence mask must match spatial shape and be nonempty"
            )
        flat = x[:, mask].reshape(len(x), -1)
    else:
        flat = x.reshape(len(x), -1)
    # Bounded work before input construction is the caller's responsibility.
    centered = flat - flat.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    leading = float(energy[0] / energy.sum()) if energy.sum() > 1e-20 else None
    dt = np.diff(t)
    velocity = np.diff(flat, axis=0) / dt[:, None]
    midtime = (t[1:] + t[:-1]) / 2
    acceleration = np.diff(velocity, axis=0) / np.diff(midtime)[:, None]
    # Test whether selected first/last frames span the sequence; it cannot prove endpoints are unmixed.
    direction = flat[-1] - flat[0]
    denom = float(direction @ direction)
    if denom > 1e-20:
        alpha = (flat - flat[0]) @ direction / denom
        fitted = flat[0] + alpha[:, None] * direction
        affine_residual = float(np.sqrt(np.mean((flat - fitted) ** 2)))
        coefficients = alpha.tolist()
    else:
        affine_residual, coefficients = None, None
    return {
        "frame_count": len(x),
        "timestamps": t.tolist(),
        "centered_singular_values": singular.tolist(),
        "centered_rank1_energy_fraction": leading,
        "velocity_rms_per_second": np.sqrt(np.mean(velocity**2, axis=1)).tolist(),
        "acceleration_rms_per_second_squared": np.sqrt(
            np.mean(acceleration**2, axis=1)
        ).tolist(),
        "first_last_affine_mix_residual_rms": affine_residual,
        "first_last_affine_mix_coefficients": coefficients,
        "admissible_as_facial_similarity_evidence": False,
        "reason": "any two-image linear crossfade has uncentered rank<=2 and centered rank<=1, irrespective of resemblance",
        "endpoint_warning": "sampled extrema are not certified unmixed source portraits",
    }


def inspect_video(
    path: str | Path,
    frame_count: int = 32,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    if not isinstance(frame_count, int) or not 3 <= frame_count <= 256:
        raise ObservationError("frame_count must be an integer in [3,256]")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ObservationError(f"cannot open video: {path}")
    try:
        fps, total = (
            float(cap.get(cv2.CAP_PROP_FPS)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if not np.isfinite(fps) or fps <= 0 or total < 3:
            raise ObservationError("video lacks usable frame count/FPS metadata")
        ids = np.unique(
            np.rint(np.linspace(0, total - 1, min(frame_count, total))).astype(int)
        )
        frames, records = [], []
        for index in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if not ok:
                raise ObservationError(f"failed to decode selected frame {index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if roi is not None:
                x, y, w, h = roi
                if (
                    min(x, y) < 0
                    or min(w, h) <= 0
                    or x + w > rgb.shape[1]
                    or y + h > rgb.shape[0]
                ):
                    raise ObservationError(
                        "ROI outside frame; rectangle is (x,y,width,height)"
                    )
                rgb = rgb[y : y + h, x : x + w]
            frames.append(rgb)
            records.append(
                {
                    "frame_index": int(index),
                    "timestamp_cfr_seconds": float(index / fps),
                    "decoded_timestamp_msec": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                    "rgb_array_sha256": array_digest(rgb),
                }
            )
        gray = np.stack(
            [
                cv2.resize(
                    cv2.cvtColor(f, cv2.COLOR_RGB2GRAY),
                    (64, 64),
                    interpolation=cv2.INTER_AREA,
                )
                / 255
                for f in frames
            ]
        )
        result = sequence_diagnostics(
            gray, np.asarray([r["timestamp_cfr_seconds"] for r in records])
        )
        from hashlib import sha256

        result.update(
            {
                "source_sha256": sha256(Path(path).read_bytes()).hexdigest(),
                "fps": fps,
                "source_frame_count": total,
                "sampling_records": records,
                "roi_xywh": roi,
                "scope": (
                    "full_screen_including_UI_not_anatomy"
                    if roi is None
                    else "explicit_ROI_not_verified_anatomy"
                ),
                "timing_assumption": "CFR frame_index/fps; decoded timestamps retained; VFR accuracy unvalidated",
            }
        )
        return result, frames
    finally:
        cap.release()
