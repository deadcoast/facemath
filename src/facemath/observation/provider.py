"""Optional local MediaPipe observation and a strict, image-bound annotation path.

No implicit downloads. No synthetic landmarks on detection failure. Detector
thresholds are not per-landmark correctness probabilities. See docs/SOURCES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import importlib.metadata
import cv2
import numpy as np

from .anatomy import SEMANTIC_21
from .runtime import require_supported_opencv
from .contracts import (
    Observation, ObservationError, ProviderUnavailable, array_digest,
    canonical_json, object_digest, rgb_image,
)


class DetectionMiss(ObservationError):
    """Expected no-face/ambiguous-face outcome, distinct from invalid output."""
    def __init__(self, kind: str) -> None:
        if kind not in ("no_face", "multiple_faces"):
            raise ValueError("invalid detection outcome")
        self.kind = kind
        super().__init__(f"primary observation: {kind}; explicitly crop/select one face")


def mediapipe_fingerprint(model_hash: str, version: str) -> str:
    """The same exact recipe is usable before model execution and after it."""
    return object_digest({"provider": "mediapipe", "version": version,
        "model_sha256": model_hash, "mode": "IMAGE", "num_faces": 2,
        "detection_threshold": .5, "presence_threshold": .5, "augmentation_revision": 1})


@dataclass(frozen=True)
class Augmentation:
    label: str
    matrix: np.ndarray
    gamma: float = 1.0


def augmentations(shape: tuple[int, int]) -> tuple[Augmentation, ...]:
    h, w = shape
    center = ((w - 1) / 2, (h - 1) / 2)
    out = [Augmentation("identity", np.eye(3))]
    for label, angle, scale in (("rotate_-2", -2, 1), ("rotate_+2", 2, 1),
                                ("scale_.97", 0, .97), ("scale_1.03", 0, 1.03)):
        m = np.eye(3)
        m[:2] = cv2.getRotationMatrix2D(center, angle, scale)
        out.append(Augmentation(label, m))
    out.extend((Augmentation("gamma_.9", np.eye(3), .9), Augmentation("gamma_1.1", np.eye(3), 1.1)))
    return tuple(out)


def apply_points(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = np.c_[xy, np.ones(len(xy))] @ matrix.T
    return q[:, :2] / q[:, 2:3]


class MediaPipeProvider:
    """FaceLandmarker IMAGE mode; accepts exactly one face per independent view.

    IMAGE mode avoids temporal smoothing disguising instability. Inference is
    local after explicitly installing the optional package and model artifact.
    """

    def __init__(self, model_path: str | Path, expected_sha256: str | None = None,
                 noise_floor_px: float = .75, use_augmentations: bool = True) -> None:
        if not np.isfinite(noise_floor_px) or noise_floor_px <= 0:
            raise ObservationError("noise floor must be finite and positive")
        path = Path(model_path)
        if not path.is_file():
            raise ProviderUnavailable(f"local FaceLandmarker model not found: {path}")
        model_hash = sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and model_hash != expected_sha256:
            raise ProviderUnavailable("model SHA-256 mismatch")
        require_supported_opencv()
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ProviderUnavailable("install facemath[observer] and supply a local .task model") from exc
        self.mp = mp
        self.model_hash = model_hash
        self.noise_floor_px = noise_floor_px
        self.use_augmentations = use_augmentations
        self.version = importlib.metadata.version("mediapipe")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,  # detect ambiguity instead of silently taking a single result
            min_face_detection_confidence=.5, min_face_presence_confidence=.5,
            output_face_blendshapes=True,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.fingerprint = mediapipe_fingerprint(model_hash, self.version)

    def _detect(self, image: np.ndarray) -> Any:
        return self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                 data=np.ascontiguousarray(image)))

    def observe(self, image: np.ndarray) -> Observation:
        image = rgb_image(image)
        h, w = image.shape[:2]
        transforms = augmentations((h, w)) if self.use_augmentations else augmentations((h, w))[:1]
        samples, labels, failures, raw = [], [], [], []
        primary = None
        for augmentation in transforms:
            warped = cv2.warpAffine(image, augmentation.matrix[:2], (w, h),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if augmentation.gamma != 1:
                warped = np.rint(255 * (warped.astype(float) / 255) ** augmentation.gamma).astype(np.uint8)
            result = self._detect(warped)
            count = len(result.face_landmarks)
            if count != 1:
                reason = "no_face" if count == 0 else "multiple_faces"
                if augmentation.label == "identity":
                    raise DetectionMiss(reason)
                failures.append({"sample": augmentation.label, "reason": reason})
                continue
            landmarks = result.face_landmarks[0]
            if len(landmarks) != 478:
                raise ObservationError(f"expected 478 mesh landmarks; provider returned {len(landmarks)}")
            normalized = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)
            if not np.all(np.isfinite(normalized)):
                raise ObservationError("detector returned nonfinite coordinates")
            # Official normalized coordinates multiply by image width/height.
            points = apply_points(normalized[:, :2] * [w, h], np.linalg.inv(augmentation.matrix))
            if primary is None:
                primary = points
            samples.append(points)
            labels.append(augmentation.label)
            blendshapes = getattr(result, "face_blendshapes", None)
            raw_blendshapes = [] if not blendshapes else [
                {"category_name": getattr(c, "category_name", None),
                 "index": getattr(c, "index", None), "score": float(c.score)}
                for c in blendshapes[0]
            ]
            raw.append({"sample": augmentation.label, "normalized_xyz": normalized.tolist(),
                        "blendshapes": raw_blendshapes,
                        "source_to_augmented": augmentation.matrix.tolist(), "gamma": augmentation.gamma})
        assert primary is not None
        return Observation(
            array_digest(image), (h, w), "mediapipe478.v1", tuple(f"mp_{i:03}" for i in range(478)),
            primary, np.asarray(samples), "mediapipe", self.fingerprint,
            self.noise_floor_px, len(transforms), tuple(labels),
            ("mesh coordinates are model estimates; z is not metric depth",
             "augmentation stability does not establish anatomical accuracy"),
            canonical_json({"model_sha256": self.model_hash, "runtime_version": self.version,
                            "raw_detector_outputs": raw, "failed_augmentations": failures,
                            "per_landmark_confidence": None}),
        )

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> MediaPipeProvider:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def annotated_observation(image: np.ndarray, xy: np.ndarray, *, annotation_id: str,
                          sigma_px: float, names: tuple[str, ...] = SEMANTIC_21,
                          source_notes: tuple[str, ...] = ()) -> Observation:
    image = rgb_image(image)
    if not annotation_id.strip():
        raise ObservationError("manual annotation requires an explicit annotation_id")
    return Observation(array_digest(image), image.shape[:2], "semantic21.v1", names,
                       xy, np.asarray(xy)[None], "explicit_manual_annotation",
                       object_digest({"provider": "explicit_manual_annotation", "schema": "semantic21.v1", "revision": 1}),
                       sigma_px, 1, ("annotation",), source_notes,
                       canonical_json({"annotation_id": annotation_id,
                                       "sigma_kind": "operator_assumed_pixel_localization_error_not_measured"}))


def locate_faces(image: np.ndarray) -> list[dict[str, Any]]:
    """Offline coarse ROI proposals only. Never fabricates full mesh anatomy."""
    image = rgb_image(image)
    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not callable(getattr(cv2, "CascadeClassifier", None)) or not cascade_dir:
        raise ProviderUnavailable("OpenCV Haar API unavailable; install the supported opencv-contrib-python>=4.9,<5 dependency")
    path = Path(cascade_dir) / "haarcascade_frontalface_default.xml"
    if not path.is_file():
        raise ProviderUnavailable(f"OpenCV frontal face cascade asset not found: {path}")
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise ProviderUnavailable("OpenCV frontal face cascade unavailable")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(24, 24))
    return [{"xywh": [int(x) for x in box], "kind": "coarse_roi_proposal_not_landmarks",
             "detector_sha256": sha256(path.read_bytes()).hexdigest()} for box in boxes]
