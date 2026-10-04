"""Validated observation records. Coordinates are pixels, ordered by semantic ID.

Frozen dataclasses alone do not freeze NumPy storage; all owned arrays here are
copied into immutable bytes-backed views. Joint augmentation samples are kept,
not reduced to independent landmark confidence scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Protocol
import json

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ObservationError(ValueError):
    """Input cannot support the requested observation."""


class ProviderUnavailable(RuntimeError):
    """Optional runtime or explicitly supplied model is unavailable."""


def frozen_array(value: Any, dtype: Any = np.float64) -> NDArray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ObservationError("arrays must contain finite values")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_digest(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def array_digest(value: NDArray) -> str:
    a = np.ascontiguousarray(value)
    head = canonical_json({"shape": list(a.shape), "dtype": a.dtype.str}).encode()
    return sha256(head + a.tobytes()).hexdigest()


def rgb_image(image: Any) -> NDArray[np.uint8]:
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ObservationError("images must be explicit RGB uint8 [H,W,3]; no inferred ranges")
    if min(a.shape[:2]) < 8:
        raise ObservationError("image dimensions must each be at least eight pixels")
    return frozen_array(a, np.uint8)


@dataclass(frozen=True)
class Observation:
    """One selected face, with a retained source coordinate system.

    `samples` are whole-shape observations mapped back to the source image.
    They estimate perturbation repeatability, NOT calibrated anatomical error.
    `noise_floor_px` is an explicit modeling assumption for sensitivity analysis.
    """

    image_sha256: str
    image_shape: tuple[int, int]
    schema: str
    names: tuple[str, ...]
    xy: FloatArray
    samples: FloatArray
    provider: str
    provider_fingerprint: str
    noise_floor_px: float = 0.75
    attempted_samples: int = 1
    sample_labels: tuple[str, ...] = ("source",)
    source_notes: tuple[str, ...] = ()
    metadata_json: str = "{}"
    correspondence: str | None = None

    def __post_init__(self) -> None:
        from .correspondence import require_correspondence
        require_correspondence(self.schema, self.correspondence)
        names = tuple(self.names)
        xy, samples = frozen_array(self.xy), frozen_array(self.samples)
        if len(set(names)) != len(names) or any(not n for n in names):
            raise ObservationError("landmark names must be unique and nonempty")
        if xy.shape != (len(names), 2) or len(names) < 3:
            raise ObservationError("xy must be [N,2], N>=3, matching names")
        if samples.ndim != 3 or samples.shape[1:] != xy.shape or len(samples) < 1:
            raise ObservationError("samples must be [K,N,2] with K>=1")
        if len(self.sample_labels) != len(samples):
            raise ObservationError("sample_labels must match successful samples")
        if self.attempted_samples < len(samples):
            raise ObservationError("attempted sample count cannot be smaller than successes")
        if not np.isfinite(self.noise_floor_px) or self.noise_floor_px <= 0:
            raise ObservationError("noise_floor_px must be finite and positive")
        if len(self.image_shape) != 2 or min(self.image_shape) < 8:
            raise ObservationError("image_shape must be [height,width]")
        for fingerprint in (self.image_sha256, self.provider_fingerprint):
            if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
                raise ObservationError("fingerprints must be full lowercase SHA-256 digests")
        metadata = json.loads(self.metadata_json)
        canonical_json(metadata)  # reject NaN/Infinity
        if not isinstance(metadata, dict):
            raise ObservationError("metadata must be a JSON object")
        if not self.schema or not self.provider:
            raise ObservationError("schema and provider are required")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "image_shape", tuple(self.image_shape))
        object.__setattr__(self, "sample_labels", tuple(self.sample_labels))
        object.__setattr__(self, "source_notes", tuple(self.source_notes))
        object.__setattr__(self, "metadata_json", canonical_json(metadata))

    @property
    def covariance(self) -> FloatArray:
        centered = self.samples - self.samples.mean(axis=0)
        c = np.einsum("kni,knj->nij", centered, centered) / max(len(centered) - 1, 1)
        c += np.eye(2)[None] * self.noise_floor_px**2
        return frozen_array(c)

    @property
    def in_bounds(self) -> NDArray[np.bool_]:
        h, w = self.image_shape
        return frozen_array(
            (self.xy[:, 0] >= 0) & (self.xy[:, 0] <= w - 1)
            & (self.xy[:, 1] >= 0) & (self.xy[:, 1] <= h - 1), bool
        )

    @property
    def digest(self) -> str:
        return object_digest(self.to_dict())

    def require_image(self, image: NDArray) -> None:
        if tuple(image.shape[:2]) != self.image_shape or array_digest(image) != self.image_sha256:
            raise ObservationError("observation is bound to a different image or coordinate frame")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "facemath.observation.v17" if self.correspondence is not None else "facemath.observation.v3",
            **({"correspondence": self.correspondence} if self.correspondence is not None else {}),
            "image_sha256": self.image_sha256,
            "image_shape": list(self.image_shape), "schema": self.schema,
            "names": list(self.names), "xy": self.xy.tolist(), "samples": self.samples.tolist(),
            "provider": self.provider, "provider_fingerprint": self.provider_fingerprint,
            "noise_floor_px": self.noise_floor_px, "attempted_samples": self.attempted_samples,
            "sample_labels": list(self.sample_labels), "source_notes": list(self.source_notes),
            "uncertainty_kind": "perturbation_repeatability_plus_assumed_noise_floor",
            "metadata": json.loads(self.metadata_json),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> Observation:
        if obj.get("format") not in ("facemath.observation.v3", "facemath.observation.v17"):
            raise ObservationError("unsupported observation format")
        if ((obj["format"] == "facemath.observation.v17" and obj.get("correspondence") is None)
                or (obj["format"] == "facemath.observation.v3" and "correspondence" in obj)):
            raise ObservationError("observation format requires its original correspondence identity")
        return cls(
            image_sha256=obj["image_sha256"], image_shape=tuple(obj["image_shape"]),
            schema=obj["schema"], names=tuple(obj["names"]), xy=obj["xy"], samples=obj["samples"],
            provider=obj["provider"], provider_fingerprint=obj["provider_fingerprint"],
            noise_floor_px=obj["noise_floor_px"], attempted_samples=obj["attempted_samples"],
            sample_labels=tuple(obj["sample_labels"]), source_notes=tuple(obj["source_notes"]),
            metadata_json=canonical_json(obj["metadata"]),
            correspondence=obj.get("correspondence"),
        )


class LandmarkProvider(Protocol):
    def observe(self, image: NDArray[np.uint8]) -> Observation:
        """Return one face or explicitly fail on absent/ambiguous selection."""
        ...
