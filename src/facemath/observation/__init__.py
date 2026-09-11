"""FACEMATH Wave 3: observation-driven, gauge-aware anatomical evidence."""

from .canonical import InstrumentConfig
from .contracts import Observation, ObservationError, ProviderUnavailable
from .instrument import FacialInstrument, InstrumentFailure, InstrumentResult
from .provider import MediaPipeProvider, annotated_observation

__all__ = [
    "FacialInstrument",
    "InstrumentConfig",
    "InstrumentFailure",
    "InstrumentResult",
    "MediaPipeProvider",
    "Observation",
    "ObservationError",
    "ProviderUnavailable",
    "annotated_observation",
]
