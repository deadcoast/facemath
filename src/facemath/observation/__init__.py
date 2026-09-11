"""FACEMATH Wave 3: observation-driven, gauge-aware anatomical evidence."""
from .contracts import Observation, ObservationError, ProviderUnavailable
from .canonical import InstrumentConfig
from .instrument import FacialInstrument, InstrumentResult, InstrumentFailure
from .provider import MediaPipeProvider, annotated_observation

__all__ = ["Observation", "ObservationError", "ProviderUnavailable", "InstrumentConfig",
           "FacialInstrument", "InstrumentResult", "InstrumentFailure", "MediaPipeProvider", "annotated_observation"]
