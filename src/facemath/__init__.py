"""FACEMATH public API, with lazy backwards-compatible research exports.

Importing the package/version does not initialize every legacy numerical
subsystem. Requesting a legacy symbol still resolves its original class.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.10.1"

_EXPORTS = {
    "Frame": ".core",
    "LandmarkSet": ".core",
    "FacePair": ".core",
    "MetricResult": ".core",
    "EvidenceLedger": ".core",
    "AnalysisBundle": ".core",
    "GeometryPipeline": ".pipeline",
    "FacialGeometryEngine": ".wave2",
    "EngineConfig": ".wave2",
    "FacialState": ".state",
    "CorrespondenceField": ".state",
    "RegistrationDiagnostics": ".state",
    "SupportLevel": ".state",
    "FacialInstrument": ".observation",
    "InstrumentConfig": ".observation",
    "Observation": ".observation",
    "InstrumentFailure": ".observation",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
