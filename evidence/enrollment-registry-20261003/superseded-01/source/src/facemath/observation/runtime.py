"""Check the installed OpenCV namespace before native acquisition."""
from __future__ import annotations

import importlib.metadata
import cv2

from .contracts import ProviderUnavailable

OPENCV_DISTRIBUTIONS = (
    'opencv-python', 'opencv-contrib-python',
    'opencv-python-headless', 'opencv-contrib-python-headless',
)


def opencv_runtime() -> dict:
    installed = {}
    for name in OPENCV_DISTRIBUTIONS:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'distributions': installed, 'loaded_version': cv2.__version__,
            'module_path': cv2.__file__}


def require_supported_opencv() -> dict:
    """Refuse conflicting distributions and bindings outside the declared ABI.

    All four wheel variants write the same cv2 namespace. Metadata for one wheel
    cannot establish which binary won when multiple distributions are installed.
    """
    state = opencv_runtime()
    installed = state['distributions']
    message = ('native acquisition requires one opencv-contrib-python>=4.9,<5 '
               'distribution; create a fresh locked observer environment')
    if set(installed) != {'opencv-contrib-python'}:
        raise ProviderUnavailable(f'{message}; found {installed}')
    try:
        loaded = tuple(int(p) for p in state['loaded_version'].split('.'))
        declared = tuple(int(p) for p in installed['opencv-contrib-python'].split('.'))
    except ValueError as exc:
        raise ProviderUnavailable(f'{message}; unrecognized OpenCV version') from exc
    if len(loaded) < 3 or loaded[:2] < (4, 9) or loaded[0] != 4 or loaded[:3] != declared[:3]:
        raise ProviderUnavailable(f'{message}; loaded {state["loaded_version"]} differs from supported metadata {installed}')
    return state
