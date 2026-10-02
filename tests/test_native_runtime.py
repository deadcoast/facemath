"""Dependency collision regression; these fixtures do not claim model inference."""
import importlib.metadata

import pytest

from facemath.observation.contracts import ProviderUnavailable
from facemath.observation import runtime


@pytest.mark.parametrize('installed,loaded', [
    ({'opencv-python': '4.14.0.94', 'opencv-contrib-python': '5.0.0.93'}, '4.14.0'),
    ({'opencv-contrib-python': '4.14.0.94', 'opencv-contrib-python-headless': '4.14.0.94'}, '4.14.0'),
    ({'opencv-contrib-python': '5.0.0.93'}, '5.0.0'),
    ({'opencv-contrib-python': '4.14.0.94'}, '5.0.0'),
    ({'opencv-contrib-python': '4.14.0.94'}, '4.13.0'),
    ({'opencv-contrib-python': '4.8.0.76'}, '4.8.0'),
    ({'opencv-python': '4.14.0.94'}, '4.14.0'),
    ({}, '4.14.0'),
])
def test_ambiguous_or_unsupported_binding_refused(monkeypatch, installed, loaded):
    def version(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]
    monkeypatch.setattr(runtime.importlib.metadata, 'version', version)
    monkeypatch.setattr(runtime.cv2, '__version__', loaded)
    assert runtime.opencv_runtime()['distributions'] == installed
    with pytest.raises(ProviderUnavailable, match='fresh locked observer environment'):
        runtime.require_supported_opencv()


def test_current_environment_has_one_matching_supported_binding():
    state = runtime.require_supported_opencv()
    assert state['module_path']
    assert list(state['distributions']) == ['opencv-contrib-python']
