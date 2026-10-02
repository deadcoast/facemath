"""Adapter contract tests use a fake detector, not claimed live ML inference."""
from pathlib import Path
from types import SimpleNamespace
import os
import numpy as np
import pytest

from facemath.observation.provider import MediaPipeProvider, augmentations, apply_points, annotated_observation
from facemath.observation.contracts import ObservationError, ProviderUnavailable, object_digest


class FakeDetector:
    def __init__(self, points, shape, counts=None):
        self.points=points
        self.shape=shape
        self.transforms=augmentations(shape)
        self.counts=counts or [1]*7
        self.calls=0
        self.closed=False
    def detect(self,image):
        index=self.calls;self.calls+=1
        h,w=self.shape
        q=apply_points(self.points,self.transforms[index].matrix)/[w,h]
        points=[SimpleNamespace(x=float(p[0]),y=float(p[1]),z=0.) for p in q]
        return SimpleNamespace(face_landmarks=[points]*self.counts[index])
    def close(self):
        self.closed=True


def fake_provider(counts=None,n=478):
    rng=np.random.default_rng(20)
    points=rng.uniform([20,20],[80,70],(n,2))
    provider=MediaPipeProvider.__new__(MediaPipeProvider)
    provider.mp=SimpleNamespace(Image=lambda **kw:kw,ImageFormat=SimpleNamespace(SRGB='SRGB'))
    provider.model_hash='f'*64
    provider.fingerprint=object_digest({'test':'fake-detector-contract'})
    provider.version='mock-contract-not-live'
    provider.noise_floor_px=.75
    provider.use_augmentations=True
    provider.detector=FakeDetector(points,(96,112),counts)
    return provider,points


def test_augmentation_samples_are_mapped_back_to_source_pixels():
    provider,points=fake_provider()
    obs=provider.observe(np.ones((96,112,3),np.uint8)*128)
    assert obs.samples.shape==(7,478,2)
    np.testing.assert_allclose(obs.samples,np.broadcast_to(points,obs.samples.shape),atol=2e-14)
    assert obs.attempted_samples==7
    assert obs.provider=='mediapipe'
    assert obs.names[33]=='mp_033'


@pytest.mark.parametrize('count,reason', [(0,'no_face'),(2,'multiple_faces')])
def test_primary_ambiguity_never_picks_a_convenient_face(count,reason):
    provider,_=fake_provider([count]+[1]*6)
    with pytest.raises(ObservationError,match=reason):
        provider.observe(np.zeros((96,112,3),np.uint8))


def test_augmentation_failures_are_retained_not_silently_erased():
    provider,_=fake_provider([1,0,1,2,1,1,1])
    obs=provider.observe(np.zeros((96,112,3),np.uint8))
    import json
    assert len(obs.samples)==5
    assert obs.attempted_samples==7
    assert len(json.loads(obs.metadata_json)['failed_augmentations'])==2


def test_incompatible_mesh_size_is_not_coerced():
    provider,_=fake_provider(n=468)
    with pytest.raises(ObservationError,match='478'):
        provider.observe(np.zeros((96,112,3),np.uint8))


def test_context_manager_closes_detector():
    provider,_=fake_provider()
    with provider:
        pass
    assert provider.detector.closed


def test_missing_model_does_not_trigger_download(tmp_path):
    with pytest.raises(ProviderUnavailable,match='not found'):
        MediaPipeProvider(tmp_path/'missing.task')
    assert not list(tmp_path.iterdir())


def test_wrong_model_hash_rejected_before_importing_optional_runtime(tmp_path):
    model=tmp_path/'dummy.task';model.write_bytes(b'not-a-real-model')
    with pytest.raises(ProviderUnavailable,match='SHA-256'):
        MediaPipeProvider(model,expected_sha256='0'*64)


@pytest.mark.live_observer
@pytest.mark.skipif(not (os.getenv('FACEMATH_MODEL') and os.getenv('FACEMATH_FACE')), reason='live MediaPipe requires local model and a supplied test face; no weights/runtime available in build environment')
def test_live_mediapipe_provider():
    import cv2
    bgr=cv2.imread(os.environ['FACEMATH_FACE'])
    assert bgr is not None
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    with MediaPipeProvider(os.environ['FACEMATH_MODEL']) as provider:
        obs=provider.observe(rgb)
    assert obs.xy.shape==(478,2)
    assert np.isfinite(obs.samples).all()


def test_provider_constructor_passes_documented_options_and_binds_model_hash(tmp_path, monkeypatch):
    """Exercises API assembly with a fake runtime; not a live detector result."""
    import importlib.metadata
    import sys
    from hashlib import sha256
    seen = {}
    detector = FakeDetector(np.random.default_rng(21).uniform(20, 60, (478, 2)), (96, 112))
    def capture_options(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(**kwargs)
    runtime = SimpleNamespace(
        Image=lambda **kwargs: kwargs,
        ImageFormat=SimpleNamespace(SRGB='srgb'),
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE='image-mode'),
                FaceLandmarkerOptions=capture_options,
                FaceLandmarker=SimpleNamespace(create_from_options=lambda options: detector),
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, 'mediapipe', runtime)
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: 'mock-runtime' if name == 'mediapipe' else original(name))
    model = tmp_path / 'fake-api-test.task'
    model.write_bytes(b'API fixture, not model weights')
    with MediaPipeProvider(model) as provider:
        observation = provider.observe(np.zeros((96, 112, 3), np.uint8))
    assert seen['num_faces'] == 2
    assert seen['running_mode'] == 'image-mode'
    assert seen['base_options'].model_asset_path == str(model)
    assert observation.samples.shape == (7, 478, 2)
    assert provider.model_hash == sha256(model.read_bytes()).hexdigest()
    assert detector.closed


def test_absent_optional_runtime_is_explicit_not_a_fallback(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'mediapipe', None)
    model = tmp_path / 'fake.task'
    model.write_bytes(b'not model weights')
    with pytest.raises(ProviderUnavailable, match='facemath'):
        MediaPipeProvider(model)


def test_offline_coarse_locator_on_blank_raster_returns_no_landmark_claim():
    from facemath.observation.provider import locate_faces
    assert locate_faces(np.zeros((96, 112, 3), np.uint8)) == []


@pytest.mark.parametrize('missing',['api','data','asset'])
def test_coarse_locator_missing_prerequisites_are_explicit(tmp_path,monkeypatch,missing):
    from facemath.observation import provider
    if missing=='api':
        monkeypatch.delattr(provider.cv2,'CascadeClassifier',raising=False)
    elif missing=='data':
        monkeypatch.delattr(provider.cv2,'data',raising=False)
    else:
        monkeypatch.setattr(provider.cv2.data,'haarcascades',str(tmp_path))
    with pytest.raises(ProviderUnavailable,match='Haar API|asset not found'):
        provider.locate_faces(np.zeros((96,112,3),np.uint8))
