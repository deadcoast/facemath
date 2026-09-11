"""478-point synthetic schema stress test, not a real detector validation."""
import numpy as np

from facemath.observation import FacialInstrument, InstrumentConfig, Observation
from facemath.observation.anatomy import MP_ANCHORS, SEMANTIC_21, anatomy_for
from facemath.observation.contracts import array_digest, object_digest
from facemath.observation.lab import phantom_points, render_phantom
from facemath.observation.provider import apply_points


def test_dense_478_schema_global_similarity_invariance_and_complete_raw_state():
    rng = np.random.default_rng(670)
    points = rng.uniform([82, 45], [174, 170], (478, 2))
    anchors = phantom_points()
    for name, xy in zip(SEMANTIC_21, anchors):
        points[MP_ANCHORS[name]] = xy
    image = render_phantom(anchors)
    transform = np.eye(3)
    angle = .04
    transform[:2, :2] = 1.01 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transform[:2, 2] = [1, -2]
    other = apply_points(points, transform)
    def observe(xy):
        return Observation(array_digest(image), image.shape[:2], 'mediapipe478.v1',
                           tuple(f'mp_{i:03}' for i in range(478)), xy, xy[None],
                           'synthetic_schema_stress_not_detector', object_digest('stress478'), .05)
    a, b = observe(points), observe(other)
    assert anatomy_for(a).point_count == 478
    result = FacialInstrument(InstrumentConfig(canvas_size=96, eye_span_px=32,
                                              uncertainty_draws=4, enable_flow=False)).run(image, image, a, b)
    values = {m.name: m.value for m in result.profile.measurements}
    assert values['geometry.all.rms'] < 1e-12
    assert result.arrays()['observation_a_xy'].shape == (478, 2)
    assert result.profile.relational_distortion.shape == (478, 478)
    assert np.max(np.abs(result.profile.relational_distortion)) < 1e-12
