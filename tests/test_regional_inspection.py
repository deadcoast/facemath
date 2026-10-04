from dataclasses import replace
import json

import cv2
import numpy as np
import pytest

from facemath.observation import FacialInstrument, InstrumentConfig
from facemath.observation.artifacts import write_bundle, verify_bundle
from facemath.observation.lab import phantom_points, phantom_observation, render_phantom
from facemath.observation.readout import pair_readout, render_pair_readout
from facemath.observation.regional import regional_pixel_support
from facemath.observation.regional_view import regional_inspection, regional_map_images, MAP_SCALES


@pytest.fixture
def pair():
    p = phantom_points(); q = p.copy(); q[19, 0] -= 7; q[20, 0] += 9
    a, b = render_phantom(p), render_phantom(q)
    result = FacialInstrument(InstrumentConfig(canvas_size=96, eye_span_px=32,
        uncertainty_draws=4, enable_edges=False, enable_flow=False)).run(
            a, b, phantom_observation(a, p), phantom_observation(b, q))
    return result, a, b


def test_inspection_retains_exact_points_residuals_and_fixed_denominators(pair):
    result, _, _ = pair
    before = {k: v.copy() for k, v in result.arrays().items()}
    data = regional_inspection(result)
    for row, expected in zip(data['rows'], result.profile.regional.rows):
        assert len(row['landmarks']) == expected.displacement.total
        for point in row['landmarks']:
            i = point['index']
            assert point['name'] == result.state.observation_a.names[i]
            np.testing.assert_array_equal(point['a'], result.state.xy_a[i])
            np.testing.assert_array_equal(point['b'], result.state.xy_b[i])
            np.testing.assert_array_equal(point['delta'], (result.state.xy_b[i]-result.state.xy_a[i])/result.state.normalization_span_px)
            np.testing.assert_array_equal(point['centered_delta'], np.array(point['delta'])-expected.displacement.shift_b_minus_a)
        centered = np.array([p['centered_delta'] for p in row['landmarks']])
        assert np.sqrt(np.mean(np.sum(centered**2, axis=1))) == pytest.approx(expected.displacement.centered_rms)
        assert sum(t['pixels'] for t in row['triangles']) == expected.deformation.supported_pixels
        assert row['unresolved_pixels'] + sum(t['pixels'] for t in row['triangles']) == expected.deformation.total_pixels
    list(regional_map_images(result))
    for key, value in before.items():
        np.testing.assert_array_equal(result.arrays()[key], value, err_msg=key)


def test_maps_and_triangle_weights_use_exact_existing_pixel_support(pair):
    result, _, _ = pair
    images = dict(regional_map_images(result))
    polar = result.profile.regional.polar
    for row in regional_inspection(result)['rows']:
        selected = regional_pixel_support(result.state, result.field, polar, row['region'], result.profile.regional.landmark_supported)
        ids, weights = np.unique(result.field.simplex_index[selected], return_counts=True)
        assert [t['index'] for t in row['triangles']] == ids.tolist()
        assert [t['pixels'] for t in row['triangles']] == weights.tolist()
        for t in row['triangles']:
            i = t['index']
            assert t['rotation_deg'] == np.rad2deg(polar.rotation_rad[i])
            assert t['area_ratio'] == np.exp(polar.log_area[i])
            assert t['anisotropy'] == pytest.approx(np.linalg.norm(polar.deviatoric_log_stretch[i]), abs=1e-14)
        outside = ~result.state.masks_a['face' if row['region'] == 'all' else row['region']]
        for m in row['maps']:
            image = images[m['image']]
            assert image.dtype == np.uint8 and image.shape == result.state.image_a.shape
            np.testing.assert_array_equal(image[outside], np.rint(result.state.image_a[outside]*.35).astype(np.uint8))


def test_color_saturation_never_clips_exported_values(pair):
    result, _, _ = pair
    regional = result.profile.regional
    polar = replace(regional.polar, rotation_rad=np.full_like(regional.polar.rotation_rad, np.pi/2))
    altered = replace(result, profile=replace(result.profile, regional=replace(regional, polar=polar)))
    data = regional_inspection(altered)
    for row, expected in zip(data['rows'], regional.rows):
        assert row['maps'][0]['above_scale_pixels'] == expected.deformation.supported_pixels
        assert all(t['rotation_deg'] == 90 for t in row['triangles'])
    assert [m['high'] for m in data['rows'][0]['maps']] == [m['high'] for m in MAP_SCALES]


def test_missing_landmarks_and_unavailable_maps_keep_their_slots(pair):
    result, _, _ = pair
    regional = result.profile.regional
    supported = regional.landmark_supported.copy()
    ids = np.flatnonzero(regional.membership[3]); supported[ids] = False
    row = regional.rows[3]
    row = replace(row, displacement=replace(row.displacement, supported=0, shift_b_minus_a=None),
                  deformation=replace(row.deformation, supported_pixels=0,
                                      unavailable_reason='fewer_than_three_supported_region_landmarks'))
    rows = list(regional.rows); rows[3] = row
    changed = replace(result, profile=replace(result.profile, regional=replace(regional, rows=tuple(rows), landmark_supported=supported)))
    data = regional_inspection(changed)['rows'][3]
    assert len(data['landmarks']) == row.displacement.total
    assert not data['maps_available'] and not data['triangles']
    assert all(p['a'] is p['b'] is p['delta'] is p['within_tolerance'] is None for p in data['landmarks'])
    assert not any(name.startswith('regional_mouth_') for name, _ in regional_map_images(changed))
    html = render_pair_readout(changed.evidence())
    assert 'Deformation maps unavailable: fewer than three supported region landmarks' in html


def test_bundle_maps_data_and_portable_html_are_integrated(pair, tmp_path):
    result, a, b = pair
    root = write_bundle(result, tmp_path/'pair', a, b)
    assert verify_bundle(root)['verified']
    evidence = json.loads((root/'evidence.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    assert summary['regional_inspection'] == evidence['regional_inspection']
    assert pair_readout(evidence)['regional_inspection'] == evidence['regional_inspection']
    assert '<section id="regional">' in (root/'report.html').read_text()
    app = render_pair_readout(evidence, image_prefix='runs/id/result/')
    assert app.count('data-region="') == 7
    for name, image in regional_map_images(result):
        assert f'runs/id/result/{name}' in app
        decoded = cv2.cvtColor(cv2.imdecode(np.frombuffer((root/name).read_bytes(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded, image)


def test_older_bundles_still_render_the_existing_regional_summary(pair):
    result, _, _ = pair
    evidence = result.evidence(); evidence.pop('regional_inspection')
    assert 'regional_inspection' not in pair_readout(evidence)
    html = render_pair_readout(evidence)
    assert html.count('data-region-row=') == 7
    assert 'data-region="' not in html
