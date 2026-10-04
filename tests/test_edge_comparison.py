"""Shared-ray displacement controls use raster edges with known locations."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import ObservationError
from facemath.observation.edge_support import (EDGE_REGIONS, PairedEdgeProfile,
    diagnose_edges, sample_edge_profile)
from facemath.observation.instrument import FacialInstrument
from facemath.observation.lab import phantom_points, phantom_observation, render_phantom
from facemath.observation.readout import pair_readout, render_edge_comparison, terminal_readout


def step(x):
    image = np.zeros((128, 128, 3), np.uint8)
    image[:, x:] = 255
    return image


def sampled(image, support=None, points=None):
    points = np.c_[np.full(8, 64.), np.linspace(25, 100, 8)] if points is None else points
    normals = np.tile([1., 0.], (len(points), 1))
    support = np.ones((128, 128), bool) if support is None else support
    return sample_edge_profile(image, support, points, normals, radius_px=8)


def paired(a, b, **kwargs):
    return PairedEdgeProfile(a, b, 40., .05, .025, .85, **kwargs)


def test_shared_raster_translation_sign_units_and_immutability():
    a, b = sampled(step(63)), sampled(step(66))
    result = paired(a, b)
    np.testing.assert_array_equal(result.delta_b_minus_a_px, np.full(8, 3.))
    np.testing.assert_array_equal(result.displacement_b_minus_a_px, np.tile([3., 0.], (8, 1)))
    summary = result.summary()
    assert summary['canonical_pixels']['rms_offset'] == 3
    assert summary['eye_spans']['mean_signed_offset'] == .075
    assert summary['within_tolerance_fraction_range'] == [0, 0]
    np.testing.assert_array_equal(paired(b, a).delta_b_minus_a_px, -result.delta_b_minus_a_px)
    for array in result.arrays().values():
        assert not array.flags.writeable
        with pytest.raises(ValueError): array.setflags(write=True)


def test_identical_visible_edges_are_zero_but_flat_images_are_unavailable():
    visible = paired(sampled(step(63)), sampled(step(63))).summary()
    assert visible['canonical_pixels']['rms_offset'] == 0
    assert visible['within_tolerance_fraction_range'] == [1, 1]
    flat = np.full((128, 128, 3), 120, np.uint8)
    absent = paired(sampled(flat), sampled(flat)).summary()
    assert not absent['available'] and absent['supported'] == 0 and absent['unresolved'] == 8
    assert absent['canonical_pixels']['rms_offset'] is None
    assert absent['within_tolerance_fraction_range'] == [0, 1]
    assert absent['rejections']['a']['weak_gradient'] == 8


def test_opposite_shifts_do_not_cancel_rms():
    points = np.array([[64., 32.], [64., 96.]])
    a = step(64)
    b = np.zeros_like(a); b[:64, 66:] = 255; b[64:, 62:] = 255
    result = paired(sampled(a, points=points), sampled(b, points=points)).summary()
    assert result['canonical_pixels']['mean_signed_offset'] == 0
    assert result['canonical_pixels']['rms_offset'] == 2
    assert result['canonical_pixels']['p90_absolute_offset'] == 2


@pytest.mark.parametrize('kind,reason', [('ambiguous','ambiguous_peak'),
    ('padding','source_unsupported'), ('boundary','search_boundary')])
def test_rejected_candidates_are_not_zero_agreement(kind, reason):
    image = step(62); support = np.ones((128, 128), bool)
    if kind == 'ambiguous': image[:, 68:] = 0
    if kind == 'padding': support[:, 68:] = False
    if kind == 'boundary': image = step(73)
    result = paired(sampled(step(64)), sampled(image, support)).summary()
    assert result['supported'] == 0 and result['total'] == 8
    assert result['canonical_pixels']['rms_offset'] is None
    assert result['rejections']['b'][reason] == 8
    for counts in result['rejections'].values(): assert sum(counts.values()) == 8


def test_partial_support_keeps_requested_denominator_and_interval():
    points = np.array([[64., 32.], [64., 96.]])
    b = step(64); b[64:] = 120
    result = paired(sampled(step(64), points=points), sampled(b, points=points)).summary()
    assert result['supported'] == 1 and result['total'] == 2
    assert result['within_tolerance'] == 1
    assert result['within_tolerance_fraction_range'] == [.5, 1.]


@pytest.mark.parametrize('field', ['points','normals','offsets'])
def test_independently_parameterized_rays_cannot_be_subtracted(field):
    original = sampled(step(64))
    other = replace(original, **{field: getattr(original,field)+.1})
    with pytest.raises(ObservationError,match='identical ray'): paired(original,other)


@pytest.mark.parametrize('scale', [0., float('nan'), float('inf'), True])
def test_invalid_normalization_is_rejected(scale):
    profile = sampled(step(64))
    with pytest.raises(ObservationError): PairedEdgeProfile(profile,profile,scale,.05,.025,.85)


def test_missing_anchor_retains_rows_and_other_anchor_uses_shared_coordinates():
    config = InstrumentConfig(edge_samples=8, edge_radius_ipd=.2)
    xy = np.array([[64., 25.], [64., 100.]])
    state = SimpleNamespace(config=config,normalization_span_px=40.,
        image_a=step(64),image_b=step(67),xy_a=xy,xy_b=xy+np.array([2.,0.]),
        valid_source_a=np.ones((128,128),bool),valid_source_b=np.ones((128,128),bool),
        observation_a=SimpleNamespace(in_bounds=np.array([True,False])),
        observation_b=SimpleNamespace(in_bounds=np.ones(2,bool)),
        anatomy=SimpleNamespace(contours={name:(0,1) for name in EDGE_REGIONS}))
    edges = diagnose_edges(state); rows = edges.paired_summary()['rows']
    assert len(rows)==8
    assert all(r['total']==8 and r['supported']==0 and r['unresolved']==8 for r in rows[:4])
    assert all(r['reason']=='contour_has_unobservable_source_vertices' for r in rows[:4])
    assert all(r['supported']==8 for r in rows[4:])
    for pair in edges.pairs.values():
        np.testing.assert_array_equal(pair.a.points,pair.b.points)
        # B minus A is +3 x even on B-anchored rays with opposite normal signs.
        np.testing.assert_array_equal(pair.displacement_b_minus_a_px,np.tile([3.,0.],(8,1)))


def test_production_pair_exports_readout_without_changing_geometry():
    points = phantom_points(); a=render_phantom(points); b=render_phantom(points,seed=8)
    oa,ob=phantom_observation(a,points),phantom_observation(b,points)
    config=InstrumentConfig(canvas_size=96,eye_span_px=32,uncertainty_draws=4,enable_flow=False)
    result=FacialInstrument(config).run(a,b,oa,ob)
    disabled=FacialInstrument(replace(config,enable_edges=False)).run(a,b,oa,ob)
    for name,array in disabled.raw_geometry_arrays().items():
        np.testing.assert_array_equal(array,result.raw_geometry_arrays()[name],err_msg=name)
    data=pair_readout(result.evidence()); comparison=data['edge_comparison']
    assert len(comparison['rows'])==8
    assert 'edge_pair_a_left_eye_delta_b_minus_a_px' in result.arrays()
    assert pair_readout(disabled.evidence())['edge_comparison'] is None
    html=render_edge_comparison(comparison)
    assert 'Image-edge comparison table' in html and 'tabindex="0"' in html
    assert html.count('<th scope="row">')==8
    text=terminal_readout({**data,'report':'result/report.html'},encoding='cp1252')
    assert 'Image-edge displacement' in text and 'unresolved=' in text


def test_terminal_escapes_path_glyphs_without_changing_their_identity():
    data={'kind':'observer_study','enrollment':{'committed':1,'eligible':1,'prospectively_excluded':0},
          'report':'C:/portraits/a–◆│/report.html'}
    text=terminal_readout(data,encoding='cp1252')
    assert r'C:/portraits/a–\u25c6\u2502/report.html' in text
