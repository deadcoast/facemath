import cv2
import numpy as np

from facemath.core import FacePair, Frame, LandmarkSet
from facemath.correspondence import dense_correspondence_farneback
from facemath.regions import build_regions
from facemath.registration import (
    ransac_similarity_transform,
    weighted_similarity_transform,
)
from facemath.state import SupportLevel
from facemath.wave2 import EngineConfig, FacialGeometryEngine

NAMES = (
    "temple_left",
    "temple_right",
    "left_eye_outer",
    "left_eye_inner",
    "left_eye_top",
    "left_eye_bottom",
    "right_eye_inner",
    "right_eye_outer",
    "right_eye_top",
    "right_eye_bottom",
    "nose_bridge",
    "nose_tip",
    "left_nostril",
    "right_nostril",
    "mouth_left",
    "mouth_right",
    "upper_lip",
    "lower_lip",
    "chin",
    "jaw_left",
    "jaw_right",
)


def synthetic_landmarks():
    return np.array(
        [
            [28, 24],
            [72, 24],
            [33, 40],
            [43, 40],
            [38, 37],
            [38, 43],
            [57, 40],
            [67, 40],
            [62, 37],
            [62, 43],
            [50, 44],
            [50, 57],
            [45, 59],
            [55, 59],
            [42, 69],
            [58, 69],
            [50, 67],
            [50, 72],
            [50, 84],
            [32, 76],
            [68, 76],
        ],
        float,
    )


def render(points, shape=(104, 100)):
    img = np.zeros((*shape, 3), np.uint8) + 25
    for p in points:
        cv2.circle(img, tuple(np.round(p).astype(int)), 2, (220, 220, 220), -1)
    # structural lines create enough texture for dense correspondence
    cv2.polylines(
        img,
        [np.round(points[[2, 3, 6, 7]]).astype(np.int32)],
        False,
        (150, 150, 150),
        1,
    )
    cv2.line(
        img,
        tuple(points[10].astype(int)),
        tuple(points[11].astype(int)),
        (180, 180, 180),
        1,
    )
    cv2.line(
        img,
        tuple(points[14].astype(int)),
        tuple(points[15].astype(int)),
        (180, 180, 180),
        1,
    )
    return img


def test_weighted_similarity_ignores_zero_weight_outlier():
    a = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    b = 2 * a + np.array([3, -4])
    b[-1] = [999, 999]
    m, _, rmse = weighted_similarity_transform(a, b, [1, 1, 1, 0])
    pred = np.c_[a, np.ones(4)] @ m.T
    assert np.allclose(pred[:3, :2], 2 * a[:3] + [3, -4], atol=1e-8)
    assert rmse < 1e-8


def test_ransac_similarity_rejects_outlier():
    a = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], float) * 20
    b = a + np.array([5, -3])
    b[-1] = [300, 10]
    m, inliers, rmse = ransac_similarity_transform(a, b, threshold_px=1, max_trials=200)
    assert inliers.sum() >= 4
    assert rmse < 1e-8


def test_regions_from_named_landmarks():
    ls = LandmarkSet(synthetic_landmarks(), NAMES)
    regs = build_regions((104, 100), ls)
    assert {"central", "left_eye", "right_eye", "nose", "mouth", "jaw", "face"} <= set(
        regs
    )
    assert regs["central"].mask.sum() > 0


def test_dense_correspondence_cycle_consistency_translation():
    a = np.zeros((96, 96), np.float32)
    cv2.circle(a, (45, 45), 14, 1, -1)
    cv2.line(a, (20, 70), (75, 70), 1, 2)
    M = np.float32([[1, 0, 2], [0, 1, 1]])
    b = cv2.warpAffine(a, M, (96, 96))
    f = dense_correspondence_farneback(a, b, np.ones_like(a, bool), fb_threshold=2.5)
    valid = f.valid & (a > 0.2)
    assert valid.mean() > 0.005
    med = np.median(f.uv[valid], axis=0)
    assert np.allclose(med, [2, 1], atol=0.8)


def test_wave2_engine_shared_state_and_gates():
    p = synthetic_landmarks()
    th = 0.025
    s = 1.03
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = (s * (R @ p.T)).T + np.array([2.5, -1.0])
    # localized shape difference: jaw widens and chin lowers
    q[-2, 0] -= 2
    q[-1, 0] += 2
    q[-3, 1] += 1.5
    la = LandmarkSet(p, NAMES, np.ones(len(p)))
    lb = LandmarkSet(q, NAMES, np.ones(len(q)))
    pair = FacePair(Frame(render(p)), Frame(render(q)), la, lb)
    eng = FacialGeometryEngine(EngineConfig(correspondence_fb_threshold_px=3.0))
    state = eng.build_state(pair)
    assert state.registration.support != SupportLevel.INSUFFICIENT
    assert state.regions
    assert state.correspondence is not None
    bundle = eng.analyze_state(state)
    assert "procrustes_distance" in bundle.ledger.results
    assert "quality_correspondence_support" in bundle.ledger.results
    assert any(k.startswith("region.central.") for k in bundle.ledger.results)


def test_wave2_no_landmarks_explicitly_degraded():
    x = np.zeros((64, 64, 3), np.uint8)
    pair = FacePair(Frame(x), Frame(x.copy()))
    state = FacialGeometryEngine().build_state(pair)
    assert state.registration.support == SupportLevel.DEGRADED
    assert "no_landmarks_registration_unverified" in state.registration.warnings


def test_calibration_is_cohort_fingerprinted_and_uncertain():
    from facemath.calibration import BaselineRegistry
    from facemath.core import Direction, MetricResult, Provenance

    reg = BaselineRegistry()
    reg.add("d", np.linspace(0.1, 1.0, 40), "synthetic_unrelated", {"seed": 1})
    r = MetricResult(
        "d",
        0.2,
        "u",
        Direction.LOWER_IS_MORE_SIMILAR,
        "geometry",
        "x",
        Provenance("test"),
    )
    reg.calibrate(r, bootstrap_draws=100)
    assert r.calibrated_percentile is not None
    assert r.uncertainty is not None
    assert any(t.startswith("baseline_fingerprint:") for t in r.tags)


def test_report_refuses_uncalibrated_percentage_consensus():
    from facemath.report import build_report

    p = synthetic_landmarks()
    ls = LandmarkSet(p, NAMES, np.ones(len(p)))
    pair = FacePair(Frame(render(p)), Frame(render(p)), ls, ls)
    bundle = FacialGeometryEngine().run(pair)
    report = build_report(bundle)
    assert report.consensus_percentile is None
    assert any(
        "must not be presented as similarity percentages" in x
        for x in report.interpretation_limits
    )
