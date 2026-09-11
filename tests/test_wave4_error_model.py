from dataclasses import replace

import numpy as np
import pytest
from test_wave4_reference import case

from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import ObservationError
from facemath.observation.error_model import (
    LocalizationCalibration,
    corrected_quantile,
    evaluate_coverage,
    fit_localization_calibration,
)
from facemath.observation.instrument import FacialInstrument
from facemath.observation.lab import phantom_observation, phantom_points, render_phantom
from facemath.observation.reference import mapped_prediction
from facemath.observation.validation import evaluate_study


def calibration(n=19, alpha=0.1):
    cases = tuple(case(i, delta=(0.1 * i, 0)) for i in range(n))
    report, rows = evaluate_study(cases)
    return fit_localization_calibration(
        rows,
        study_sha256=report["content_sha256"],
        alpha=alpha,
        domain="schematic-validation-only",
    ), rows


def test_conformal_rank_not_ordinary_numpy_quantile():
    q, k = corrected_quantile(np.arange(1, 10), 0.1)
    assert q == 9 and k == 9
    assert q != np.quantile(np.arange(1, 10), 0.9)


def test_too_few_groups_produces_unbounded_not_max_score():
    assert corrected_quantile(np.arange(1, 9), 0.1) == (None, 9)


def test_failed_calibration_group_kept_as_infinite_score():
    data = np.r_[np.arange(1, 9), np.inf]
    assert corrected_quantile(data, 0.1) == (None, 9)


@pytest.mark.parametrize(
    "scores,alpha", [([], 0.1), ([1, np.nan], 0.1), ([-1], 0.1), ([1], 0), ([1], 0.7)]
)
def test_invalid_calibration_data(scores, alpha):
    with pytest.raises(ObservationError):
        corrected_quantile(scores, alpha)


def test_group_max_not_landmark_pseudoreplication():
    cases = tuple(case(i, group=f"group-{i // 3}", delta=(i, 0)) for i in range(12))
    report, rows = evaluate_study(cases)
    c = fit_localization_calibration(
        rows, study_sha256=report["content_sha256"], alpha=0.1, domain="test"
    )
    assert len(c.group_scores) == 4 and c.quantile is None
    assert dict(c.group_scores)["group-0"] == pytest.approx(20)


def test_partial_labels_do_not_calibrate_unobserved_points():
    cases = [case(i, delta=(1, 0)) for i in range(9)]
    r = cases[-1].references[0]
    v = r.visible.copy()
    v[-1] = False
    p = r.xy.copy()
    p[-1] = 0
    cases[-1] = replace(cases[-1], references=(replace(r, xy=p, visible=v),))
    report, rows = evaluate_study(tuple(cases))
    c = fit_localization_calibration(
        rows, study_sha256=report["content_sha256"], alpha=0.1, domain="test"
    )
    assert c.quantile is None


def test_serialized_calibration_recomputes_order_statistic():
    c, _ = calibration()
    d = c.to_dict()
    clone = LocalizationCalibration.from_dict(d)
    assert clone.digest == c.digest
    d["quantile"] += 1
    with pytest.raises(ObservationError, match="order statistic"):
        LocalizationCalibration.from_dict(d)


@pytest.mark.parametrize(
    "what", ["image", "group", "domain", "recipe", "synthetic_scope"]
)
def test_query_rejects_out_of_scope_or_leaking_calibration(what):
    c, _ = calibration()
    query = case(100).observation
    group = "new"
    domain = c.domain
    if what == "image":
        query = case(0).observation
    if what == "domain":
        domain = "unrelated"
    elif what == "group":
        group = "group-0"
    if what == "recipe":
        query = replace(query, noise_floor_px=3)
    if what == "synthetic_scope":
        query = replace(query, provider="mediapipe")
        from facemath.observation.reference import observer_recipe

        c = replace(c, observer_recipe_sha256=observer_recipe(query))
    with pytest.raises(ObservationError):
        c.require_query(query, group, domain)


def test_heldout_coverage_is_separate_and_failure_aware():
    c, _ = calibration()
    _, rows = evaluate_study(
        (
            case(101, split="evaluation", delta=(0, 0)),
            case(102, split="evaluation", failure="scripted absence"),
        )
    )
    report = evaluate_coverage(c, rows)
    assert (
        report["evaluation_groups"] == 2 and report["empirical_group_coverage"] == 0.5
    )
    assert report["failures"] == 1


def test_evaluation_never_reuses_a_calibration_case():
    c, rows = calibration()
    with pytest.raises(ObservationError, match="leak"):
        evaluate_coverage(c, tuple(replace(r, split="evaluation") for r in rows))


def test_unbounded_envelope_is_reported_unscored_not_zero_coverage():
    c, _ = calibration(n=2)
    _, rows = evaluate_study((case(101, split="evaluation"),))
    report = evaluate_coverage(c, rows)
    assert report["empirical_group_coverage"] is None and not report["finite_envelope"]


def test_pair_error_envelope_inherits_same_state_and_contains_fixed_gauge_truth():
    c, _ = calibration()
    # Bias is smaller than the calibrated maximum. Two synthetic source pixels;
    # neither is a calibration image or declared group.
    true = phantom_points()
    a = render_phantom(true, seed=4000)
    b = render_phantom(true, seed=5000)
    oa = phantom_observation(a, true + [1, 0])
    ob = phantom_observation(b, true + [-1, 0])
    result = FacialInstrument(
        InstrumentConfig(uncertainty_draws=4, enable_flow=False, enable_edges=False),
        localization_calibration=c,
        query_groups=("queryA", "queryB"),
        domain=c.domain,
    ).run(a, b, oa, ob)
    e = result.localization
    assert e is not None and e.supported.all()
    from facemath.observation.provider import apply_points

    true_a = mapped_prediction(phantom_observation(a, true))[0]
    true_b = mapped_prediction(phantom_observation(b, true))[0]
    fixed_gap = (
        np.linalg.norm(
            apply_points(true_a, result.state.source_to_canonical_a)
            - apply_points(true_b, result.state.source_to_canonical_b),
            axis=1,
        )
        / result.state.normalization_span_px
    )
    assert np.all(e.lower_ipd <= fixed_gap + 1e-12) and np.all(
        e.upper_ipd >= fixed_gap - 1e-12
    )
    assert "localization_lower_ipd" in result.arrays()
    assert e.to_dict()["pair_union_bound"] == pytest.approx(0.8)
    assert all(r["status"] == "unresolved_at_tolerance" for r in e.to_dict()["rows"])
    assert e.calibration.reference_kind == "synthetic_fixture"


def test_assessment_does_not_change_uncalibrated_geometry():
    c, _ = calibration()
    p = phantom_points()
    a = render_phantom(p, seed=4000)
    b = render_phantom(p, seed=5000)
    oa = phantom_observation(a, p)
    ob = phantom_observation(b, p + [1, 0])
    cfg = InstrumentConfig(uncertainty_draws=4, enable_flow=False, enable_edges=False)
    raw = FacialInstrument(cfg).run(a, b, oa, ob)
    qualified = FacialInstrument(
        cfg, localization_calibration=c, query_groups=("qa", "qb"), domain=c.domain
    ).run(a, b, oa, ob)
    assert [m.to_dict() for m in raw.profile.measurements] == [
        m.to_dict() for m in qualified.profile.measurements
    ]


def test_public_calibration_api_rejects_split_leakage_even_without_study_runner():
    c, rows = calibration()
    _, heldout = evaluate_study((case(300, split="evaluation"),))
    with pytest.raises(ObservationError, match="leakage"):
        from facemath.observation.error_model import fit_localization_calibration

        fit_localization_calibration(
            rows + (replace(heldout[0], group_id=rows[0].group_id),),
            study_sha256="0" * 64,
            alpha=0.1,
            domain="test",
        )


def test_public_calibration_api_does_not_pseudoreplicate_rows():
    _, rows = calibration()
    with pytest.raises(ObservationError, match="unique"):
        fit_localization_calibration(
            rows + (rows[0],), study_sha256="0" * 64, alpha=0.1, domain="test"
        )


def test_rater_threshold_cannot_be_loosened_in_heldout_evaluation():
    c, _ = calibration()
    c = replace(c, minimum_raters=2)
    _, rows = evaluate_study((case(200, split="evaluation", delta=(0, 0)),))
    score = evaluate_coverage(c, rows)
    assert score["empirical_group_coverage"] == 0 and score["failures"] == 1


def test_rater_threshold_cannot_be_loosened_in_calibration():
    _, rows = calibration()
    c = fit_localization_calibration(
        rows, study_sha256="0" * 64, alpha=0.1, domain="test", minimum_raters=2
    )
    assert c.quantile is None and all(v is None for _, v in c.group_scores)


def test_query_lineage_is_checked_by_shared_pair_instrument():
    c, _ = calibration()
    c = replace(c, calibration_lineages=("declared-session-10",))
    p = phantom_points()
    a = render_phantom(p, seed=8000)
    b = render_phantom(p, seed=9000)
    from facemath.observation.instrument import InstrumentFailure

    with pytest.raises(InstrumentFailure, match="lineage leakage"):
        FacialInstrument(
            InstrumentConfig(
                uncertainty_draws=4, enable_edges=False, enable_flow=False
            ),
            localization_calibration=c,
            query_groups=("newA", "newB"),
            domain=c.domain,
            query_lineages=(("declared-session-10",), ()),
        ).run(a, b, phantom_observation(a, p), phantom_observation(b, p))
