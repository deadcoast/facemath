import pytest

from facemath.observation.cohort import Cohort, CohortRow
from facemath.observation.contracts import ObservationError, object_digest


def digest(value):
    return object_digest(value)


RECIPE = digest("recipe")


def rows(values=(0.1, 0.2, 0.2, 0.3)):
    return [
        CohortRow(
            f"p{i}",
            digest(f"a{i}"),
            digest(f"b{i}"),
            {"rms": v, "double": 2 * v, "constant": 1.0},
        )
        for i, v in enumerate(values)
    ]


def test_percentile_ties_use_midrank():
    cohort = Cohort("reference", RECIPE, rows(), kind="synthetic")
    result = cohort.calibrate(
        "rms", 0.2, RECIPE, (digest("queryA"), digest("queryB")), bootstrap_draws=30
    )
    assert result["percentile"] == 50
    assert result["cohort_kind"] == "synthetic"
    assert result["independent_components"] == 4


def test_cohort_roundtrip_preserves_fingerprint():
    cohort = Cohort("ref", RECIPE, rows())
    assert Cohort.from_dict(cohort.to_dict()).fingerprint == cohort.fingerprint


def test_mutable_input_cannot_modify_cohort_behind_hash():
    vals = {"rms": 1.0}
    row = CohortRow("x", digest("x1"), digest("x2"), vals)
    vals["rms"] = 999
    assert row.values["rms"] == 1
    with pytest.raises(TypeError):
        row.values["rms"] = 3


def test_recipe_mismatch_rejected():
    with pytest.raises(ObservationError, match="recipe"):
        Cohort("ref", RECIPE, rows()).calibrate(
            "rms", 0.2, digest("other"), (digest("q1"), digest("q2"))
        )


def test_query_image_leakage_rejected():
    with pytest.raises(ObservationError, match="leaks"):
        Cohort("ref", RECIPE, rows()).calibrate(
            "rms", 0.2, RECIPE, (digest("a1"), digest("new"))
        )


def test_subject_leakage_across_different_images_rejected():
    r = rows()
    r[0] = CohortRow(
        "p0", digest("a0"), digest("b0"), {"rms": 0.1}, subject_a="subjectA"
    )
    with pytest.raises(ObservationError, match="subject"):
        Cohort("ref", RECIPE, r).calibrate(
            "rms",
            0.2,
            RECIPE,
            (digest("q1"), digest("q2")),
            query_subjects=("subjectA",),
        )


def test_dependent_pairs_cannot_bootstrap_as_independent_rows():
    r = [
        CohortRow(f"p{i}", digest("shared"), digest(f"b{i}"), {"rms": float(i)})
        for i in range(5)
    ]
    result = Cohort("ref", RECIPE, r).calibrate(
        "rms", 2, RECIPE, (digest("q1"), digest("q2"))
    )
    assert result["independent_components"] == 1
    assert result["interval"] is None
    assert result["interval_unavailable_reason"] is not None


def test_duplicate_unordered_pairs_rejected():
    r = rows()
    a = r[0]
    r.append(CohortRow("new", a.image_b_sha256, a.image_a_sha256, {"rms": 0.2}))
    with pytest.raises(ObservationError, match="unordered"):
        Cohort("ref", RECIPE, r)


def test_metric_dependence_kept_visible_and_constant_correlation_missing():
    result = Cohort("ref", RECIPE, rows()).dependence()
    a = result["metrics"].index("double")
    b = result["metrics"].index("rms")
    c = result["metrics"].index("constant")
    assert result["spearman"][a][b] == pytest.approx(1)
    assert result["spearman"][c][b] is None


def test_bootstrap_is_seed_reproducible():
    cohort = Cohort("ref", RECIPE, rows())
    kwargs = dict(
        metric="rms",
        value=0.2,
        recipe_sha256=RECIPE,
        query_images=(digest("q1"), digest("q2")),
        bootstrap_draws=100,
    )
    assert cohort.calibrate(**kwargs) == cohort.calibrate(**kwargs)
