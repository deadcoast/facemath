"""Freeze-selected estimation, held-out qualification and prospective slices.

The selected estimator is specified before acquisition. Reporting both the raw
baseline and centered score is diagnostic, never an automatic winner-selection
procedure on evaluation data. Slice summaries do not inherit conditional
conformal guarantees from an aggregate marginal statement.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from .bias import BiasCalibration, fit_bias_calibration
from .joint import JointCalibration, fit_joint_calibration
from .contracts import ObservationError, object_digest
from .error_model import evaluate_coverage, fit_localization_calibration
from .reference import REFERENCE_NAMES
from .study import FrozenStudy
from .validation import LocalizationRow, summarize


def rows_from_report(report: dict[str, Any]) -> tuple[LocalizationRow, ...]:
    names = {f.name for f in fields(LocalizationRow)}
    rows = []
    for value in report["cases"]:
        data = {k: value[k] for k in names}
        for key in ("lineage_ids", "reference_digests"):
            data[key] = tuple(data[key])
        rows.append(LocalizationRow(**data))
    return tuple(rows)


def binomial_interval(successes: int, total: int, alpha: float = .05) -> tuple[float, float] | None:
    """Clopper-Pearson interval; its interpretation requires independent groups."""
    if (type(successes) is not int or type(total) is not int or not 0 <= successes <= total
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ObservationError("invalid binomial counts or error rate")
    if total == 0:
        return None
    lower = 0. if successes == 0 else float(beta_distribution.ppf(alpha/2, successes, total-successes+1))
    upper = 1. if successes == total else float(beta_distribution.ppf(1-alpha/2, successes+1, total-successes))
    return lower, upper


def _slice_coverage(rows: tuple[LocalizationRow, ...], quantile: float | None,
                    minimum_raters: int) -> dict[str, Any]:
    groups: dict[str, bool] = {}
    counts: dict[str, int] = {}
    for r in rows:
        ready = bool(r.simultaneous_score is not None and np.all(r.reference_visible)
                     and np.all(r.rater_count >= minimum_raters))
        covered = ready and quantile is not None and r.simultaneous_score <= quantile
        groups[r.group_id] = groups.get(r.group_id, True) and covered
        counts[r.group_id] = counts.get(r.group_id, 0) + 1
    n = len(groups)
    hit = sum(groups.values()) if quantile is not None else None
    return {"cases": len(rows), "groups": n, "covered_groups": hit,
            "empirical_coverage": hit/n if n and hit is not None else None,
            "binomial_95_interval": binomial_interval(hit, n) if hit is not None else None,
            "case_count_per_group": counts,
            "scope": "all scored reference points in this slice's cases per group; overlapping slices are not pooled",
            "interval_assumption": "independent comparable Bernoulli groups; not implied by marginal conformal exchangeability"}


def qualify(study: FrozenStudy, raw_report: dict[str, Any], session_status: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], Any]:
    rows = rows_from_report(raw_report)
    data = study.data
    if {r.case_id for r in rows} != {c["case_id"] for c in study.included}:
        raise ObservationError("qualification rows differ from frozen eligible inventory")
    baseline = fit_localization_calibration(rows, study_sha256=raw_report["content_sha256"],
        domain=data["domain"], alpha=data["alpha"], minimum_raters=data["minimum_raters"])
    baseline_coverage = evaluate_coverage(baseline, rows)
    bias = None
    arrays: dict[str, np.ndarray] = {}
    if data["bias_mode"] == "development_joint_ocular":
        selected, scored = fit_joint_calibration(rows, study_sha256=raw_report["content_sha256"],
            domain=data["domain"], alpha=data["alpha"], minimum_raters=data["minimum_raters"],
            minimum_development_groups=data["minimum_development_groups"], **data["joint_error"])
        bias = selected.bias
        coverage = evaluate_coverage(selected.residual, scored)
        arrays.update(development_group_bias_vectors=bias.group_vectors, development_bias_beta=bias.beta,
            development_joint_group_second_moments=selected.scatter.group_second_moments,
            development_joint_covariance=selected.scatter.covariance,
            development_joint_group_case_counts=np.asarray(selected.scatter.group_case_counts, dtype=np.int64))
    elif data["bias_mode"] == "development_ocular":
        selected, scored = fit_bias_calibration(rows, study_sha256=raw_report["content_sha256"],
            domain=data["domain"], alpha=data["alpha"], minimum_raters=data["minimum_raters"],
            minimum_development_groups=data["minimum_development_groups"])
        bias = selected.bias
        coverage = evaluate_coverage(selected.residual, scored)
        arrays["development_group_bias_vectors"] = bias.group_vectors
        arrays["development_bias_beta"] = bias.beta
    else:
        selected, scored, coverage = baseline, tuple(r for r in rows if r.split != "development"), baseline_coverage
    source_bias = np.zeros((len(rows), 8, 2))
    centered = np.zeros_like(source_bias)
    centered_valid = np.zeros((len(rows), 8), dtype=bool)
    for i, r in enumerate(rows):
        if np.all(r.observed[:4]):
            try:
                source_bias[i] = bias.vectors(r.predicted_xy) if bias is not None else 0.
            except ObservationError:
                continue
            centered_valid[i] = r.observed
            centered[i, r.observed] = r.error_vectors_px[r.observed] - source_bias[i, r.observed]
    arrays.update(raw_error_vectors_px=np.stack([r.error_vectors_px for r in rows]),
        source_bias_vectors_px=source_bias, centered_error_vectors_px=centered,
        centered_error_valid=centered_valid, source_predicted_xy=np.stack([r.predicted_xy for r in rows]),
        source_reference_xy=np.stack([r.reference_xy for r in rows]),
        covariance_px2=np.stack([r.covariance for r in rows]))
    scored_by_id = {r.case_id: r for r in scored}
    by_case = {c["case_id"]: c for c in study.cases}
    per_case = []
    receipt_by_id = {r["case_id"]: r for r in session_status["receipts"]}
    case_indices = {c["case_id"]: i for i, c in enumerate(study.cases)}
    for i, r in enumerate(rows):
        entry = by_case[r.case_id]
        observed = r.observed
        corrected_error = np.linalg.norm(centered[i, observed], axis=1) if np.all(centered_valid[i, observed]) and observed.any() else []
        per_case.append({"case_id": r.case_id, "group_id": r.group_id, "split": r.split,
            "stratum": entry["stratum"], "attempt_outcome": receipt_by_id[r.case_id]["status"],
            "raw_mean_error_px": float(r.error_norm_px[observed].mean()) if observed.any() else None,
            "centered_mean_error_px": float(np.mean(corrected_error)) if len(corrected_error) else None,
            "raw_simultaneous_score": r.simultaneous_score,
            "selected_simultaneous_score": scored_by_id[r.case_id].simultaneous_score if r.case_id in scored_by_id else None,
            "development_not_heldout": r.split == "development", "observed_reference_points": int(observed.sum()),
            "image": "acquisition/study/"+entry["image"],
            "receipt": f"acquisition/cases/case_{case_indices[r.case_id]:05d}/receipt.json"})
    slices = []
    for name in data["strata"]:
        cal = tuple(r for r in scored if r.split == "calibration" and by_case[r.case_id]["stratum"] == name)
        evaluation = tuple(r for r in scored if r.split == "evaluation" and by_case[r.case_id]["stratum"] == name)
        slices.append({"stratum": name, "prespecified": True, "calibration_groups": len({r.group_id for r in cal}),
            "has_calibration_representation": bool(cal),
            "evaluation": _slice_coverage(evaluation, selected.quantile, data["minimum_raters"]),
            "raw_localization": summarize(evaluation) if evaluation else None,
            "conditional_coverage_guaranteed": False})
    coverage["full_error_model_sha256"] = selected.digest
    coverage["score_centering"] = data["bias_mode"]
    coverage["binomial_95_interval"] = binomial_interval(coverage["covered_groups"], coverage["evaluation_groups"]) if coverage["covered_groups"] is not None else None
    per_landmark = []
    for k, name in enumerate(REFERENCE_NAMES):
        eligible = np.array([r.split == "evaluation" for r in rows]) & centered_valid[:, k]
        per_landmark.append({"name": name, "observed_evaluation_cases": int(eligible.sum()),
            "raw_mean_error_px": float(np.linalg.norm(arrays["raw_error_vectors_px"][eligible, k], axis=1).mean()) if eligible.any() else None,
            "centered_mean_error_px": float(np.linalg.norm(centered[eligible, k], axis=1).mean()) if eligible.any() else None})
    warnings = []
    if selected.quantile is None:
        warnings.append("Selected reference envelope is unbounded; it cannot qualify a pair.")
    if any(s["evaluation"]["cases"] and not s["has_calibration_representation"] for s in slices):
        warnings.append("At least one evaluation stratum has no calibration groups: no deployment coverage transfer is justified for that stratum.")
    if any(session_status["outcomes"][k] for k in ("timeout", "interrupted", "runtime_error")):
        warnings.append("Infrastructure failures are retained as unavailable outputs, not mislabeled as no-face detections.")
    if coverage["empirical_group_coverage"] is not None and coverage["empirical_group_coverage"] < 1-data["alpha"]:
        warnings.append("Empirical aggregate coverage is below the nominal target; this is reported, not repaired using evaluation labels.")
    if study.observer.execution_kind == "synthetic_fixture":
        warnings.append("Every observation in this study is a controlled synthetic fixture, not live human-landmark inference.")
    report = {"format": "facemath.qualification.v5", "study_sha256": study.digest,
        "raw_validation_sha256": raw_report["content_sha256"], "observer_recipe_sha256": study.observer.recipe_sha256,
        "execution_kind": study.observer.execution_kind,
        "live_human_observer_validation": study.observer.execution_kind == "native_mediapipe" and session_status["outcomes"]["observed"] > 0,
        "domain": data["domain"], "bias_mode_selected_before_acquisition": data["bias_mode"],
        "raw_coordinates_changed": False, "estimator_selected_using_evaluation": False,
        "enrollment": {k: session_status[k] for k in ("enrolled", "eligible", "prospectively_excluded", "committed", "outcomes")},
        "excluded_cases": [{k: c[k] for k in ("case_id", "group_id", "split", "stratum", "exclusion_reason")}
                            for c in study.cases if c["eligibility"] == "excluded"],
        "raw_baseline": {"quantile": baseline.quantile, "rank": baseline.rank, "coverage": baseline_coverage},
        "selected_error_model": {"calibration_sha256": selected.digest, "quantile": selected.quantile,
            "rank": selected.residual.rank if isinstance(selected, BiasCalibration) else selected.rank,
            "bias_sha256": bias.digest if bias else None,
            "joint_scatter_sha256": selected.scatter.digest if isinstance(selected, JointCalibration) else None,
            "error_geometry": "full16_shared_latent_ellipsoid" if isinstance(selected, JointCalibration) else "marginal_product_ellipses",
            "q_comparability": "score definitions differ across model types; compare coverage and widths, not q alone",
            "development_groups": len(bias.development_groups) if bias else 0,
            "fitted_development_groups": len(bias.fitted_groups) if bias else 0,
            "development_cases_rejected": list(bias.rejected_cases) if bias else [],
            "coverage": coverage},
        "slices": slices, "cases": per_case, "per_landmark": per_landmark,
        "warnings": warnings,
        "interpretation": "error-to-independent-reference study; centered residuals describe the error model, not changed detector outputs or facial resemblance",
        "guarantee_limit": "marginal new-group conformal rank argument needs exchangeable groups under the frozen protocol; no per-image or conditional-slice guarantee",
        "array_order": {"cases": [r.case_id for r in rows], "reference_points": list(REFERENCE_NAMES)}}
    report["content_sha256"] = object_digest(report)
    return report, arrays, selected
