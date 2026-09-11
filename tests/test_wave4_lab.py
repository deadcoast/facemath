"""End-to-end scientific controls, not real detector validation."""

import json
import subprocess
import sys

import numpy as np

from facemath.observation.validation_io import replay_validation_bundle
from facemath.observation.validation_lab import run_localization_laboratory


def test_biased_but_repeatable_observer_is_not_mistaken_for_precise_localization(
    tmp_path,
):
    result = run_localization_laboratory(tmp_path / "lab")
    assert result["mean_perturbation_rms_px"] < 0.025
    assert result["heldout_mean_localization_error_px"] > 1.0
    assert result["bias_to_repeatability_ratio"] > 50
    assert result["naive_3sigma_simultaneous_hit_fraction_including_failures"] == 0
    assert result["calibration_failed_groups"] == 1
    assert result["evaluation_failed_cases"] == 1
    assert result["conformal_rank"] == 37
    assert result["fixed_gauge_bounds_contain_rendered_reference"]
    assert set(result["pair_localization_statuses"]) == {"unresolved_at_tolerance"}
    validation = tmp_path / "lab/validation"
    assert replay_validation_bundle(validation, tmp_path / "validation_replay")[
        "replay_verified"
    ]
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "facemath",
            "replay",
            str(tmp_path / "lab/qualified_pair"),
            "--out",
            str(tmp_path / "pair_replay"),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["replay_verified"]
    with np.load(
        tmp_path / "lab/qualified_pair/arrays.npz", allow_pickle=False
    ) as arrays:
        assert arrays["localization_ellipse_combined_support"].shape == (8, 128)
        assert arrays["localization_ellipse_normalized_covariances"].shape == (
            2,
            8,
            2,
            2,
        )
