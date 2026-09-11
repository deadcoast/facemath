import numpy as np
import pytest

from facemath.observation.contracts import ObservationError
from facemath.observation.sequence_guard import crossfade, sequence_diagnostics


@pytest.mark.parametrize("seed", range(8))
def test_unrelated_endpoints_have_rank_one_centered_crossfade(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.random((20, 20)), rng.random((20, 20))
    t = np.linspace(0, 1, 11)
    out = sequence_diagnostics(crossfade(a, b, t), t)
    assert out["centered_rank1_energy_fraction"] == pytest.approx(1)
    assert not out["admissible_as_facial_similarity_evidence"]
    assert out["first_last_affine_mix_residual_rms"] < 1e-12


def test_irregular_time_derivatives_have_correct_units():
    t = np.array([0, 0.1, 0.4, 0.9, 1.5])
    x = t[:, None, None] ** 2 * np.ones((1, 8, 8))
    out = sequence_diagnostics(x, t)
    out2 = sequence_diagnostics(x, 2 * t)
    np.testing.assert_allclose(out["acceleration_rms_per_second_squared"], 2)
    np.testing.assert_allclose(
        out2["velocity_rms_per_second"], np.array(out["velocity_rms_per_second"]) / 2
    )
    np.testing.assert_allclose(out2["acceleration_rms_per_second_squared"], 0.5)


def test_static_sequence_has_no_spurious_rank_fraction():
    out = sequence_diagnostics(np.ones((5, 8, 8)), np.arange(5))
    assert out["centered_rank1_energy_fraction"] is None
    assert out["first_last_affine_mix_coefficients"] is None


def test_supplied_mask_excludes_changing_background():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 20, 20))
    x[:, :5, :5] = 0
    mask = np.zeros((20, 20), bool)
    mask[:5, :5] = True
    out = sequence_diagnostics(x, np.arange(7), mask)
    assert out["centered_rank1_energy_fraction"] is None
    assert np.max(out["velocity_rms_per_second"]) == 0


@pytest.mark.parametrize("t", [[0, 0, 1], [0, 2, 1], [0, np.nan, 2]])
def test_invalid_time_order_is_rejected(t):
    with pytest.raises(ObservationError):
        sequence_diagnostics(np.ones((3, 8, 8)), np.array(t))


def test_empty_mask_is_not_silently_full_frame():
    with pytest.raises(ObservationError):
        sequence_diagnostics(np.ones((3, 8, 8)), np.arange(3), np.zeros((8, 8), bool))


@pytest.fixture
def actual_video(tmp_path):
    import cv2

    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 64))
    assert writer.isOpened(), "MJPG writer is required for this codec integration test"
    for index in range(9):
        image = np.full((64, 80, 3), index * 20, np.uint8)
        writer.write(image)
    writer.release()
    return path


def test_real_decode_retains_frame_identity_and_explicit_crop(actual_video):
    from facemath.observation.sequence_guard import inspect_video

    info, frames = inspect_video(actual_video, frame_count=5, roi=(5, 6, 30, 40))
    assert info["source_frame_count"] == 9
    assert [r["frame_index"] for r in info["sampling_records"]] == [0, 2, 4, 6, 8]
    assert all(f.shape == (40, 30, 3) for f in frames)
    assert info["scope"] == "explicit_ROI_not_verified_anatomy"
    assert not info["admissible_as_facial_similarity_evidence"]


def test_invalid_video_roi_does_not_silently_clip(actual_video):
    from facemath.observation.sequence_guard import inspect_video

    with pytest.raises(ObservationError, match="ROI"):
        inspect_video(actual_video, frame_count=3, roi=(70, 5, 40, 50))


def test_invalid_video_sampling_budget_and_missing_input(tmp_path):
    from facemath.observation.sequence_guard import inspect_video

    with pytest.raises(ObservationError, match="frame_count"):
        inspect_video(tmp_path / "nothing.mp4", frame_count=1)
    with pytest.raises(ObservationError, match="cannot open"):
        inspect_video(tmp_path / "nothing.mp4", frame_count=3)
