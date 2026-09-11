import json
from hashlib import sha256

import numpy as np
import pytest

from facemath.observation import FacialInstrument, InstrumentConfig, ObservationError
from facemath.observation.artifacts import verify_bundle, write_bundle
from facemath.observation.contracts import canonical_json
from facemath.observation.journal import verify_journal
from facemath.observation.lab import phantom_observation, phantom_points, render_phantom


@pytest.fixture(scope="module")
def result():
    p = phantom_points()
    q = p.copy()
    q[19, 0] -= 5
    a, b = render_phantom(p), render_phantom(q)
    config = InstrumentConfig(
        canvas_size=96, eye_span_px=32, uncertainty_draws=4, enable_flow=False
    )
    result = FacialInstrument(config).run(
        a, b, phantom_observation(a, p), phantom_observation(b, q)
    )
    return result, a, b


def publish(result, tmp_path):
    r, a, b = result
    return write_bundle(r, tmp_path / "run", a, b)


def test_bundle_contains_verified_raw_arrays_and_original_observations(
    result, tmp_path
):
    out = publish(result, tmp_path)
    check = verify_bundle(out)
    assert check["verified"]
    assert check["arrays"] >= 40
    with np.load(out / "arrays.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(
            raw["observation_a_xy"], result[0].state.observation_a.xy
        )
        np.testing.assert_allclose(raw["triangle_jacobians"], result[0].field.jacobians)
    assert "data:image/png;base64" in (out / "report.html").read_text()


def test_bundle_is_create_only(result, tmp_path):
    out = publish(result, tmp_path)
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_bundle(result[0], out, result[1], result[2])
    assert (out / "manifest.json").read_bytes() == before


def test_byte_tampering_is_detected(result, tmp_path):
    out = publish(result, tmp_path)
    with (out / "arrays.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ObservationError, match="artifact mismatch"):
        verify_bundle(out)


def test_manifest_path_traversal_is_rejected_even_with_recomputed_hash(
    result, tmp_path
):
    out = publish(result, tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    m.pop("manifest_content_sha256")
    m["files"]["../outside.txt"] = {"bytes": 0, "sha256": sha256(b"").hexdigest()}
    m["manifest_content_sha256"] = sha256(canonical_json(m).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ObservationError, match="escapes"):
        verify_bundle(out)


def test_chain_event_tampering_detected(result):
    records = [json.loads(line) for line in result[0].journal_jsonl.splitlines()]
    verify_journal(records)
    records[2]["payload"]["invented"] = "event"
    with pytest.raises(ValueError, match="chain mismatch"):
        verify_journal(records)


def test_wrong_source_images_cannot_be_attached_to_result(result, tmp_path):
    a = result[1].copy()
    a[0, 0, 0] ^= 1
    with pytest.raises(ObservationError, match="different image"):
        write_bundle(result[0], tmp_path / "wrong", a, result[2])
    assert not (tmp_path / "wrong").exists()


def test_expected_failure_retains_real_partial_execution_journal(result):
    from facemath.observation.instrument import InstrumentFailure

    with pytest.raises(InstrumentFailure) as caught:
        FacialInstrument().run(result[1], result[2])
    records = [json.loads(line) for line in caught.value.journal_jsonl.splitlines()]
    verify_journal(records)
    assert records[-1]["operation"] == "observation.acquire.error"
    assert records[-1]["payload"]["exception_type"] == "ObservationError"
    assert all(r["operation"] != "execution.complete" for r in records)
