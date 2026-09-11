import numpy as np
import pytest

from facemath.geometry import generalized_procrustes, graph_laplacian
from facemath.registration import weighted_similarity_transform
from facemath.representation import linear_cka
from facemath.sequence import SequenceEvidence


def test_legacy_weighted_similarity_reflection_scale_corrected():
    points = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], float)
    _, scale, _ = weighted_similarity_transform(points, points * [-1, 1])
    assert scale == pytest.approx(0.6)


def test_legacy_cka_does_not_mutate_shared_array():
    x = np.arange(20, dtype=float).reshape(5, 4)
    before = x.copy()
    assert linear_cka(x, x) == pytest.approx(1)
    np.testing.assert_array_equal(x, before)
    x.setflags(write=False)
    assert linear_cka(x, x) == pytest.approx(1)


def test_legacy_procrustes_requires_explicit_reflection():
    a = np.array([[0, 0], [3, 0], [1, 1], [0, 2]], float)
    b = a * [-1, 1]
    assert generalized_procrustes(a, b).distance > 0.1
    assert generalized_procrustes(a, b, allow_reflection=True).distance < 1e-12


def test_isolated_graph_vertices_have_zero_normalized_laplacian():
    np.testing.assert_array_equal(graph_laplacian(np.zeros((3, 3))), np.zeros((3, 3)))


def test_directed_graph_cannot_be_silently_treated_as_undirected():
    with pytest.raises(ValueError, match="symmetric"):
        graph_laplacian(np.array([[0, 1], [0, 0]], float))


def test_legacy_sequence_is_explicitly_non_evidence_for_facial_resemblance():
    ev = SequenceEvidence(np.arange(3), np.zeros(2), np.zeros(2), 0, 0, None, 1)
    assert ev.admissible_as_facial_similarity is False
