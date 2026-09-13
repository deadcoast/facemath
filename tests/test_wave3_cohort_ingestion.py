"""A cohort must come from verified, compatible evidence, not convenient numbers."""
from dataclasses import replace
import json

import pytest

from facemath.observation import FacialInstrument, InstrumentConfig, ObservationError
from facemath.observation.artifacts import write_bundle
from facemath.observation.cohort import cohort_from_bundles
from facemath.observation.lab import phantom_points, phantom_observation, render_phantom


@pytest.fixture
def bundles(tmp_path):
    config = InstrumentConfig(canvas_size=96, eye_span_px=32, uncertainty_draws=4, enable_flow=False)
    output = []
    for i in range(3):
        p = phantom_points()
        q = p.copy()
        q[19, 0] -= 2 * (i + 1)
        a, b = render_phantom(p, seed=10 + i), render_phantom(q, seed=20 + i)
        result = FacialInstrument(config).run(a, b, phantom_observation(a, p), phantom_observation(b, q))
        output.append(str(write_bundle(result, tmp_path / f'run{i}', a, b)))
    return output


def test_ingestion_retains_source_manifest_and_exact_values(bundles):
    cohort = cohort_from_bundles('known-controls', bundles, kind='synthetic')
    assert len(cohort.rows) == 3
    assert all(row.source_manifest_sha256 for row in cohort.rows)
    assert cohort.rows[0].values['geometry.all.rms'] < cohort.rows[-1].values['geometry.all.rms']


def test_synthetic_evidence_cannot_be_relabeled_as_human_population(bundles):
    with pytest.raises(ObservationError, match='synthetic'):
        cohort_from_bundles('people', bundles, kind='observational')


def test_missing_metric_does_not_silently_drop_rows(bundles):
    with pytest.raises(ObservationError, match='missing'):
        cohort_from_bundles('reference', bundles, kind='synthetic', metrics=('fabricated_metric',))


def test_repeated_bundle_does_not_become_independent_sample(bundles):
    with pytest.raises(ObservationError, match='duplicate'):
        cohort_from_bundles('reference', [bundles[0]] * 3, kind='synthetic')
