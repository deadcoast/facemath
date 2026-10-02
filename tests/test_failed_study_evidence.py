from pathlib import Path
import json
import pytest

from facemath.observation.contracts import ObservationError
from facemath.observation.journal import verify_journal
from facemath.observation.study import freeze_study, read_json
from facemath.observation.study_lab import make_fixture_study
from facemath.observation.study_runner import run_session
from facemath.observation.study_io import finalize_session


@pytest.mark.parametrize('stage',['validation','qualification'])
def test_fault_preserves_partial_bytes_and_actual_events_in_distinct_runs(tmp_path,monkeypatch,stage):
    spec,executor,_=make_fixture_study(tmp_path/'inputs',development=5,calibration=2,evaluation=2)
    study=freeze_study(spec,tmp_path/'study')
    run_session(study.root,tmp_path/'session',executor)
    def fault(*args,**kwargs):
        raise ObservationError('injected '+stage+' fault after acquisition')
    target=('facemath.observation.validation_io.evaluate_study' if stage=='validation'
            else 'facemath.observation.study_io.qualify')
    monkeypatch.setattr(target,fault)
    paths=[]
    for _ in range(2):
        with pytest.raises(ObservationError,match='injected') as caught:
            finalize_session(tmp_path/'session',tmp_path/'result')
        failed=Path(caught.value.failure_path);paths.append(failed)
        assert read_json(failed/'failure.json')['completed'] is False
        assert (failed/'acquisition/cases/case_00000/observation.json').is_file()
        journal=[json.loads(s) for s in (failed/'failure.events.jsonl').read_text().splitlines()]
        verify_journal(journal)
        assert journal[-1]['operation']=='execution.failed'
        assert not any(r['operation']=='execution.complete' for r in journal)
        if stage=='validation':
            children=list(failed.glob('raw_validation.failure-*/failure.events.jsonl'))
            assert len(children)==1
    assert paths[0]!=paths[1] and all(p.is_dir() for p in paths)
    assert not (tmp_path/'result').exists()
