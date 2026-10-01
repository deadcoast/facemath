import copy
import json
from pathlib import Path
import numpy as np
import pytest

from facemath.observation.acquisition_lab import make_measured_study
from facemath.observation.study import freeze_study, load_study, read_json, file_inventory
from facemath.observation.contracts import ObservationError, object_digest


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    root=tmp_path_factory.mktemp('measured-study-freeze')
    p,_=make_measured_study(root/'inputs',compact=True)
    return p


def alternate(inputs,spec,name):
    path=inputs.parent/(name+'.json');path.write_text(json.dumps(spec));return path


def test_declared_label_cannot_override_computed_stratum(inputs,tmp_path):
    spec=read_json(inputs);spec['cases'][0]['stratum']='pixel_clear'
    # First source is smoothed, fixed by fixture schedule.
    with pytest.raises(ObservationError,match='conflicts'):
        freeze_study(alternate(inputs,spec,'override'),tmp_path/'frozen')
    assert not(tmp_path/'frozen').exists()


def test_measured_group_limit_cannot_differ_from_enrollment_limit(inputs,tmp_path):
    spec=read_json(inputs);spec['maximum_cases_per_group']=1
    with pytest.raises(ObservationError,match='enrollment size'):
        freeze_study(alternate(inputs,spec,'limit'),tmp_path/'frozen')


def test_reference_changes_do_not_change_measured_categories(inputs,tmp_path):
    # The measurement helper API accepts only image/source metadata. The larger
    # freeze still validates references independently, as required by the study.
    from facemath.observation.acquisition import AcquisitionPolicy,measure_frozen_groups
    from facemath.observation.study import load_rgb
    spec=read_json(inputs);policy=AcquisitionPolicy.from_dict(spec['stratified_error']['measurement_policy'])
    cases=tuple(spec['cases']);images={c['case_id']:load_rgb(inputs.parent/c['image'])for c in cases}
    first=measure_frozen_groups(policy,cases,lambda c:images[c['case_id']])
    second=measure_frozen_groups(policy,tuple({**c,'references':['not-read'],'predictions':np.zeros((8,2)),'pair_similarity':1.}for c in cases),lambda c:images[c['case_id']])
    assert [g.digest for g in first]==[g.digest for g in second]


def test_consistently_rehashed_study_measurement_lie_rejected(inputs,tmp_path):
    frozen=freeze_study(inputs,tmp_path/'frozen');root=frozen.root
    lock=read_json(root/'study.lock.json')
    lock['measured_acquisition']['groups'][0]['features'][0][-1]=999.
    (root/'study.lock.json').write_text(json.dumps(lock))
    manifest=read_json(root/'manifest.json');manifest.pop('content_sha256')
    manifest['study_sha256']=object_digest(lock);manifest['files']=file_inventory(root,exclude=('manifest.json',))
    manifest['content_sha256']=object_digest(manifest);(root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ObservationError,match='pixel recomputation'):load_study(root)


def test_excluded_sources_remain_in_frozen_group_measurement_inventory(inputs,tmp_path):
    frozen=freeze_study(inputs,tmp_path/'frozen')
    excluded=[c for c in frozen.cases if c['eligibility']=='excluded']
    assert len(excluded)==1
    g=next(g for g in frozen.data['measured_acquisition']['groups']if g['group_id']==excluded[0]['group_id'])
    assert excluded[0]['image_sha256']in{m['image_sha256']for m in g['members']}


def test_unknown_policy_categories_rejected_before_collection(inputs,tmp_path):
    spec=read_json(inputs);spec['strata'].append('extra')
    with pytest.raises(ObservationError,match='categories differ'):
        freeze_study(alternate(inputs,spec,'unknown'),tmp_path/'frozen')
