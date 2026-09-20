import argparse
import csv
import io
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from facemath.cli import _replay, build_parser, _configuration
from facemath.observation.artifacts import write_bundle, verify_bundle
from facemath.observation.contracts import ObservationError, canonical_json
from facemath.observation.intrinsic_lab import run_set_controls, run_intrinsic_laboratory
from facemath.observation.intrinsic_report import relation_csv, render_intrinsic_section, CSV_FIELDS
from test_wave6_intrinsic import run_pair


@pytest.fixture(scope='module')
def result():
    return run_pair(delta=5)


@pytest.fixture(scope='module')
def controls(tmp_path_factory):
    return run_set_controls(tmp_path_factory.mktemp('set-parent')/'sets',samples=64)


def publish(result,tmp_path):
    r,(a,b,_,_)=result
    return write_bundle(r,tmp_path/'pair',a,b)


def rehash(root,names):
    m=json.loads((root/'manifest.json').read_text())
    m.pop('manifest_content_sha256')
    for name in names:
        raw=(root/name).read_bytes()
        m['files'][name]={'bytes':len(raw),'sha256':sha256(raw).hexdigest()}
    m['manifest_content_sha256']=sha256(canonical_json(m).encode()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(m))


def test_full_intrinsic_csv_uses_immutable_profile_values(result):
    profile=result[0].intrinsic
    rows=list(csv.DictReader(io.StringIO(relation_csv(profile))))
    assert len(rows)==256 and tuple(rows[0])==CSV_FIELDS
    for row,original in zip(rows,profile.to_dict()['rows']):
        for key in CSV_FIELDS:
            if isinstance(original[key],float): assert float(row[key])==original[key]
            elif original[key] is None: assert row[key]==''
            else: assert row[key]==str(original[key])


def test_uncalibrated_csv_has_empty_bounds_not_zero(result):
    bare,_=run_pair(q=None)
    rows=list(csv.DictReader(io.StringIO(relation_csv(bare.intrinsic))))
    assert all(row['reference_gap_lower']=='' and row['reference_gap_upper']=='' for row in rows)
    assert all(row['observed_a']!='' for row in rows)


def test_report_reads_one_embedded_json_not_another_geometry_calculator(result):
    html=render_intrinsic_section(result[0].intrinsic)
    import re
    data=json.loads(re.search(r'<script id="intrinsic-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert data['rows']==result[0].intrinsic.to_dict()['rows']
    assert len(data['display']['ellipse'])==2
    assert not any(s in html for s in ('src="http','href="http','fetch('))
    assert 'intrinsic_source' not in html or 'intrinsic' in html


def test_intrinsic_bundle_writes_all_rows_and_preserves_array_contracts(result,tmp_path):
    root=publish(result,tmp_path)
    assert verify_bundle(root)['verified']
    assert json.loads((root/'intrinsic.json').read_text())==result[0].intrinsic.to_dict()
    assert len(list(csv.DictReader((root/'intrinsic.csv').open())))==256
    with np.load(root/'arrays.npz',allow_pickle=False) as data:
        for k,v in result[0].intrinsic.arrays().items():np.testing.assert_array_equal(data[k],v)


@pytest.mark.parametrize('artifact',['intrinsic.json','intrinsic.csv','arrays.npz'])
def test_tampered_intrinsic_artifact_is_detected(result,tmp_path,artifact):
    root=publish(result,tmp_path)
    with (root/artifact).open('ab') as f:f.write(b'tamper')
    with pytest.raises(ObservationError):verify_bundle(root)


def test_rehash_does_not_hide_intrinsic_json_disagreement(result,tmp_path):
    root=publish(result,tmp_path)
    data=json.loads((root/'intrinsic.json').read_text())
    data['rows'][0]['observed_a']+=.1
    (root/'intrinsic.json').write_text(json.dumps(data))
    rehash(root,['intrinsic.json'])
    with pytest.raises(ObservationError,match='intrinsic'):verify_bundle(root)


def test_consistently_forged_narrative_is_rejected_by_recomputation(result,tmp_path):
    root=publish(result,tmp_path)
    evidence=json.loads((root/'evidence.json').read_text())
    intrinsic=evidence['intrinsic_profile'];intrinsic['rows'][0]['observed_a']+=.1
    (root/'intrinsic.json').write_text(json.dumps(intrinsic))
    (root/'evidence.json').write_text(json.dumps(evidence))
    rehash(root,['intrinsic.json','evidence.json'])
    assert verify_bundle(root)['verified']  # hashes are not an external truth oracle
    with pytest.raises(ObservationError,match='replayed intrinsic'):
        _replay(argparse.Namespace(bundle=str(root),out=str(tmp_path/'replayed')))
    assert not (tmp_path/'replayed').exists()


def test_intrinsic_replays_with_calibration_and_nondefault_tolerances(result,tmp_path):
    root=publish(result,tmp_path)
    out=tmp_path/'replayed'
    response=_replay(argparse.Namespace(bundle=str(root),out=str(out)))
    assert response['replay_verified'] and verify_bundle(out)['verified']
    assert json.loads((root/'intrinsic.json').read_text())==json.loads((out/'intrinsic.json').read_text())


def test_cli_freezes_degree_input_as_radians():
    args=build_parser().parse_args(['demo','--out','not-executed','--relation-angle-tolerance-deg','3.5',
        '--relation-length-tolerance','.025','--relation-area-tolerance','.03','--relation-asymmetry-tolerance','.07'])
    c=_configuration(args)
    assert c.intrinsic_angle_tolerance_rad==pytest.approx(np.deg2rad(3.5))
    assert c.intrinsic_length_tolerance==.025 and c.intrinsic_area_tolerance==.03
    assert c.intrinsic_asymmetry_tolerance==.07


def test_cli_disable_remains_explicit():
    args=build_parser().parse_args(['demo','--out','not-executed','--no-intrinsic'])
    assert not _configuration(args).enable_intrinsic


def test_sampled_relations_and_independent_refits_stay_inside_outer_bounds(controls):
    assert controls['sampled_relation_violations']==0
    assert controls['sampled_refit_violations']==0
    assert controls['relation_checks']==64*256


def test_normalizer_falsification_is_material_not_roundoff(controls):
    c=controls['normalizer_counterexample']
    assert c['absolute_frozen_error']>.15 and c['contains_actual']


def test_mirror_and_zero_uncertainty_controls_are_nontrivial(controls):
    assert controls['reflection_control']['distance_matrix_max_gap']==0
    assert controls['reflection_control']['proper_shape_rms']>.8
    assert controls['zero_radius_limit']['range_width']<1e-10
    assert not controls['collapsed_normalizer']['mouth_ratio']['available']


def test_lab_uses_existing_frozen_study_and_real_calibration_code(tmp_path):
    report=run_intrinsic_laboratory(tmp_path/'lab')
    assert report['prior_study_preserved']['scripted_calls']==48
    assert report['qualification_verification']['verified']
    assert report['no_population_similarity_score']
    assert len(report['queries'])==4
    assert report['heldout_study_coverage'] is not None
    for row in report['queries']:
        assert row['raw_arrays_unchanged']
        assert row['known_shape_contained'] and row['known_relation_violations']==0
        assert row['finite_relation_count']==256
    assert report['queries'][-1]['outer_shape_range'][0]>.05
    with pytest.raises(FileExistsError):run_intrinsic_laboratory(tmp_path/'lab')
