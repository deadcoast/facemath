"""Seal an adjacent, create-only record of completed reference-input validation."""
from __future__ import annotations
import argparse
from hashlib import file_digest
import shutil
import tomllib
from pathlib import Path
import xml.etree.ElementTree as ET
import json

from facemath.observation.journal import code_fingerprint
from facemath.observation.study import read_json, write_json_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('release',type=Path);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];release=args.release.resolve()
    feature=root/'evidence/annotation-coordinates-20261004'
    result=read_json(release/'validation.json')
    assert result['completed'] and all(c['returncode']==0 for c in result['commands'])
    assert not result['native_pipeline_executed']
    browser=read_json(feature/'browser-05/browser.json')
    parity=read_json(feature/'pair-parity.json')
    assert browser['passed'] and len(browser['views'])==6
    assert parity['raw_arrays_equal_to_parent']==158 and parity['code_sha256']==code_fingerprint()
    assert all(c['returncode']==0 for c in read_json(feature/'pair-regression-02/commands.json'))
    counts={}
    for tier,path in [('source',release/'source.xml'),('wheel',release/'wheel-check/pytest.xml'),('archive',release/'archive.xml')]:
        suite=ET.parse(path).getroot().find('testsuite')
        count={k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
        assert count['errors']==count['failures']==0
        live=[t for t in suite.findall('testcase') if t.attrib['name']=='test_live_mediapipe_provider']
        assert len(live)==1 and live[0].find('skipped') is None
        count['live_observer_smoke_passed']=True
        count['skip_reasons']=[t.find('skipped').attrib['message'] for t in suite.findall('testcase') if t.find('skipped') is not None]
        count['passed']=count['tests']-count['skipped'];counts[tier]=count
    milestones={}
    for tier,name in [('source','source-lab'),('wheel','wheel-check/milestone'),('archive','archive-lab')]:
        lab=read_json(release/name/'laboratory.json')
        assert lab['admitted_pair']['raw_arrays_unchanged'] and lab['withheld_query']['observer_calls']==0
        milestones[tier]={'raw_arrays_unchanged':lab['admitted_pair']['raw_arrays_checked'],
            'qualification_snapshot_replayed':lab['admitted_pair']['qualification_snapshot_replayed'],
            'withheld_query_observer_calls':0,'scope':'synthetic workflow control, not human accuracy'}
    target=args.evidence.resolve();target.mkdir(parents=True,exist_ok=False)
    for path in release.iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.xml'):shutil.copyfile(path,target/path.name)
    for name in ('archive-lab','archive-qualification-replay','archive-pair-replay'):
        shutil.copytree(release/name,target/name)
    wheel=target/'wheel-check';wheel.mkdir()
    for path in (release/'wheel-check').iterdir():
        if path.is_file():shutil.copyfile(path,wheel/path.name)
    version=tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
    archive=release/f'facemath-{version}.zip'
    with archive.open('rb') as stream:digest=file_digest(stream,'sha256').hexdigest()
    record={'format':'facemath.reference_input_release.v20','version':version,'code_sha256':code_fingerprint(),
       'parent':read_json(feature/'parent/parent.json'),'tests':counts,'milestones':milestones,
       'archive':{'path':f'packages/facemath-{version}-reference-input-20261004.zip','sha256':digest,'bytes':archive.stat().st_size,'files':result['archive']['files']},
       'changes':['Source pixel-center capture, marker and magnifier alignment through content-box transforms',
          'Versioned draft isolation and explicit in-page restore; keyboard pan',
          'Prepared workspace reference kind follows frozen observer execution kind',
          'Windows redirected overview preserves values with stream-compatible decoration'],
       'behavior':{'parent_maximum_coordinate_error_px':.535,'corrected_browser':str(feature/'browser-05/browser.json'),
          'corrected_maximum_export_error_px':max(v['maximum_error_px'] for v in browser['views']),
          'browser_export_fixture':'tests/fixtures/annotation-v20/desktop.reference.json',
          'intake_freeze_regression':'tests/test_annotation_coordinates.py',
          'terminal_regression':'tests/test_terminal_encoding.py','native_pair_regression':str(feature/'pair-parity.json')},
       'failures':str(feature/'failure-history.json'),'workbench':str(feature/'portrait-workbench.html'),
       'verification':str(target/'commands.json'),'installed_wheel_bytes':str(wheel/'installed-byte-check.json'),
       'native_inference':'Existing pinned live-observer smoke asset used by tests; pair regression reuses earlier native observations, no new cohort or fit.',
       'open':['Independent anatomical and visibility references still required; automated clicks are synthetic controls',
          'Legacy draft and exported annotation bytes remain unchanged; old labels are not retrospectively corrected',
          'Browser matrix executed on headless Edge on Windows, not every browser/platform',
          'Installed wheel uses the existing interpreter dependencies']}
    write_json_new(args.out,record)
    print(json.dumps({'archive':record['archive'],'tests':counts,'code_sha256':code_fingerprint()},indent=2))


if __name__=='__main__':main()
