"""Retain completed 0.21 paired-edge release evidence beside the sealed archive."""
from __future__ import annotations
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import tomllib
import xml.etree.ElementTree as ET
from facemath.observation.journal import code_fingerprint
from facemath.observation.study import read_json, write_json_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('release',type=Path);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];release=args.release.resolve()
    feature=root/'evidence/edge-comparison-20261004'
    result=read_json(release/'validation.json')
    assert result['completed'] and all(c['returncode']==0 for c in result['commands'])
    assert not result['native_pipeline_executed']
    browser=read_json(feature/'browser-02/browser.json');assert browser['passed']
    parity=read_json(feature/'pair-parity.json');assert parity['code_sha256']==code_fingerprint()
    assert parity['parent_arrays_unchanged']==158 and parity['raw_arrays_unchanged_by_calibration']==358
    assert all(c['returncode']==0 for c in read_json(feature/'pair-02/commands.json'))
    counts={};milestones={}
    for tier,path in [('source',release/'source.xml'),('wheel',release/'wheel-check/pytest.xml'),('archive',release/'archive.xml')]:
        suite=ET.parse(path).getroot().find('testsuite')
        count={k:int(suite.attrib[k]) for k in ('tests','errors','failures','skipped')}
        assert count['errors']==count['failures']==0
        live=[t for t in suite.findall('testcase') if t.attrib['name']=='test_live_mediapipe_provider']
        assert len(live)==1 and live[0].find('skipped') is None
        count['live_observer_smoke_passed']=True
        count['skip_reasons']=[t.find('skipped').attrib['message'] for t in suite.findall('testcase') if t.find('skipped') is not None]
        count['passed']=count['tests']-count['skipped'];counts[tier]=count
    for tier,name in [('source','source-lab'),('wheel','wheel-check/milestone'),('archive','archive-lab')]:
        lab=read_json(release/name/'laboratory.json')
        assert lab['admitted_pair']['raw_arrays_unchanged'] and lab['withheld_query']['observer_calls']==0
        assert lab['admitted_pair']['qualification_snapshot_replayed']
        milestones[tier]={'raw_arrays_unchanged':lab['admitted_pair']['raw_arrays_checked'],
            'qualification_snapshot_replayed':True,'withheld_query_observer_calls':0,
            'scope':'synthetic workflow control, not human accuracy'}
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
    record={'format':'facemath.edge_comparison_release.v21','version':version,'code_sha256':code_fingerprint(),
       'parent':read_json(feature/'parent/parent.json'),'tests':counts,'milestones':milestones,
       'archive':{'path':f'packages/facemath-{version}-paired-edges-20261004.zip','sha256':digest,'bytes':archive.stat().st_size,'files':result['archive']['files']},
       'changes':['Common-ray A/B image-edge displacement, separate A- and B-anchored inventories',
          'Signed/RMS/absolute displacement, per-side rejection reasons and fixed missingness denominators',
          'API and terminal output, report table and two spatial edge overlays',
          'Windows terminal fallback preserves literal path glyphs'],
       'native_pair':read_json(feature/'pair-02/edge-audit.json'),'parent_array_parity':parity,
       'browser':browser,'failures':str(feature/'failure-history.json'),
       'report':str(feature/'pair-02/calibrated/report.html'),
       'verification':str(target/'commands.json'),'installed_wheel_bytes':str(wheel/'installed-byte-check.json'),
       'open':['Image gradients do not establish independent anatomical or visibility accuracy',
          'Only normal displacement within the finite search window is measured; tangential changes and edge identity remain unresolved',
          'Browser matrix ran headless Edge on Windows, not every platform',
          'Installed wheel uses existing interpreter dependencies; no new native cohort or model fit']}
    write_json_new(args.out,record)
    print(json.dumps({'archive':record['archive'],'tests':counts,'code_sha256':code_fingerprint()},indent=2))


if __name__=='__main__':main()
