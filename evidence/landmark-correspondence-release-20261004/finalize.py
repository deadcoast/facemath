"""Publish a final handoff only after every required release command succeeds."""
from pathlib import Path
from hashlib import file_digest, sha256
from datetime import datetime, timezone
import json, shutil, sys, xml.etree.ElementTree as ET
import numpy as np
from facemath.observation.journal import code_fingerprint
from facemath.observation.contracts import Observation, array_digest
root=Path(__file__).resolve().parents[2]
work=root/'runs/landmark-repair-20261004';release=work/'release-02'
evidence=root/'evidence/landmark-correspondence-20261004'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    with p.open('rb') as f:return file_digest(f,'sha256').hexdigest()
def write(p,data):
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2);f.write('\n')
def tests(path):
    tree=ET.parse(path).getroot(); suites=list(tree.iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    counts['skip_reasons']=[s.attrib.get('message',s.text) for s in tree.iter('skipped')]
    assert counts['failures']==counts['errors']==0
    return counts
v=read(release/'validation.json'); assert v['completed'] and all(c['returncode']==0 for c in v['commands'])
for name in ('release-02','release-native-audits','diff-check'):
    assert read(work/(name+'.json'))['returncode'] == 0
for name in ('wheel-audit.json','archive-audit.json'):
    assert read(work/name)['verified']
code=code_fingerprint(); installed=read(release/'wheel-check/installed-byte-check.json')
assert installed['import']['code_sha256']==code
assert read(release/'source-native/study/study.lock.json')['engine_code_sha256']==code
checks={name:tests(path) for name,path in {
    'source':release/'source.xml','installed_wheel':release/'wheel-check/pytest.xml','extracted_archive':release/'archive.xml'}.items()}
contexts={'source':release/'source-native','installed_wheel':release/'wheel-check/native-pipeline','extracted_archive':release/'archive-native'}
comparison=[];source=contexts['source']
assert (work/'release-01/source-native/qualification/localization_calibration.json').read_bytes()==(source/'qualification/localization_calibration.json').read_bytes()
for name,path in contexts.items():
    assert read(path/'pipeline.json')['completed']
    assert read(path/'study/study.lock.json')['engine_code_sha256']==code
    rows=[]
    for folder in sorted((source/'session/cases').iterdir()):
        receipt=read(folder/'receipt.json');other=path/'session/cases'/folder.name
        assert read(other/'receipt.json')['status']==receipt['status']
        row={'case_id':folder.name,'status':receipt['status']}
        if receipt['status']=='observed':
            old=Observation.from_dict(read(folder/'observation.json'));new=Observation.from_dict(read(other/'observation.json'))
            for key in ('xy','samples'):
                assert array_digest(getattr(old,key))==array_digest(getattr(new,key)),(name,folder.name,key)
            row.update(xy_sha256=array_digest(new.xy),samples_sha256=array_digest(new.samples))
        rows.append(row)
    assert (source/'qualification/localization_calibration.json').read_bytes()==(path/'qualification/localization_calibration.json').read_bytes()
    comparison.append({'context':name,'raw_observations_match_source':True,'model_bytes_match_source':True,'cases':rows})
audit=read(evidence/'source-audit-final.json'); held=audit['splits']['evaluation']
old=held['legacy_nose']['median_error_eye_spans'];new=held['landmarks'][4]['median_error_eye_spans']
rows=[c for c in audit['cases'] if c['split']=='evaluation']
q=read(source/'qualification/qualification.json'); pair=read(evidence/'native-pair-final/invariance.json')
assert q['live_human_observer_validation'] is False
correction=read(evidence/'reporting-correction.json')
assert correction['raw_observation_bytes_identical'] and correction['fitted_model_bytes_identical']
assert correction['settings_or_thresholds_retuned'] is False
if sys.argv[1:] == ['--check-only']:
    print(json.dumps(dict(ready=True, engine_code_sha256=code, validation=checks,
        native_contexts=[dict(context=c['context'], cases=len(c['cases']),
            raw_observations_match_source=c['raw_observations_match_source'],
            model_bytes_match_source=c['model_bytes_match_source']) for c in comparison]), indent=2))
    raise SystemExit(0)
archive=release/'facemath-0.17.0.zip'; target=root/'packages/facemath-0.17.0-correspondence-20261004.zip'
assert not target.exists();shutil.copyfile(archive,target)
assert digest(target)==v['archive']['archive_sha256']
post=root/'evidence/landmark-correspondence-release-20261004';post.mkdir()
for p in sorted(release.iterdir()):
    if p.is_file() and p.suffix in ('.json','.log','.xml'):shutil.copyfile(p,post/p.name)
(post/'wheel-check').mkdir()
for p in sorted((release/'wheel-check').iterdir()):
    if p.is_file() and p.suffix in ('.json','.xml'):shutil.copyfile(p,post/'wheel-check'/p.name)
for name in ('wheel-audit.json','archive-audit.json','pacer-validation.json'):
    if (work/name).exists():shutil.copyfile(work/name,post/name)
for name in ('release-02','release-native-audits','diff-check','wheel-audit-command','archive-audit-command'):
    for suffix in ('.json','.log'):
        shutil.copyfile(work/(name+suffix),post/(name+suffix))
shutil.copyfile(Path(__file__),post/'finalize.py')
shutil.copytree(root/'src/.pacer',post/'pacer',ignore=shutil.ignore_patterns('run'))
shutil.copytree(root/'src/.pacer/run/probes',post/'pacer-probes')
write(post/'context-comparison.json',comparison)
report={
 'format':'facemath.landmark_repair.v17','created_utc':datetime.now(timezone.utc).isoformat(),
 'version':'0.17.0','engine_code_sha256':code,
 'pacer_validation':read(work/'pacer-validation.json'),
 'implementation_status':'versioned_correspondence_repair_validated',
 'anatomical_accuracy_accepted':False,'global_research_inventory_complete':False,
 'changes':[
   {'path':'src/facemath/observation/qualification.py','result':'Published dataset agreement no longer claims live blind-human validation; recorded native execution and declared reference kinds are separate.'},
   {'path':'src/facemath/observation/correspondence.py','result':'Immutable legacy MP1 and new MP4 assignments; new recipe binds correspondence identity and full anchor map.'},
   {'path':'src/facemath/observation/contracts.py','result':'Explicit v17 observations; historical v3 serialization/digests preserved; unknown or downgraded correspondence refused.'},
   {'path':'src/facemath/observation/study.py','result':'Frozen native worker correspondence and model compatibility use the same identity as reference prediction.'},
   {'path':'src/facemath/observation/canonical.py','result':'Mixed pair/anatomy correspondence refuses before geometry; existing full-mesh instrument retained.'},
   {'path':'src/facemath/observation/source_cohort.py','result':'Explicit v17 PTS plans; legacy recipes remain unchanged.'},
   {'path':'scripts/check_wheel.py','result':'Isolated pytest basetemp avoids shared Windows temporary-directory collisions; skip reasons retained.'}],
 'parent':read(work/'parent.json'),
 'supported_parent_baseline':dict(read(work/'parent-supported.json'),summary='1156 passed, 1 Windows directory-symlink skip; live smoke executed'),
 'preserved_historical_evidence':read(evidence/'historical-preservation.json'),
 'pre_fix_reproduction':read(work/'pre-fix-reproduction.json'),
 'superseded_candidate':read(work/'superseded-01.json'),
 'reference_reporting':q['reference_evidence'],
 'reporting_correction':correction,
 'known_outcome_reexecution':True,
 'legacy_model_refusal':read(evidence/'legacy-model-refusal.json'),
 'validation':checks,
 'installed_package':installed,
 'native_study':{
   'final_provenance_flag_corrected':q['live_human_observer_validation'] is False,
   'plan':'examples/helen-correspondence-cohort.json','source_window':[132,192],
   'registry_prior_cases':255,'next_registry_cases':315,'enrolled':60,'observed':60,
   'development_enrolled':20,'development_fitted':19,'development_rejected':q['selected_error_model']['development_cases_rejected'],
   'calibration_groups':20,'evaluation_groups':20,'evaluation_covered_groups':19,
   'nose_median_old_eye_spans':old,'nose_median_new_eye_spans':new,'nose_relative_median_reduction':1-new/old,
   'nose_within_005_old':held['legacy_nose']['within_005'],'nose_within_005_new':held['landmarks'][4]['within_005'],
   'nose_cases_improved':sum(c['error_eye_spans'][4]<c['legacy_nose_error_eye_spans'] for c in rows),
   'nose_cases_worsened':sum(c['error_eye_spans'][4]>c['legacy_nose_error_eye_spans'] for c in rows),
   'raw_error_unit':'reference eye span','bound_width_unit':'observed eye span',
   'chin':held['landmarks'][7], 'bound_utility':q['bound_utility']['splits']['evaluation'],
   'observed_arrays_unchanged_across_qualification':audit['raw_arrays_checked'],
   'source_wheel_archive_native_arrays_and_model_bytes_match':True,
   'repeated_executions_are_new_independent_sources':False,'empirical_admission':False},
 'native_pair':pair,
 'artifacts':{
   'native_pair_report':'evidence/landmark-correspondence-20261004/native-pair-final/calibrated/report.html',
   'source_audit':'evidence/landmark-correspondence-20261004/source-audit-final.json',
   'heldout_plot':'evidence/landmark-correspondence-20261004/evaluation-nose-errors.png',
   'model':'evidence/landmark-correspondence-20261004/model-final.json',
   'next_registry':'evidence/landmark-correspondence-20261004/next-registry.json',
   'raw_release_records':post.relative_to(root).as_posix(),
   'archive':target.relative_to(root).as_posix(),'archive_sha256':digest(target),
   'wheel':str(next((release/'wheels').glob('*.whl')).relative_to(root))},
 'remaining':[
   {'id':'LOCALIZATION-PRECISION','status':'NOT_ACCEPTED','evidence':'10/20 nose points meet 0.05; broad marginal diameters median 0.754 and maximum 2.526 eye spans; three nose cases worsened.','next':'Diagnose residuals only on development inputs; freeze any revised estimator and a new unseen evaluation before outcomes.'},
   {'id':'CHIN-CORRESPONDENCE','status':'UNRESOLVED','evidence':'7/20 within 0.05; median error 0.070121 eye spans; no replacement selected.','next':'Review original development chin contour/pose/visibility semantics before proposing a new mapping.'},
   {'id':'ANATOMICAL-VALIDATION','status':'OPEN','evidence':'Published PTS agreement does not verify anatomy, visibility, person independence or native-training separation.','next':'Acquire independently supported anatomical/visibility references under a frozen protocol.'},
   {'id':'CROSS-PLATFORM','status':'NOT_EXECUTED_FOR_017','evidence':'This repair validates Windows with the retained Python 3.11 runtime.','next':'Run other supported hosts separately; exact replay must enforce recorded numerical-runtime compatibility.'},
   {'id':'CHECKOUT-ENVIRONMENT','status':'STALE_ENVIRONMENT_RETAINED','evidence':'The initial .venv baseline failed on OpenCV 5 and stale 0.8 metadata; no tests were weakened.','next':'Use runs/enrollment-registry-20261003/env311/Scripts/python.exe, the validated 0.17 interpreter, or sync a supported locked environment.'}],
 'completion_boundary':'Implementation, native study, raw-array invariance and Windows source/wheel/archive validation complete. Product precision, anatomical acceptance and the historical research inventory are not declared complete.',
 'publication':{'committed':False,'pushed':False,'package_created_locally':True},
 'commands':v['commands']}
write(root/'LANDMARK_REPAIR.json',report)
write(target.with_suffix('.validation.json'),report)
write(target.with_suffix('.release.json'),dict(v['archive'],copied_artifact=str(target),copy_sha256=digest(target)))
print(json.dumps({'report':'LANDMARK_REPAIR.json','checks':checks,'archive':str(target),'sha256':digest(target)},indent=2))
