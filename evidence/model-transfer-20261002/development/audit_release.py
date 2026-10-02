"""Read-only observed-state, frozen-fit and final archive byte checks."""
from hashlib import sha256
import json
from pathlib import Path
import zipfile
import numpy as np
from facemath.observation.contracts import Observation, array_digest
from facemath.observation.transfer_io import verify_transfer_pipeline

run=Path(__file__).resolve().parent
root=run.parents[1]
release=run/'release'
paths=dict(source=release/'source-native', installed=release/'wheel-check/native-pipeline',
           archive=release/'archive-native', linux=run/'linux-native')
donor=root/'evidence/bound-utility-20261002/preseal/native-pipeline/qualification/localization_calibration.json'
donor_bytes=donor.read_bytes()
observations={}
reports={}
result=dict(pipelines={}, comparisons={}, model_bytes_sha256=sha256(donor_bytes).hexdigest())
for label,path in paths.items():
    verification=verify_transfer_pipeline(path)
    observations[label]={}
    checked=[]
    for case in sorted((path/'session/cases').iterdir()):
        receipt=json.loads((case/'receipt.json').read_text())
        if receipt['status']!='observed':
            checked.append(dict(case=case.name,status=receipt['status'],raw_arrays=0));continue
        copies=[case/'observation.json',path/'evaluation/acquisition/cases'/case.name/'observation.json',
                path/'replay/acquisition/cases'/case.name/'observation.json']
        raw=[p.read_bytes() for p in copies]
        assert raw[0]==raw[1]==raw[2]
        obs=Observation.from_dict(json.loads(raw[0]))
        observations[label][case.name]=(obs,raw[0])
        checked.append(dict(case=case.name,status='observed',raw_arrays=2,
                            xy_sha256=array_digest(obs.xy),samples_sha256=array_digest(obs.samples)))
    for stage in ('evaluation','replay'):
        assert (path/stage/'localization_calibration.json').read_bytes()==donor_bytes
    reports[label]=json.loads((path/'evaluation/transfer.json').read_text())
    result['pipelines'][label]=dict(verification=verification, cases=checked,
        raw_arrays_unchanged=sum(c['raw_arrays'] for c in checked),model_bytes_unchanged=True,
        actual_environment=json.loads((path/'pipeline.json').read_text())['environment'])
for label in ('installed','archive','linux'):
    cases=[]
    for name,(left,raw) in observations['source'].items():
        if name not in observations[label]:
            cases.append(dict(case=name,same_outcome=False));continue
        right,other=observations[label][name]
        arrays={field:dict(bytes_equal=getattr(left,field).tobytes()==getattr(right,field).tobytes(),
            maximum_absolute_delta_px=float(np.max(np.abs(getattr(left,field)-getattr(right,field))))) for field in ('xy','samples')}
        cases.append(dict(case=name,same_outcome=True,observation_bytes_equal=raw==other,arrays=arrays))
    same_report=reports['source']==reports[label]
    with np.load(paths['source']/'evaluation/raw_validation/arrays.npz',allow_pickle=False) as a, np.load(paths[label]/'evaluation/raw_validation/arrays.npz',allow_pickle=False) as b:
        arrays={k:dict(bytes_equal=a[k].tobytes()==b[k].tobytes(),
            max_absolute_delta=float(np.max(np.abs(a[k].astype(float)-b[k].astype(float))))) for k in a.files}
    if label!='linux':
        assert same_report and all(c['observation_bytes_equal'] for c in cases)
        assert all(a['bytes_equal'] for a in arrays.values())
    result['comparisons'][label]=dict(cases=cases,report_exact=same_report,raw_arrays=arrays,
        limits='observed repeatability only; no cross-OS exact replay or universal native determinism claim')
archive=release/'facemath-0.14.0.zip'
checked=[]
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        relative=name.split('/',1)[1]
        if relative.startswith(('src/facemath/','tests/','docs/','scripts/','examples/','.github/')) or relative in (
                'README.md','AGENTS.md','CHANGELOG.md','REPOSITORY_REVIEW.json','pyproject.toml','uv.lock','.gitattributes','.gitignore'):
            assert z.read(name)==(root/relative).read_bytes(),relative
            checked.append(relative)
result.update(archive_sha256=sha256(archive.read_bytes()).hexdigest(),source_files_exact=checked,
              archive_bytes=archive.stat().st_size)
with (run/'release-audit.result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(archive_sha256=result['archive_sha256'],source_files=len(checked),
    raw_arrays_unchanged={k:v['raw_arrays_unchanged'] for k,v in result['pipelines'].items()},
    reports_equal={k:v['report_exact'] for k,v in result['comparisons'].items()}),indent=2))
