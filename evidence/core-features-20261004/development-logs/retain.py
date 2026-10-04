from pathlib import Path
import json, shutil
from facemath.observation.journal import code_fingerprint

root=Path(__file__).resolve().parents[2]
source=Path(__file__).resolve().parent
evidence=root/'evidence/core-features-20261004'
target=evidence/'development-logs'
for path in source.iterdir():
    if path.is_file() and path.suffix in ('.json','.log','.py'):
        destination=target/path.name
        if destination.exists():
            assert destination.read_bytes()==path.read_bytes(),path
        else:shutil.copyfile(path,destination)
probes=target/'pacer-probes'; probes.mkdir(exist_ok=False)
for identifier in ('OGM-001','OGM-003','OGM-004','OGM-005'):
    directory=root/'src/.pacer/run/probes'/identifier
    latest=max(directory.glob('*.txt'),key=lambda p:p.name)
    shutil.copyfile(latest,probes/(identifier+'-'+latest.name))
failures={'format':'facemath.development_failures.v19','failures':[
    {'record':'development-logs/inherited-partial-failure.log','kind':'inherited_core_failure',
     'outcome':'primary registration succeeded; optional holdout aborted the full pair',
     'resolution':'development-logs/partial-repaired.log retains primary success and unavailable holdouts'},
    {'record':'development-logs/core-focused-01.log','kind':'test_integration_error',
     'outcome':'62 passed, one failed because the new test used a nonexistent export helper',
     'resolution':'correct existing write_bundle and CLI replay APIs; focused and final targeted logs retained'},
    {'kind':'editable_environment_setup','command':'uv pip install --python runs/enrollment-registry-20261003/env311/Scripts/python.exe --no-deps --no-build-isolation -e . --offline',
     'returncode':1,'observed_stderr':'ModuleNotFoundError: No module named setuptools',
     'raw_output_location':'tool response in this development conversation; this entry is a summary, not a retained raw shell log',
     'resolution':'editable-install-02 used the declared isolated build requirements from the existing uv cache'},
    {'record':'development-logs/registry-extend.log','kind':'correct_registry_refusal','returncode':2,
     'outcome':'registry extension cannot change the prior scope',
     'resolution':'registry-extend-02 preserves the exact inherited scope and adds the twenty enrolled cases'}],
    'code_sha256':code_fingerprint(),
    'open':['Full-mesh component accuracy and independent anatomical/visibility validation remain open; no fit or empirical admission performed.']}
with (evidence/'failure-history.json').open('x',encoding='utf-8') as stream:json.dump(failures,stream,indent=2)
print(json.dumps({'code_sha256':code_fingerprint(),'evidence':str(evidence)}))
