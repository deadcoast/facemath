"""Describe the intermediate run exactly; it does not qualify the final build."""
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from facemath.observation.study import read_json, write_json_new

run = Path(__file__).resolve().parent
def counts(path):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == 'testsuite' else list(tree)
    out = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    out['passed'] = out['tests']-out['errors']-out['failures']-out['skipped']
    return out
record = dict(created_utc=datetime.now(timezone.utc).isoformat(), disposition='superseded intermediate validation',
    reason='Empty registry/prior path guard was completed after this run began; only attempt 03 uses one frozen final fingerprint throughout',
    initial_windows_environment_code_sha256=read_json(run/'release-02/environment.log')['engine_code_sha256'],
    later_windows_study_code_sha256=read_json(run/'release-02/source-native/study/study.lock.json')['engine_code_sha256'],
    windows_commands_returncode=read_json(run/'validate-02.json')['returncode'],
    windows_validation=read_json(run/'release-02/validation.json'),
    windows_source=counts(run/'release-02/source.xml'),windows_archive=counts(run/'release-02/archive.xml'),
    linux=counts(run/'linux-02.xml'),linux_command_returncode=read_json(run/'wsl-launch-02.json')['returncode'],
    linux_error='study code differs; freeze a new study rather than mutating its protocol',
    retained_linux_fixture='evidence/enrollment-registry-20261003/superseded-02/linux-fixture',
    final_qualification_attempt='release-03 and linux-native-03',final_qualification=False)
assert record['initial_windows_environment_code_sha256'] != record['later_windows_study_code_sha256']
assert record['linux']['errors'] == 17 and record['linux_command_returncode'] == 1
write_json_new(run/'intermediate-attempt-02.json',record)
print(json.dumps({k:v for k,v in record.items() if k != 'windows_validation'},indent=2))
