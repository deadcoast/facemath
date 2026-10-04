"""Create distinct final-audit scripts; never replace earlier run evidence."""
from pathlib import Path

run = Path(__file__).resolve().parent

def create(old, new, changes):
    text = (run/old).read_text(encoding='utf-8-sig')
    for before,after in changes:
        assert before in text, (old,before)
        text = text.replace(before,after)
    if (run/new).exists():
        assert (run/new).read_text(encoding='utf-8') == text
    else:
        with (run/new).open('x',encoding='utf-8') as stream: stream.write(text)

common = [("run/'release'", "run/'release-03'"), ("run/'linux-native'", "run/'linux-native-03'"),
          ('prospective-freeze.json','prospective-freeze-03.json')]
create('audit_release.py','audit_release_03.py', common + [('release-audit.result.json','release-audit-03.result.json')])
create('registry_audit.py','registry_audit_03.py', [
    ('release/source-native','release-03/source-native'),('release/wheel-check','release-03/wheel-check'),
    ('release/archive-native','release-03/archive-native'),("run/'linux-native'","run/'linux-native-03'"),
    ('prospective-freeze.json','prospective-freeze-03.json'),('registry-audit.result.json','registry-audit-03.result.json')])
create('check_linux.py','check_linux_03.py', [('release/source-native','release-03/source-native'),
    ("run/'linux-native'","run/'linux-native-03'"),('linux-comparison.result.json','linux-comparison-03.result.json')])
create('cross_host_check.py','cross_host_check_03.py', [('linux-native/','linux-native-03/'),('cross-host','cross-host-03')])
create('compatibility.py','compatibility_03.py', [('prospective-lock','prospective-lock-03'),
    ('focused-03-temp','release-03/source-temp'),('old-study-reader','old-study-reader-03'),
    ('old-capsule-reader','old-capsule-reader-03'),('compatibility.result.json','compatibility-03.result.json')])
create('retain_release.py','retain_release_03.py', common + [
    ('wsl-launch.json','wsl-launch-03.json'),('cross-host.result.json','cross-host-03.result.json'),
    ('release-audit.result.json','release-audit-03.result.json'),('registry-audit.result.json','registry-audit-03.result.json'),
    ('linux.xml','linux-03.xml'),('linux-comparison.result.json','linux-comparison-03.result.json'),
    ('compatibility.result.json','compatibility-03.result.json'),
    ('runs/enrollment-registry-20261003/release/wheels','runs/enrollment-registry-20261003/release-03/wheels')])
