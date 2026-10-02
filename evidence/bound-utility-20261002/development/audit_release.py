"""Read-only final package, fit, utility and observation comparison."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np

run=Path(__file__).resolve().parent
root=run.parents[1]
release=run/'release'
results={}
for label, path in [('source',release/'source-native'),('installed',release/'wheel-check/native-pipeline'),
                    ('archive',release/'archive-native'),('linux',run/'linux-native')]:
    subprocess.run([sys.executable,run/'audit_native.py',path,run/f'audit-{label}.result.json'],check=True)
    if label!='source':
        subprocess.run([sys.executable,run/'compare_runs.py',release/'source-native',path,
                        run/f'compare-{label}.result.json'],check=True)
    if label in ('installed','archive'):
        subprocess.run([sys.executable,run/'compare_fit.py',release/'source-native/qualification',path/'qualification',
                        run/f'fit-{label}.result.json'],check=True)
        left=json.loads((release/'source-native/qualification/qualification.json').read_text())
        right=json.loads((path/'qualification/qualification.json').read_text())
        assert left['bound_utility']==right['bound_utility']
        results[label+'_utility_exact']=True

archive=release/'facemath-0.13.0.zip'
checked=[]
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        relative=name.split('/',1)[1]
        if relative.startswith(('src/facemath/','tests/','docs/','scripts/','examples/','.github/')) or relative in (
                'README.md','AGENTS.md','CHANGELOG.md','REPOSITORY_REVIEW.json','pyproject.toml','uv.lock','.gitattributes','.gitignore'):
            assert z.read(name)==(root/relative).read_bytes(),relative
            checked.append(relative)
results.update(archive_sha256=sha256(archive.read_bytes()).hexdigest(),source_files_exact=checked,
               bytes=archive.stat().st_size)
with (run/'release-audit.result.json').open('x') as f:json.dump(results,f,indent=2)
print(json.dumps(dict(source_files=len(checked),archive_sha256=results['archive_sha256'])))
