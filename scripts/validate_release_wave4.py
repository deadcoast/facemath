"""Execute source/wheel validation; retain real commands, exits and raw output.

No model downloads or missing network-dependency installation. Live ML inference
remains a separate, explicitly asset-dependent test.
"""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import sysconfig
import xml.etree.ElementTree as ET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--browser', default='/usr/bin/chromium')
    parser.add_argument('--phase', choices=['all','source','artifacts','wheel','wheel-tests','summary'], default='all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'evidence' / 'wave4'
    evidence.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix='facemath-release-v4-'))
    scratch.mkdir(parents=True, exist_ok=True)
    records_path = evidence / "release-executions.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.pop('PYTHONPATH', None)
    env.pop('COVERAGE_PROCESS_START', None)

    def run(name, command, cwd=root):
        start = perf_counter()
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = evidence / (name + '.txt')
        attempt = 1
        while log.exists():
            attempt += 1
            log = evidence / (name + f'.attempt-{attempt}.txt')
        log.write_text(result.stdout, encoding='utf-8')
        records.append({'name': name, 'command': [str(x) for x in command], 'cwd': str(cwd),
                        'returncode': result.returncode, 'elapsed_seconds': perf_counter()-start,
                        'log': str(log.relative_to(root)), 'log_sha256': sha256(log.read_bytes()).hexdigest()})
        (evidence/'release-executions.json').write_text(json.dumps(records, indent=2)+'\n')
        print(f'{name}: exit {result.returncode} ({perf_counter()-start:.2f}s)', flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
        return result.stdout

    if args.phase in ('all','source'):
        run('release-source-tests', [sys.executable,'-m','coverage','run','-m','pytest','-q',
                                    '--junitxml='+str(evidence/'source-tests.xml')])
        run('release-coverage', [sys.executable,'-m','coverage','json','-o',str(evidence/'coverage.json')])
    lab = evidence/'laboratory'
    if args.phase in ('all','artifacts'):
        if not lab.exists():
            run('release-laboratory', [sys.executable,'-m','facemath','validation-lab','--out',lab])
        for kind in ['validation', 'qualified_pair']:
            run('release-source-verify-'+kind,[sys.executable,'-m','facemath','verify',lab/kind])
            run('release-source-replay-'+kind,[sys.executable,'-m','facemath','replay',lab/kind,'--out',scratch/('source-replay-'+kind)])
        run('release-browser',[sys.executable,root/'scripts/browser_qa_wave4.py',lab,evidence/'browser','--chromium',args.browser])
    wheel=root/'dist/facemath-0.4.0-py3-none-any.whl'
    py=scratch/'wheel-env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    outside=scratch/'outside-source';outside.mkdir(exist_ok=True)
    if args.phase in ('all','wheel'):
        run('release-wheel-build',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',root,'-w',root/'dist'])
        run('release-venv',[sys.executable,'-m','venv','--system-site-packages',scratch/'wheel-env'])
        # The build interpreter is itself a venv. Nested --system-site-packages
        # sees /usr, not that parent's numerical dependencies. Add only its
        # existing package directory, after this wheel environment's own path.
        shared_dependencies = sysconfig.get_path('purelib')
        run('release-wheel-dependency-path',[py,'-c',
            'import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_path("purelib"))/"facemath-validation-dependencies.pth";p.write_text('+repr(shared_dependencies+'\n')+');print(p.read_text())'],outside)
        run('release-wheel-install',[py,'-m','pip','install','--no-deps','--force-reinstall',wheel],outside)
        imported=run('release-wheel-import',[py,'-c',
            'import facemath,json;from facemath.observation.journal import code_fingerprint;print(json.dumps({"module":facemath.__file__,"code_sha256":code_fingerprint()}))'],outside)
        import_record=json.loads(imported)
        if not Path(import_record['module']).is_relative_to(scratch/'wheel-env'):
            raise RuntimeError('wheel test accidentally imported source/global editable package')
        if (outside/'tests').exists():
            raise FileExistsError('wheel test tree already exists; select fresh scratch')
        shutil.copytree(root/'tests',outside/'tests',ignore=shutil.ignore_patterns('__pycache__'))
        (outside/'pytest.ini').write_text('[pytest]\nmarkers =\n    live_observer: requires explicit local runtime/model/source image\n')
    if args.phase in ('all','wheel-tests'):
        run('release-wheel-tests',[py,'-m','pytest','-q','tests','--junitxml='+str(evidence/'wheel-tests.xml')],outside)
        for kind in ['validation', 'qualified_pair']:
            run('release-wheel-replay-'+kind,[py,'-m','facemath','replay',lab/kind,'--out',scratch/('wheel-replay-'+kind)],outside)
    if args.phase in ('all','summary'):
        from facemath.observation.journal import code_fingerprint, environment
        import_record=json.loads((root / next(r["log"] for r in reversed(records) if r["name"]=="release-wheel-import" and r["returncode"]==0)).read_text())
        if import_record['code_sha256']!=code_fingerprint():
            raise RuntimeError('wheel and source Python bytes differ')

        def junit(path):
            suites=ET.parse(path).getroot().findall('testsuite')
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
            counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
            return counts

        live_cases=[e for e in ET.parse(evidence/'source-tests.xml').getroot().iter('testcase') if e.attrib.get('name')=='test_live_mediapipe_provider']
        live_executed=bool(live_cases and all(e.find('skipped') is None and e.find('failure') is None and e.find('error') is None for e in live_cases))
        result={
            'format':'facemath.release-validation.v4','version':'0.4.0',
            'completed_utc':datetime.now(timezone.utc).isoformat(),
            'environment':environment(),'source_tests':junit(evidence/'source-tests.xml'),
            'wheel_tests':junit(evidence/'wheel-tests.xml'),'wheel_import':import_record,
            'wheel_sha256':sha256(wheel.read_bytes()).hexdigest(),'executions':records,
            'wheel_dependency_scope':'facemath installed from wheel outside source; existing build-environment numerical dependencies exposed via .pth; not a fresh dependency installation',
            'laboratory':json.loads((lab/'laboratory.json').read_text()),
            'browser':json.loads((evidence/'browser/browser_validation.json').read_text()),
            'live_detector_inference':live_executed,'human_localization_calibration':False,
            'skipped_scope':'real MediaPipe runtime/model inference' if not live_executed else None,
            'boundary':'executed numerical/synthetic/provider-fixture/browser evidence; not external attestation or hidden model telemetry',
        }
        (root/'evidence/WAVE4_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'source':result['source_tests'],'wheel':result['wheel_tests'],'scratch':str(scratch)},indent=2))


if __name__=='__main__':main()
