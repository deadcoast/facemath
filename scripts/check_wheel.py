"""Install the built wheel away from the source tree, test it, and replay evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('wheel')
    parser.add_argument('--tests', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--pair-bundle')
    parser.add_argument('--out', required=True)
    parser.add_argument('--milestone', action='store_true', help='execute the installed CLI milestone control as well')
    parser.add_argument('--native-dataset', type=Path, help='execute a complete pinned native dataset pipeline from the installed wheel')
    parser.add_argument('--frozen-model', type=Path, help='evaluate an exported frozen model on the native dataset')
    parser.add_argument('--registry', type=Path, help='freeze the native study against this prior-enrollment snapshot')
    args = parser.parse_args()
    wheel = Path(args.wheel).resolve()
    tests = Path(args.tests).resolve()
    bundle = Path(args.bundle).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='facemath-installed-') as workspace:
        work = Path(workspace)
        target = work / 'installed'
        config = work / 'pytest.ini'
        config.write_text('[pytest]\nmarkers = live_observer: explicit local model integration\n')
        env = dict(os.environ, PYTHONPATH=str(target), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        commands = [
            [shutil.which('uv') or 'uv', 'pip', 'install', '--python', sys.executable,
             '--no-deps', '--no-index', '--target', str(target), str(wheel)],
            [sys.executable, '-c', 'import facemath,json; from facemath.observation.journal import code_fingerprint; '
             'print(json.dumps({"file":facemath.__file__,"version":facemath.__version__,"code_sha256":code_fingerprint()}))'],
            [sys.executable, '-m', 'pytest', '-q', '-ra', str(tests), '-c', str(config),
             f'--basetemp={work / "pytest-temp"}', f'--junitxml={out / "pytest.xml"}'],
            [sys.executable, '-m', 'facemath', 'verify', str(bundle)],
            [sys.executable, '-m', 'facemath', 'replay', str(bundle), '--out', str(out / 'replayed')],
        ]
        if args.pair_bundle:
            commands.append([sys.executable, '-m', 'facemath', 'verify', str(Path(args.pair_bundle).resolve())])
            commands.append([sys.executable, '-m', 'facemath', 'replay', str(Path(args.pair_bundle).resolve()),
                             '--out', str(out / 'pair-replayed')])
        if args.milestone:
            commands.append([sys.executable, '-m', 'facemath', 'milestone-lab', '--out', str(out / 'milestone')])
        if args.native_dataset:
            commands.append([sys.executable, '-m', 'facemath', 'dataset', 'run', str(args.native_dataset.resolve()),
                             '--out', str(out / 'native-pipeline'),
                             *(['--frozen-model', str(args.frozen_model.resolve())] if args.frozen_model else []),
                             *(['--registry', str(args.registry.resolve())] if args.registry else [])])
            commands.append([sys.executable, '-m', 'facemath', 'verify', str(out / 'native-pipeline')])
        for index, command in enumerate(commands):
            start = datetime.now(timezone.utc).isoformat()
            process = subprocess.Popen(command, cwd=work, env=env, text=True, encoding='utf-8',
                                       errors='replace', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = [], []
            def forward(stream, retained, destination):
                for line in stream:
                    retained.append(line)
                    print(line, end='', file=destination, flush=True)
                stream.close()
            readers = [threading.Thread(target=forward, args=(process.stdout, stdout, sys.stdout)),
                       threading.Thread(target=forward, args=(process.stderr, stderr, sys.stderr))]
            for reader in readers:
                reader.start()
            process.wait()
            for reader in readers:
                reader.join()
            result = subprocess.CompletedProcess(command, process.returncode, ''.join(stdout), ''.join(stderr))
            record = {'command': command, 'cwd': str(work), 'started_utc': start,
                      'finished_utc': datetime.now(timezone.utc).isoformat(),
                      'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
            (out / f'command-{index:02}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            if result.returncode:
                raise SystemExit(result.returncode)
            if index == 1:
                imported = json.loads(result.stdout)
                assert Path(imported['file']).is_relative_to(target), 'source-tree shadowing detected'
                original = json.loads((bundle / 'environment.json').read_text())
                assert imported['code_sha256'] == original['engine_code_sha256']
                with zipfile.ZipFile(wheel) as archive:
                    names = [n for n in archive.namelist() if n.startswith('facemath/') and n.endswith('.py')]
                    assert names
                    for name in names:
                        assert (target / name).read_bytes() == archive.read(name), name
                (out / 'installed-byte-check.json').write_text(json.dumps({
                    'verified': True, 'python_files': len(names), 'import': imported,
                    'scope': 'installed wheel Python bytes; same interpreter dependencies, not a new dependency stack'
                }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
