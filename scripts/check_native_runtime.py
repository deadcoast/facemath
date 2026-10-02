"""Require actual isolated inference on the first eligible pinned dataset crop.

Retain the source, declaration, process output and smoke manifest even when the
native library aborts. This is an execution gate, not reference accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from facemath.observation.dataset import verify_dataset
from facemath.observation.native_check import native_smoke, verify_smoke
from facemath.observation.study import ObserverDeclaration
from facemath.observation.study_runner import NativeExecutor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verify_dataset(args.dataset)
    recipe = json.loads((args.dataset / 'recipe.json').read_text())
    spec = json.loads((args.dataset / 'study.spec.json').read_text())
    case = next(c for c in spec['cases'] if c['eligibility'] == 'included')
    model = next(a for a in recipe['assets'] if a['path'] == recipe['model_asset'])
    NativeExecutor(args.dataset / 'assets' / recipe['model_asset']).preflight(
        ObserverDeclaration.from_dict(spec['observer']))
    report = native_smoke(args.dataset / case['image'],
                          args.dataset / 'assets' / recipe['model_asset'], args.out,
                          expected_sha256=model['sha256'], timeout_seconds=spec['timeout_seconds'])
    verify_smoke(args.out)
    print(json.dumps(report, indent=2))
    if report['status'] != 'observed':
        stderr = args.out / 'attempt/stderr.txt'
        if stderr.exists():
            print(stderr.read_text(encoding='utf-8', errors='replace'))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
