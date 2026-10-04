"""Exercise the production CLI on the first two prospectively enrolled native sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from facemath.observation.contracts import array_digest
from facemath.observation.study import read_json, write_json_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.pipeline.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    study = read_json(root/'study/study.lock.json')
    cases = study['cases'][:2]  # Before inspecting observation success or comparison values.
    assert len(cases) == 2 and all(c['split'] == 'evaluation' for c in cases)
    model_path = root/'evaluation/localization_calibration.json'
    model = read_json(model_path)
    commands = []

    def run(name, command):
        from datetime import datetime, timezone
        started = datetime.now(timezone.utc).isoformat()
        with (out/f'{name}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-m', 'facemath', *map(str, command)], stdout=log, stderr=subprocess.STDOUT)
        commands.append({'name': name, 'command': [sys.executable, '-m', 'facemath', *map(str, command)],
                         'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'returncode': result.returncode})
        (out/'commands.json').write_text(json.dumps(commands, indent=2), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see retained log')

    base = ['analyze', *(root/'study'/c['image'] for c in cases),
            '--observation-a', root/'session/cases/case_00000/observation.json',
            '--observation-b', root/'session/cases/case_00001/observation.json']
    run('nominal', [*base, '--out', out/'nominal'])
    args_model = ['--localization-calibration', model_path, '--domain', study['domain'],
                  '--query-groups', *(c['group_id'] for c in cases)]
    for source, case in zip(('a', 'b'), cases):
        for lineage in case['lineage_ids']:
            args_model.extend([f'--lineage-{source}', lineage])
    run('calibrated', [*base, *args_model, '--out', out/'calibrated'])
    for name in ('nominal', 'calibrated'):
        run(name+'-verify', ['verify', out/name])
    run('replay', ['replay', out/'calibrated', '--out', out/'replayed'])
    run('terminal', ['overview', out/'calibrated'])
    observed_intrinsic = {'raw_points', 'source_covariance', 'source_chart', 'source_supported',
                          'nominal', 'nominal_valid', 'relation_a', 'relation_b', 'support_normals'}
    observed_structure = {'nominal', 'nominal_valid', 'expression_a', 'expression_b'}
    observed_bilateral = {'nominal', 'nominal_valid', 'coefficients', 'display_segments', 'display_axes'}
    digests = {}
    with np.load(out/'nominal/arrays.npz') as a, np.load(out/'calibrated/arrays.npz') as b:
        for name in a.files:
            if name.startswith(('localization_', 'acquisition_')):
                continue
            if name.startswith('intrinsic_') and name.removeprefix('intrinsic_') not in observed_intrinsic:
                continue
            if name.startswith('structure_') and name.removeprefix('structure_') not in observed_structure:
                continue
            if name.startswith('bilateral_') and name.removeprefix('bilateral_') not in observed_bilateral:
                continue
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)
            digests[name] = array_digest(a[name])
        assert all('structure_'+key in digests for key in observed_structure)
    nominal = read_json(out/'nominal/summary.json')['structural_comparison']
    calibrated = read_json(out/'calibrated/summary.json')['structural_comparison']
    cli = read_json(out/'calibrated.log')['comparison']['structural_comparison']
    assert calibrated == cli
    assert len(calibrated['rows']) == 14 and nominal['reference_counts']['unavailable'] == 14
    for a, b in zip(nominal['rows'], calibrated['rows']):
        for key in ('observed_a', 'observed_b', 'observed_delta_b_minus_a', 'observed_status'):
            assert a[key] == b[key]
    result = {'cases': [c['case_id'] for c in cases], 'selection': 'first two frozen evaluation entries, no outcome replacement',
              'raw_arrays_unchanged': True, 'raw_arrays_checked': len(digests), 'raw_array_sha256': digests,
              'structure': calibrated, 'fit_performed': False, 'empirical_admission': False,
              'scope': 'Production CLI using real pinned native observations; full mesh, flow, edges, 96 sensitivity draws; source labels are published PTS'}
    write_json_new(out/'comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('raw_array_sha256', 'structure')}, indent=2))


if __name__ == '__main__':
    main()
