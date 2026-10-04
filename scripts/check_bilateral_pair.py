"""Check the new bilateral result using retained native observations, no fit."""
import argparse
import json
from pathlib import Path

import numpy as np

from check_structural_pair import main as run_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_pair()
    def read(name):
        return json.loads((args.out/name).read_text(encoding='utf-8'))
    nominal, calibrated, replayed = [read(name+'/summary.json')['bilateral_comparison']
                                     for name in ('nominal', 'calibrated', 'replayed')]
    assert calibrated == replayed == read('calibrated.log')['comparison']['bilateral_comparison']
    assert nominal['observed_counts']['unavailable'] == 0
    assert nominal['reference_counts']['unavailable'] == 5
    for a, b in zip(nominal['rows'], calibrated['rows']):
        for key in a:
            if key.startswith('observed_') or key == 'display_segments':
                assert a[key] == b[key], key
        delta = np.array(b['observed_b']) - b['observed_a']
        np.testing.assert_array_equal(delta, b['observed_delta_b_minus_a'])
        assert np.linalg.norm(delta) == b['observed_configuration_change']
    comparison = read('comparison.json')
    result = {'raw_arrays_checked': comparison['raw_arrays_checked'], 'raw_arrays_unchanged': True,
              'bilateral_replay_identical': True, 'rows': calibrated['rows'],
              'fit_performed': False, 'new_cohort': False,
              'scope': 'Same first native pair; no independent anatomical validation'}
    (args.out/'bilateral-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
