"""Exercise regional inspection on the retained native pair and frozen model."""
import argparse
import json
from pathlib import Path

from check_structural_pair import main as run_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_pair()
    def read(name):
        return json.loads((args.out/name).read_text(encoding='utf-8'))
    nominal, calibrated, replayed = [read(name+'/summary.json')['regional_inspection']
                                     for name in ('nominal', 'calibrated', 'replayed')]
    assert nominal == calibrated == replayed
    images = 0
    for row in nominal['rows']:
        for m in row['maps'] if row['maps_available'] else []:
            before = (args.out/'nominal'/m['image']).read_bytes()
            assert before == (args.out/'calibrated'/m['image']).read_bytes()
            assert before == (args.out/'replayed'/m['image']).read_bytes()
            images += 1
    comparison = read('comparison.json')
    result = {'raw_arrays_checked': comparison['raw_arrays_checked'], 'raw_arrays_unchanged': True,
              'regional_inspection_identical_with_calibration_and_replay': True,
              'map_images_identical': images, 'fit_performed': False, 'new_cohort': False,
              'scope': 'Retained native observations; no new independent anatomical validation'}
    (args.out/'regional-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
