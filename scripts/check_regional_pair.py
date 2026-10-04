"""Run the existing real-pair CLI checks, then audit regional component identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from check_structural_pair import main as run_pair
from facemath.observation.study import read_json, write_json_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_pair()  # Same arguments; preserves create-only results and every command log.
    root = args.out.resolve()
    summaries = {kind: read_json(root/kind/'summary.json')['regional_comparison']
                 for kind in ('nominal', 'calibrated', 'replayed')}
    assert summaries['nominal'] == summaries['calibrated'] == summaries['replayed']
    cli = read_json(root/'calibrated.log')['comparison']['regional_comparison']
    assert cli == summaries['calibrated']
    with np.load(root/'calibrated/arrays.npz') as arrays:
        supported = arrays['regional_triangle_supported']
        r = arrays['regional_triangle_rotation'][supported]
        u = arrays['regional_triangle_stretch'][supported]
        h = arrays['regional_triangle_log_stretch'][supported]
        dev = arrays['regional_triangle_deviatoric_log_stretch'][supported]
        area = arrays['regional_triangle_log_area'][supported]
        np.testing.assert_allclose(r @ u, arrays['triangle_jacobians'][supported], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(np.linalg.det(r), 1., atol=2e-12)
        np.testing.assert_allclose(np.trace(h, axis1=1, axis2=2), area, atol=2e-12)
        np.testing.assert_allclose(np.sum(h*h, axis=(1,2)), area*area/2 + np.sum(dev*dev, axis=(1,2)), atol=2e-12)
        registered_rms = {row['name']: row['value'] for row in read_json(root/'calibrated/evidence.json')['measurement_profile']}
        for row in cli['rows']:
            d = row['displacement']
            if d['total_rms'] is not None:
                np.testing.assert_allclose(d['total_rms']**2, d['translation_magnitude']**2+d['centered_rms']**2, atol=2e-12)
                np.testing.assert_allclose(d['total_rms'], registered_rms[f"geometry.{row['region']}.rms"], atol=2e-12)
        triangle_count = int(supported.sum())
    comparison = read_json(root/'comparison.json')
    assert comparison['raw_arrays_checked'] == 158
    diagnostic = read_json(root/'calibrated/evidence.json')['registration_diagnostics']
    result = {'format': 'facemath.regional_pair_check.v19',
        'selection': comparison['selection'], 'cases': comparison['cases'],
        'raw_arrays_checked': comparison['raw_arrays_checked'], 'raw_arrays_unchanged': True,
        'polar_reconstruction_and_energy_identity': True, 'supported_triangles': triangle_count,
        'same_components_before_after_calibration_and_replay': True,
        'regional_comparison': cli, 'registration_diagnostics': diagnostic,
        'limits': ['native projected model geometry; independent anatomical accuracy not established',
                   'regional deformation uses inherited A-domain support; no exchange-symmetric surface claim',
                   'the full mesh regional components have no propagated localization confidence intervals']}
    write_json_new(root/'regional-audit.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('regional_comparison','registration_diagnostics')}, indent=2))


if __name__ == '__main__':
    main()
