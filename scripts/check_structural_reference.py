"""Audit named measurements against retained PTS labels without fitting or retuning."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import Observation
from facemath.observation.reference import ReferenceAnnotation, mapped_prediction
from facemath.observation.structure import observed_value, structural_registry
from facemath.observation.study import read_json, write_json_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.pipeline.resolve()
    study = read_json(root/'study/study.lock.json')
    specs = structural_registry(InstrumentConfig())
    cases = []
    for i, case in enumerate(study['cases']):
        assert case['split'] == 'evaluation'
        path = root/'session/cases'/f'case_{i:05d}'/'observation.json'
        observation = Observation.from_dict(read_json(path)) if path.is_file() else None
        refs = [ReferenceAnnotation.from_dict(read_json(root/'study'/p)) for p in case['references']]
        assert len(refs) == 1 and refs[0].kind == 'external_dataset'
        ref = refs[0]
        raw = mapped_prediction(observation) if observation is not None else None
        rows = []
        for spec in specs:
            label, label_valid = observed_value(ref.xy, ref.visible, spec)
            observed, valid = observed_value(raw[0], raw[2], spec) if raw is not None else (0., False)
            error = observed-label if valid and label_valid else None
            rows.append({'key': spec.key, 'observed': observed if valid else None,
                         'published_label': label if label_valid else None,
                         'signed_error': error, 'tolerance': spec.tolerance,
                         'within_tolerance': abs(error) <= spec.tolerance if error is not None else False})
        cases.append({'case_id': case['case_id'], 'group_id': case['group_id'], 'rows': rows})
    summary = []
    for i, spec in enumerate(specs):
        errors = [abs(c['rows'][i]['signed_error']) for c in cases if c['rows'][i]['signed_error'] is not None]
        summary.append({'key': spec.key, 'denominator': len(cases), 'measured': len(errors),
                        'within_tolerance': sum(c['rows'][i]['within_tolerance'] for c in cases),
                        'tolerance': spec.tolerance, 'median_absolute_error': float(np.median(errors)) if errors else None,
                        'maximum_absolute_error': max(errors) if errors else None})
    result = {'study_sha256': read_json(root/'study/manifest.json')['study_sha256'],
              'fit_performed': False, 'thresholds_changed': False, 'empirical_admission': False,
              'reference_kind': 'external_dataset', 'summary': summary, 'cases': cases,
              'meaning': 'Recorded agreement with published PTS geometry; no anatomical-equivalence, visibility or person-independence claim; dependent rows are not independent samples'}
    write_json_new(args.out, result)
    print({r['key']: (r['within_tolerance'], r['denominator']) for r in summary})


if __name__ == '__main__':
    main()
