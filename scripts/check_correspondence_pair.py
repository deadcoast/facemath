"""Run the full native pair instrument before/after adding the new error model.

Cases are explicit command inputs; this is an interaction of existing production
components, not another alignment or a new qualification protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from facemath.observation.artifacts import write_bundle, verify_bundle
from facemath.observation.bias import load_localization_calibration
from facemath.observation.canonical import InstrumentConfig
from facemath.observation.contracts import Observation, array_digest
from facemath.observation.instrument import FacialInstrument, InstrumentFailure
from facemath.observation.study import read_json, load_rgb, write_json_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pipeline', type=Path)
    parser.add_argument('--cases', nargs=2, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.pipeline.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    study = read_json(root/'study/study.lock.json')
    indexed = {c['case_id']:(i,c) for i,c in enumerate(study['cases'])}
    selected = [indexed[name] for name in args.cases]
    assert all(c['split'] == 'evaluation' for _,c in selected)
    images = [load_rgb(root/'study'/c['image']) for _,c in selected]
    observations = [Observation.from_dict(read_json(root/'session/cases'/f'case_{i:05d}'/'observation.json'))
                    for i,c in selected]
    model = load_localization_calibration(read_json(root/'qualification/localization_calibration.json'))
    config = InstrumentConfig()  # full mesh, flow, edges, intrinsic relations and 96 sensitivity draws
    bare = FacialInstrument(config).run(*images, *observations)
    write_bundle(bare, out/'nominal', *images)
    try:
        qualified = FacialInstrument(config, localization_calibration=model,
            query_groups=tuple(c['group_id'] for _,c in selected), domain=study['domain'],
            query_lineages=tuple(tuple(c['lineage_ids']) for _,c in selected)).run(*images, *observations)
    except InstrumentFailure as exc:
        (out/'calibrated.failure.events.jsonl').write_text(exc.journal_jsonl, encoding='utf-8')
        raise
    write_bundle(qualified, out/'calibrated', *images)
    digests = {}
    for key, value in bare.raw_geometry_arrays().items():
        np.testing.assert_array_equal(value, qualified.arrays()[key], err_msg=key)
        digests[key] = array_digest(value)
    for name in ('nominal','calibrated'):
        assert verify_bundle(out/name)['verified']
    assert bare.profile.relational_distortion.shape == (478,478)
    result = dict(cases=args.cases, raw_arrays_unchanged=True, raw_arrays_checked=len(digests),
        raw_array_sha256=digests, native_vertices_per_image=478, relational_matrix_shape=[478,478],
        config=config.recipe(), calibration_sha256=model.digest,
        empirical_admission=False,
        scope='Full native pair instrument on two declared evaluation portraits; attaching the research model preserves measured geometry')
    write_json_new(out/'invariance.json', result)
    print({k:v for k,v in result.items() if k != 'raw_array_sha256'})


if __name__ == '__main__':
    main()
