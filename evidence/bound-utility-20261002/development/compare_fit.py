"""Compare stored numerical state across releases without changing either run."""
import json
from pathlib import Path
import sys
import numpy as np

left, right, out = map(Path, sys.argv[1:])
arrays = []
with np.load(left/'arrays.npz', allow_pickle=False) as a, np.load(right/'arrays.npz', allow_pickle=False) as b:
    assert set(a.files) == set(b.files)
    for name in a.files:
        x, y = a[name], b[name]
        equal = x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
        arrays.append(dict(name=name, shape=list(x.shape), dtype=x.dtype.str, bytes_equal=equal))
result = dict(left=str(left), right=str(right), arrays=arrays, all_equal=all(a['bytes_equal'] for a in arrays),
    scope='saved measured and fitted numerical state only; different code/runtime provenance is preserved')
with out.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(dict(arrays=len(arrays), all_equal=result['all_equal'])))
assert result['all_equal']
