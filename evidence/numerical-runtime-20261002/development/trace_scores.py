"""Read-only diagnosis of the retained 0.11.1 cross-host score differences."""
import json
import math
from pathlib import Path
import sys

import numpy as np

from facemath.observation.bias import DevelopmentBias, ocular_basis
from facemath.observation.qualification import rows_from_report

source, target = map(Path, sys.argv[1:])
model = json.loads((source / 'localization_calibration.json').read_text())
bias = DevelopmentBias.from_dict(model['bias'])
covariance = np.asarray(model['scatter']['covariance'])
chol = np.linalg.cholesky(covariance)
rows = rows_from_report(json.loads((source / 'raw_validation/validation.json').read_text()))
marginal = np.stack([covariance[2*i:2*i+2, 2*i:2*i+2] for i in range(8)])
traces = []
for row in rows:
    if row.split == 'development' or row.failure:
        continue
    basis = ocular_basis(row.predicted_xy)
    residual = np.linalg.solve(basis, row.error_vectors_px.T).T - bias.beta
    whitened = np.linalg.solve(chol, residual.reshape(16))
    marginal_solved = np.linalg.solve(marginal, residual[..., None])[..., 0]
    traces.append(dict(case_id=row.case_id, basis=basis.tolist(), residual=residual.tolist(),
        whitened=whitened.tolist(), marginal_solved=marginal_solved.tolist(),
        norm=float(np.linalg.norm(whitened)),
        sum_norm=float(np.sqrt(np.sum(whitened*whitened))),
        fsum_norm=math.sqrt(math.fsum(float(x)*float(x) for x in whitened)),
        hypot=math.hypot(*map(float, whitened))))
with target.open('x', encoding='utf-8') as stream:
    json.dump(dict(cholesky=chol.tolist(), cases=traces), stream, indent=2)
print(json.dumps(dict(cases=len(traces), destination=str(target))))
