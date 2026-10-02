"""Hold the intermediate factor fixed, without changing any released result."""
import json
from pathlib import Path
import sys
import numpy as np

source, target = map(Path, sys.argv[1:])
other = json.loads(source.read_text())
chol = np.asarray(other['cholesky'])
rows = []
for row in other['cases']:
    whitened = np.linalg.solve(chol, np.asarray(row['residual']).reshape(16))
    rows.append(dict(case_id=row['case_id'],
        whitened_equal=whitened.tolist() == row['whitened'],
        whitened_max_delta=float(np.max(np.abs(whitened - row['whitened']))),
        norm_equal=float(np.linalg.norm(whitened)) == row['norm'],
        norm_delta=float(np.linalg.norm(whitened)) - row['norm']))
result = dict(source=str(source), cases=rows, numerical_contract_changed=False,
    meaning='read-only intermediate substitution diagnostic, not a replayed qualification')
with target.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(dict(cases=len(rows), different_whitened=sum(not r['whitened_equal'] for r in rows),
    different_norm=sum(not r['norm_equal'] for r in rows))))
