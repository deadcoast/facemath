import json
from pathlib import Path

run=Path(__file__).resolve().parent
windows=run/'release-check/source-native/session/cases'
linux=run/'linux-pipeline/session/cases'
rows=[]
for path in sorted(windows.iterdir()):
    other=linux/path.name
    a,b=((p/'observation.json').read_bytes() for p in (path,other))
    ra,rb=(json.loads((p/'receipt.json').read_text()) for p in (path,other))
    rows.append(dict(case=path.name,canonical_observation_digest_equal=ra['observation_sha256']==rb['observation_sha256'],
        json_values_equal=json.loads(a)==json.loads(b),file_bytes_equal=a==b,
        only_crlf_lf_differs=a.replace(b'\r\n',b'\n')==b))
assert len(rows)==43 and all(r['canonical_observation_digest_equal'] and r['json_values_equal'] and r['only_crlf_lf_differs'] for r in rows)
with (run/'cross-observation-serialization.json').open('x') as f:
    json.dump(dict(cases=rows,observations=43,canonical_values_equal=43,
                   meaning='observation JSON serialization differs only by platform newline convention'),f,indent=2)
print('43 canonical observations equal; observation files differ only by CRLF/LF serialization')
