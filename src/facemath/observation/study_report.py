"""Offline study console. All visible numbers come from qualification JSON."""
from __future__ import annotations
from html import escape
import json
from typing import Any


def render_study_report(report: dict[str, Any]) -> str:
    def number(x, digits=3):
        return "—" if x is None else f"{x:.{digits}f}"
    def fraction(x):
        return "UNBOUNDED" if x is None else f"{100*x:.1f}%"
    model = report["selected_error_model"]
    coverage = model["coverage"]
    fixture = report["execution_kind"] == "synthetic_fixture"
    case_rows = []
    for c in report["cases"]:
        case_rows.append(f'''<tr data-split="{escape(c['split'])}" data-stratum="{escape(c['stratum'], quote=True)}" data-status="{escape(c['attempt_outcome'])}">
<td><a href="{escape(c['image'], quote=True)}">{escape(c['case_id'])}</a><small>{escape(c['group_id'])}</small></td>
<td>{escape(c['split'])}</td><td>{escape(c['stratum'])}</td><td><a href="{escape(c['receipt'], quote=True)}">{escape(c['attempt_outcome'])}</a></td>
<td class="num">{number(c['raw_mean_error_px'])}</td><td class="num">{number(c['centered_mean_error_px'])}</td>
<td class="num">{number(c['selected_simultaneous_score'])}</td></tr>''')
    slice_rows = []
    for s in report["slices"]:
        e = s["evaluation"]
        interval = e["binomial_95_interval"]
        ci = "—" if interval is None else f"{fraction(interval[0])}–{fraction(interval[1])}"
        flag = "represented" if s["has_calibration_representation"] else "NO CALIBRATION SUPPORT"
        slice_rows.append(f"<tr><td>{escape(s['stratum'])}<small>{flag}</small></td><td>{s['calibration_groups']}</td><td>{e['groups']}</td><td class='num'>{fraction(e['empirical_coverage']) if e['groups'] else '—'}</td><td class='num'>{ci}</td></tr>")
    landmark_rows = []
    for row in report["per_landmark"]:
        raw, centered = row["raw_mean_error_px"], row["centered_mean_error_px"]
        ratio = min(1., centered/raw) if raw and centered is not None else 0.
        landmark_rows.append(f"<div class='landmark'><code>{escape(row['name'])}</code><div class='track' aria-hidden='true'><i style='width:{100*ratio:.2f}%'></i></div><span>{number(raw)} → {number(centered)} px</span></div>")
    options = ''.join(f'<option value="{escape(s["stratum"],quote=True)}">{escape(s["stratum"])}</option>' for s in report['slices'])
    warnings = ''.join(f"<li>{escape(w)}</li>" for w in report['warnings'])
    excluded = ''.join(f"<li><code>{escape(c['case_id'])}</code> — {escape(c['exclusion_reason'])}</li>" for c in report['excluded_cases']) or '<li>None.</li>'
    scope = 'SYNTHETIC CONTROL / NO LIVE DETECTOR VALIDATION' if fixture else 'NATIVE OBSERVATION / REFERENCE-BASED STUDY'
    rawq = number(report['raw_baseline']['quantile']); newq = number(model['quantile'])
    payload = json.dumps(report, allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FACEMATH 05 · Observer qualification</title><style>
:root{{--bg:#0c1218;--panel:#131e28;--muted:#acbccb;--ink:#edf5fb;--line:#304553;--accent:#6de2c4;--warn:#ffd08a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}}a{{color:var(--accent)}}
main{{max-width:1320px;margin:auto;padding:38px 32px 70px}}header{{border-top:3px solid var(--accent);border-bottom:1px solid var(--line);padding:25px 0 30px}}
.eyebrow,code{{font:12px ui-monospace,monospace}}.eyebrow{{letter-spacing:.16em;color:var(--accent)}}h1{{font-size:clamp(28px,4vw,44px);letter-spacing:-.045em;line-height:1.1;margin:16px 0}}h2{{font-size:24px;letter-spacing:-.025em;margin:32px 0 12px}}
p{{color:var(--muted);max-width:93ch}}small{{display:block;color:var(--muted);font-size:12px}}.tag{{display:inline-block;border:1px solid var(--line);padding:7px 12px;font:11px ui-monospace,monospace;margin:6px 8px 0 0;overflow-wrap:anywhere}}
.scope{{color:var(--warn)}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:26px 0}}.flow div{{padding:18px;border:1px solid var(--line);background:var(--panel)}}.flow strong{{display:block;font:30px ui-monospace,monospace;line-height:1.6}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.panel{{border:1px solid var(--line);padding:22px;background:var(--panel)}}.panel h2{{margin:0 0 12px}}.q{{display:flex;justify-content:space-between;gap:15px;font:32px ui-monospace,monospace;padding:15px 0}}.q small{{font:12px system-ui,sans-serif}}
.landmark{{display:grid;grid-template-columns:1.25fr 1fr auto;gap:12px;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);font-size:12px}}.track{{height:6px;background:#41566a}}.track i{{display:block;height:100%;background:var(--accent)}}.notice{{border-left:3px solid var(--warn);padding:16px 20px;background:var(--panel);margin:24px 0}}
.scroll{{overflow:auto;border:1px solid var(--line)}}table{{width:100%;border-collapse:collapse;font-size:13px;min-width:600px}}th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:var(--panel)}}.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}code{{overflow-wrap:anywhere}}tr[hidden]{{display:none}}
.controls{{display:flex;align-items:end;flex-wrap:wrap;gap:12px;margin:18px 0}}label{{font-size:12px;color:var(--muted)}}select,button{{display:block;background:var(--panel);border:1px solid var(--line);color:var(--ink);padding:11px 14px;border-radius:3px;font:13px system-ui}}button{{cursor:pointer;color:var(--accent)}}:focus-visible{{outline:2px solid var(--accent);outline-offset:3px}}
.links{{display:flex;flex-wrap:wrap;gap:12px}}.links a{{padding:9px 12px;border:1px solid var(--line)}}footer{{margin-top:38px;border-top:1px solid var(--line);padding-top:18px;font-size:12px;color:var(--muted)}}
@media(max-width:800px){{main{{padding:20px 14px 40px}}.grid{{grid-template-columns:1fr}}.flow{{grid-template-columns:repeat(2,1fr)}}.landmark{{grid-template-columns:1.1fr .7fr auto;gap:6px}}.landmark code{{font-size:10px}}.panel{{padding:16px}}}}
</style></head><body><main>
<header><div class="eyebrow">FACEMATH / WAVE 05 / OBSERVER QUALIFICATION</div><h1>Separate the bias.<br>Keep the observation.</h1><p>Frozen enrollment → committed observation attempts → development-only error model → untouched held-out evaluation. The registered anatomy is never moved to make these results improve.</p>
<div class="tag scope">{scope}</div><div class="tag">{escape(report['domain'])}</div></header>
<section class="flow" aria-label="Study counts"><div><small>01 · Enrolled</small><strong>{report['enrollment']['enrolled']}</strong><small>{report['enrollment']['prospectively_excluded']} excluded before acquisition</small></div>
<div><small>02 · Committed attempts</small><strong>{report['enrollment']['committed']}</strong><small>no hidden retries</small></div><div><small>03 · Bias-fit groups</small><strong>{model['fitted_development_groups']}</strong><small>development split only</small></div><div><small>04 · Held-out group coverage</small><strong>{fraction(coverage['empirical_group_coverage'])}</strong><small>{coverage['evaluation_groups']} evaluation groups; not resemblance</small></div></section>
<div class="grid"><section class="panel"><h2>What calibration changed</h2><p>Simultaneous reference-error multiplier, before and after the frozen error-centering rule.</p><div class="q"><div>{rawq}<small>zero-centered baseline</small></div><div>→</div><div>{newq}<small>selected residual model</small></div></div><p>Estimator chosen before acquisition: <code>{escape(report['bias_mode_selected_before_acquisition'])}</code>. Raw image coordinates and the geometry profile remain unchanged. Lower is not automatically better: inspect held-out coverage.</p></section>
<section class="panel"><h2>Where the errors remain</h2><p>Held-out mean error: raw prediction → centered error-model residual. Bars show the residual fraction of each raw error.</p>{''.join(landmark_rows)}</section></div>
<div class="notice"><strong>Evidence constraints</strong><ul>{warnings or '<li>No additional acquisition warning.</li>'}</ul></div>
<h2>Prespecified acquisition slices</h2><p>These expose undercoverage and missing calibration representation. A global conformal rank does not establish conditional coverage in every slice. The binomial intervals require independent, comparable groups.</p><div class="scroll"><table><thead><tr><th>Stratum</th><th>Calibration groups</th><th>Evaluation groups</th><th>Empirical coverage</th><th>Binomial 95% interval</th></tr></thead><tbody>{''.join(slice_rows)}</tbody></table></div>
<h2>Every eligible attempt</h2><p>Click a case for its source raster or its outcome for the original execution receipt. Failed outputs remain in the denominator.</p>
<div class="controls"><label>Split<select id="split"><option value="all">All splits</option><option>development</option><option>calibration</option><option>evaluation</option></select></label>
<label>Stratum<select id="stratum"><option value="all">All strata</option>{options}</select></label><label>Outcome<select id="outcome"><option value="all">All outcomes</option><option value="observed">Observed</option><option value="failed">Failed / unavailable</option></select></label>
<button id="export" type="button">Export visible rows · CSV</button><small id="visible-count" role="status" aria-live="polite"></small></div>
<div class="scroll"><table id="cases"><thead><tr><th>Case / group</th><th>Split</th><th>Stratum</th><th>Receipt outcome</th><th>Raw error · px</th><th>Centered residual · px</th><th>Selected score</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table></div>
<details><summary>Prospectively excluded enrollment</summary><ul>{excluded}</ul></details>
<h2>Inspect and reproduce</h2><div class="links"><a href="qualification.json">Complete qualification JSON</a><a href="arrays.npz">Raw numerical arrays</a><a href="localization_calibration.json">Frozen error model</a><a href="raw_validation/report.html">Unadjusted source-coordinate validation</a><a href="acquisition/study/study.lock.json">Prospective study lock</a><a href="events.jsonl">Qualification execution journal</a><a href="manifest.json">Artifact manifest</a></div>
<footer>Study: <code>{report['study_sha256']}</code><br>Checksums establish internal consistency, not external authorship or independently attested chronology. No remote assets or analytics are used by this report. Native inference is a separate process and its third-party network behavior is not certified by this HTML.</footer>
<script type="application/json" id="data">{payload}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const rows=[...document.querySelectorAll('#cases tbody tr')];
function chosen(){{return {{split:document.getElementById('split').value,stratum:document.getElementById('stratum').value,outcome:document.getElementById('outcome').value}}}}
function matches(c,q){{return (q.split==='all'||c.split===q.split)&&(q.stratum==='all'||c.stratum===q.stratum)&&(q.outcome==='all'||(q.outcome==='failed'?c.attempt_outcome!=='observed':c.attempt_outcome==='observed'))}}
function filter(){{let count=0;const q=chosen();rows.forEach((r,i)=>{{const show=matches(data.cases[i],q);r.hidden=!show;if(show)count++;}});document.getElementById('visible-count').textContent=count+' / '+rows.length+' attempts shown';}}
['split','stratum','outcome'].forEach(id=>document.getElementById(id).addEventListener('change',filter));filter();
document.getElementById('export').addEventListener('click',()=>{{const keys=['case_id','group_id','split','stratum','attempt_outcome','raw_mean_error_px','centered_mean_error_px','selected_simultaneous_score'];const cell=x=>{{let v=String(x??'');if(typeof x==='string'&&/^[=+@\\-\\t\\r]/.test(v))v="'"+v;return '"'+v.replaceAll('"','""')+'"';}};const selected=data.cases.filter(c=>matches(c,chosen()));const csv=[keys.join(','),...selected.map(c=>keys.map(k=>cell(c[k])).join(','))].join('\\r\\n');const url=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));const a=document.createElement('a');a.href=url;a.download='facemath-visible-cases.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}});
</script></main></body></html>'''
