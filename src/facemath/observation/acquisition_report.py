"""Offline displays of saved measured-source contracts; controls never reroute."""
from __future__ import annotations

from base64 import b64encode
from html import escape
import json
from typing import Any
import cv2
import numpy as np

from .acquisition import FEATURE_NAMES, FEATURE_UNITS, AGGREGATES, MeasuredGroup


def _png(image: np.ndarray) -> str:
    ok, payload = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError('could not encode acquisition preview')
    return 'data:image/png;base64,'+b64encode(payload).decode()


def acquisition_panel(groups: tuple[MeasuredGroup, ...], *, section_id: str = 'acquisition9') -> str:
    """Accept only computed immutable groups; no report-side measurement."""
    if not section_id.isalnum():
        raise ValueError('section identifier must be alphanumeric')
    data = []
    for group in groups:
        value = group.to_dict()
        value['policy'] = group.policy.to_dict()
        value['group_sha256'] = group.digest
        value['previews'] = [_png(s.image) for s in group.sources]
        value['grids'] = [_png(np.repeat(np.clip(a[...,None]*255,0,255).astype(np.uint8),3,2)) for a in group.analyzed]
        data.append(value)
    embedded = json.dumps(data, allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    return f'''<section class="acq9" id="{section_id}">
<style>
.acq9{{background:#111c27;border:1px solid #345065;border-radius:14px;padding:24px;margin:28px 0;color:#e2ebef;font:14px/1.6 system-ui}}
.acq9 h2,.acq9 h3{{margin:0 0 9px;color:#f3f7f9;overflow-wrap:anywhere}}.acq9 a{{color:#a5dfed}}.acq9 .muted{{color:#9cb0bc}}.acq9 .tools{{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:16px 0}}
.acq9 select,.acq9 button{{font:inherit;color:#ecf5f7;background:#1b3040;border:1px solid #536e80;border-radius:6px;padding:8px;max-width:100%}}
.acq9 .twocol{{display:grid;grid-template-columns:minmax(180px,1fr) minmax(260px,1.5fr);gap:22px;align-items:start}}.acq9 .twocol>*{{min-width:0}}.acq9 .scroll{{overflow:auto;max-width:100%;min-width:0}}
.acq9 svg{{width:100%;max-height:420px;display:block;background:#0c1319}}.acq9 table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}.acq9 td,.acq9 th{{border-bottom:1px solid #314554;padding:8px;text-align:left;vertical-align:top}}
.acq9 code{{word-break:break-all;font-size:12px}}.acq9 .status{{color:#9ee2d2;font-weight:700}}.acq9 .proof{{margin-top:20px}}
.acq9 .note{{border-left:3px solid #d2aa63;padding:10px 15px;background:#202c36}}.acq9 .metric{{font-family:ui-monospace,monospace;font-size:12px}}
@media(max-width:700px){{.acq9{{padding:15px}}.acq9 .twocol{{grid-template-columns:minmax(0,1fr)}}.acq9 .tools>*{{flex:1 1 160px}}}}
</style>
<h2>Measured source contract</h2><p class="muted">Original source pixels → declared aperture → frozen group predicates. No landmarks, reference labels, or pair resemblance enter this routing calculation.</p>
<div class="tools"><label>Source group <select data-role="group"></select></label><label>Enrolled member <select data-role="member"></select></label><label>Display <select data-role="view"><option value="source">Stored raster + aperture</option><option value="grid">Exact analyzed grayscale grid</option></select></label><button type="button" data-role="csv">Export group measurements CSV</button></div>
<div class="twocol"><div><svg data-role="preview" xmlns="http://www.w3.org/2000/svg" aria-label="Saved source measurement aperture"><image data-role="image"/><rect data-role="aperture" fill="none" stroke="#67e0c1" stroke-width="2" vector-effect="non-scaling-stroke"/></svg><p class="muted" data-role="caption"></p></div><div><h3 data-role="category"></h3><p class="status" data-role="status"></p><div class="scroll"><table><thead><tr><th>Metric</th><th>Selected source</th><th>Group min</th><th>Group max</th></tr></thead><tbody data-role="features"></tbody></table></div></div></div>
<div class="proof"><h3>Every routing predicate</h3><div class="scroll"><table><thead><tr><th>Category</th><th>Predicate</th><th>Measured</th><th>Holds</th></tr></thead><tbody data-role="predicates"></tbody></table></div></div>
<p class="note">Apertures are declared, not detected faces. Gradient detail is not calibrated blur accuracy: noise can increase it. Identical measured categories can still have different observer errors. Unknown or overlapping rules do not borrow a radius.</p>
<details><summary>Exact bindings and definitions</summary><p data-role="definition"></p><p>Group SHA-256: <code data-role="digest"></code></p><p>Measurement policy SHA-256: <code data-role="policy"></code></p><p>The complete declared group controls extrema. This interface cannot change the aperture, category, thresholds, or measurements.</p></details>
<script type="application/json" data-role="data">{embedded}</script>
<script>(()=>{{
const root=document.getElementById('{section_id}'),q=n=>root.querySelector('[data-role="'+n+'"]'),D=JSON.parse(q('data').textContent);
const pretty=x=>Number.isFinite(x)?Number(x).toPrecision(7):'unavailable';
const cell=(row,text,cls)=>{{const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;row.appendChild(td);return td}};
D.forEach((g,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=(D.length===2?(i===0?'A · ':'B · '):'')+g.group_id;q('group').appendChild(o)}});
function draw(){{const g=D[+q('group').value],j=+q('member').value,m=g.members[j],grid=q('view').value==='grid';
const shape=grid?g.analyzed_patches[j].shape:m.image_shape,h=shape[0],w=shape[1];q('preview').setAttribute('viewBox','0 0 '+w+' '+h);q('image').setAttribute('href',grid?g.grids[j]:g.previews[j]);q('image').setAttribute('width',w);q('image').setAttribute('height',h);
q('aperture').style.display=grid?'none':'';['x','y','width','height'].forEach((n,k)=>q('aperture').setAttribute(n,[m.aperture[0],m.aperture[1],m.aperture[2]-m.aperture[0],m.aperture[3]-m.aperture[1]][k]));
q('caption').textContent=(grid?'Stored measurement grid':'Source raster')+' · '+w+' × '+h+' · '+g.members.length+' enrolled member(s)';q('category').textContent='Derived category: '+g.category;q('status').textContent=g.deployment_permitted_by_rule?'Rule permits calibration lookup — finite cohort support still required':'Rule withholds calibrated deployment';
q('features').replaceChildren();g.feature_order.forEach((n,k)=>{{const row=document.createElement('tr');cell(row,n+' ['+g.feature_units[k]+']','metric');cell(row,pretty(g.features[j][k]));cell(row,pretty(g.aggregate[0][k]));cell(row,pretty(g.aggregate[1][k]));q('features').appendChild(row)}});
q('predicates').replaceChildren();g.policy.rules.forEach((rule,r)=>rule.predicates.forEach((p,k)=>{{const row=document.createElement('tr'),index=g.aggregate_order.indexOf(p.feature),v=g.aggregate.flat()[index];cell(row,rule.category+(g.rule_matches[r]?' · matched':''));cell(row,p.feature+' '+p.op+' '+p.threshold,'metric');cell(row,pretty(v));cell(row,g.predicate_truth[g.predicate_offsets[r]+k]?'true':'false');q('predicates').appendChild(row)}}));
q('definition').textContent=g.policy.definition;q('digest').textContent=g.group_sha256;q('policy').textContent=g.policy_sha256;}}
function selectGroup(){{const g=D[+q('group').value];q('member').replaceChildren();g.members.forEach((m,j)=>{{const o=document.createElement('option');o.value=j;o.textContent=String(j+1)+' · '+m.image_sha256.slice(0,14);q('member').appendChild(o)}});draw()}}
q('group').addEventListener('change',selectGroup);q('member').addEventListener('change',draw);q('view').addEventListener('change',draw);
q('csv').addEventListener('click',()=>{{const g=D[+q('group').value],esc=x=>'"'+String(x).replaceAll('"','""')+'"',rows=[['image_sha256','metric','unit','value','group_min','group_max']];g.members.forEach((m,j)=>g.feature_order.forEach((f,k)=>rows.push([m.image_sha256,f,g.feature_units[k],g.features[j][k],g.aggregate[0][k],g.aggregate[1][k]])));const blob=new Blob([rows.map(r=>r.map(esc).join(',')).join('\\r\\n')+'\\r\\n'],{{type:'text/csv;charset=utf-8'}}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='measured-acquisition.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000)}});selectGroup();
}})();</script></section>'''


def group_html(group: MeasuredGroup, *, artifact_prefix: str = "") -> str:
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FACEMATH · measured source contract</title><body style="margin:0;background:#09121a;color:#eef4f7;font-family:system-ui"><main style="max-width:1250px;margin:auto;padding:20px"><h1>FACEMATH / source acquisition</h1>'+acquisition_panel((group,))+f'<p style="display:flex;gap:10px;flex-wrap:wrap"><a style="color:#a5dfed" href="{escape(artifact_prefix, quote=True)}measurements.json">Exact JSON</a> <a style="color:#a5dfed" href="{escape(artifact_prefix, quote=True)}arrays.npz">Raw arrays</a> <a style="color:#a5dfed" href="{escape(artifact_prefix, quote=True)}policy.json">Frozen rules</a> <a style="color:#a5dfed" href="manifest.json">Byte inventory</a></p></main></body></html>'


def study_acquisition_section(value: dict[str, Any]) -> str:
    """Compact complete-inventory proof in the existing qualification report."""
    groups=value['groups']; rows=[]
    for g in groups:
        rows.append('<tr><td>'+escape(g['group_id'])+'</td><td>'+escape(g['category'])+'</td><td>'+str(len(g['members']))+'</td><td>'+str(g['deployment_permitted_by_rule']).lower()+'</td><td><code>'+escape(g['policy_sha256'][:16])+'</code></td></tr>')
    return '<section class="panel"><h2>Source-measured acquisition assignment</h2><p>Derived from all enrolled source apertures before observer invocation. No reference coordinates, prediction residuals or pair scores select a category. All rejected groups remain in the enrolled accounting.</p><div class="scroll"><table><thead><tr><th>Group</th><th>Derived category</th><th>Members</th><th>Rule permits deployment</th><th>Policy</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p>Exact aperture coordinates, feature rows and each predicate outcome are retained in qualification.json and the frozen study lock. A category is not a detector-accuracy or hidden-shift guarantee.</p></section>'
