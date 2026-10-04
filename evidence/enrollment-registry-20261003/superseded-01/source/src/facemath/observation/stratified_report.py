"""Display the frozen calibration route without implementing calibration in JavaScript."""
from __future__ import annotations

from html import escape
import json
from typing import Any


def _payload(value: Any) -> str:
    return json.dumps(value, allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')


def render_routing_study(routing: dict[str, Any] | None) -> str:
    if not routing:
        return ''
    categories = routing['categories']
    coverage = routing['coverage']
    options = ''.join(f'<option value="{escape(r["stratum"],quote=True)}">{escape(r["stratum"])}</option>' for r in categories)
    body = []
    for r in categories:
        c = next(x for x in coverage['strata'] if x['stratum'] == r['stratum'])['evaluation']
        q = 'WITHHELD' if r['quantile'] is None else f'{r["quantile"]:.6g}'
        finite_groups = c['groups'] if r['quantile'] is not None else 0
        body.append(f'<tr><td><strong>{escape(r["stratum"])}</strong><small>{escape(r["status"].replace("_"," "))}</small></td>'
            f'<td>{len(r["group_scores"])}</td><td>{r["rank"]}</td><td class="num">{q}</td>'
            f'<td>{c["covered_groups"] if c["covered_groups"] is not None else 0} / {c["groups"]}<small>{finite_groups} with finite envelope</small></td></tr>')
    return '''<section id="acquisition-routing" aria-labelledby="routing-title">
<h2 id="routing-title">One error model. Different acquisition support.</h2>
<p>Development fixes the shared mean and full covariance. Each prespecified acquisition category has its own calibration-group order statistic. The pooled radius is diagnostic only: there is no fallback when a category is absent, too small, or unbounded.</p>
<div class="notice"><strong>Category labels do not detect hidden shift.</strong> Membership is declared acquisition metadata. Conditional coverage requires comparable, exchangeable whole groups within the selected category. Unavailable categories remain in the full held-out denominator.</div>
<div class="scroll"><table id="routing-summary"><thead><tr><th>Frozen category</th><th>Calibration groups</th><th>Rank k</th><th>Deployed q</th><th>Delivered and covered / evaluated</th></tr></thead><tbody>''' + ''.join(body) + '''</tbody></table></div>
<div class="controls"><label>Inspect calibration category<select id="routing-category">''' + options + '''</select></label><button id="routing-export" type="button">Export retained group scores · CSV</button></div>
<p id="routing-detail" role="status" aria-live="polite"></p>
<div class="scroll" style="max-height:330px"><table id="routing-scores"><thead><tr><th>Order</th><th>Calibration group</th><th>Score</th><th>Frozen selected rank</th></tr></thead><tbody></tbody></table></div>
<p>Full-precision group scores, masks and category indices are in <a href="arrays.npz">arrays.npz</a>. Order and group IDs are recorded in <a href="qualification.json">qualification.json</a>. The table selects views of saved values; it does not fit a model or recompute the quantile.</p>
<script type="application/json" id="routing-data">''' + _payload(routing) + '''</script>
<script>(()=>{const d=JSON.parse(document.getElementById('routing-data').textContent),selector=document.getElementById('routing-category'),body=document.querySelector('#routing-scores tbody');
function selected(){return d.categories.find(r=>r.stratum===selector.value)}
function ordered(r){return r.group_scores.map(([id,score])=>({id,score})).sort((a,b)=>(a.score??Infinity)-(b.score??Infinity)||a.id.localeCompare(b.id))}
function render(){const r=selected();body.replaceChildren();ordered(r).forEach((v,i)=>{const tr=document.createElement('tr');if(i+1===r.rank)tr.setAttribute('data-selected-rank','true');[i+1,v.id,v.score===null?'unavailable / +infinity':String(v.score),i+1===r.rank?'k = '+r.rank:''].forEach(x=>{const td=document.createElement('td');td.textContent=String(x);tr.append(td)});body.append(tr)});document.getElementById('routing-detail').textContent=r.stratum+' · '+r.status.replaceAll('_',' ')+' · '+r.group_scores.length+' calibration groups · rank '+r.rank+' · '+(r.quantile===null?'no deployable radius':('q = '+String(r.quantile)));}
selector.addEventListener('change',render);render();
document.getElementById('routing-export').addEventListener('click',()=>{const r=selected();const cell=x=>{let s=String(x??'');if(typeof x==='string'&&/^[=+@\\-\\t\\r]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'};const rows=ordered(r).map((x,i)=>[r.stratum,i+1,x.id,x.score,x.score!==null,r.rank,i+1===r.rank,r.quantile,r.status]);const csv=[['category','order','group_id','score','score_available','frozen_k','selected_rank','deployed_q','status'],...rows].map(r=>r.map(cell).join(',')).join('\\r\\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='facemath-category-calibration.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
})();</script></section>'''


def render_routing_pair(calibration: Any) -> str:
    from .stratified import BoundStratifiedCalibration
    if not isinstance(calibration, BoundStratifiedCalibration):
        return ''
    info = calibration.routing_evidence()
    cards = []
    for name, source in zip(('A', 'B'), info['sources']):
        cards.append(f'''<div class="route-card"><small>SOURCE {name} · FIXED DECLARATION</small>
<h3>{escape(source['stratum'])}</h3><div class="route-number">q = {source['quantile']:.8g}</div>
<p>{source['calibration_groups']} calibration groups. Shared development mean and covariance; source-specific radius.</p>
<small>Image</small><code>{escape(source['image_sha256'])}</code><small>Acquisition group</small><code>{escape(source['group_id'])}</code></div>''')
    return '''<style>.route-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.route-card{border:1px solid var(--line);background:var(--panel);padding:20px;min-width:0}.route-card h3{font-size:26px;margin:8px 0}.route-card code{word-break:break-all}.route-number{font:25px ui-monospace,monospace;color:var(--accent)}@media(max-width:700px){.route-grid{grid-template-columns:1fr}}</style>
<section id="pair-routing"><h2>Acquisition support stays with its source</h2>
<p>These declarations are bound to the exact image and observation fingerprints. No category selector is offered here: changing a declaration requires a new recorded analysis, not a display adjustment.</p><div class="route-grid">''' + ''.join(cards) + '''</div><p>Both radii propagate through the existing fixed-gauge and refitted-shape bounds. The covariance is not rescaled to conceal the difference. Membership is a curator declaration, not verified domain recognition or a facial-similarity score.</p><p><a href="localization_calibration.json">Complete source-bound calibration</a> · <a href="arrays.npz">localization_source_quantiles / intrinsic_source_quantiles</a></p><script type="application/json" id="pair-routing-data">''' + _payload(info) + '''</script></section>'''
