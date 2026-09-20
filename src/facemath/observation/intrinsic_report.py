"""Offline, inspectable presentation of the SAME immutable intrinsic profile.

No measurement is computed in JavaScript. Filters, diagrams and CSV export read
frozen profile values; neither display mode nor selection changes the recipe.
"""
from __future__ import annotations

import csv
from html import escape
import io
import json

import numpy as np

from .intrinsic import IntrinsicProfile


CSV_FIELDS = (
    "key", "family", "unit", "tolerance", "observed_a", "observed_b", "observed_absolute_gap",
    "reference_set_center_a", "reference_set_center_b", "reference_gap_lower", "reference_gap_upper", "status",
)


def relation_csv(profile: IntrinsicProfile) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in profile.to_dict()["rows"]:
        writer.writerow({k: row[k] for k in CSV_FIELDS})
    return stream.getvalue()


def render_intrinsic_section(profile: IntrinsicProfile) -> str:
    data = profile.to_dict()
    data["display"] = {
        "raw": profile.raw_points.tolist(), "centers": profile.source_centers.tolist(),
        "support": profile.source_supported.tolist(), "ellipse": [],
    }
    for source_index, covariance in enumerate(profile.reference_error_covariance):
        eig, vec = np.linalg.eigh(covariance)
        q = profile.source_quantiles[source_index] or 0.
        data["display"]["ellipse"].append([
            [float(q*np.sqrt(max(0., e[1]))), float(q*np.sqrt(max(0., e[0]))),
             float(np.degrees(np.arctan2(v[1,1],v[0,1])))] for e,v in zip(eig,vec)
        ])
    encoded = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    def number(value):
        return "Unavailable" if value is None else f"{value:.6g}"
    shape = data["shape_distance"]
    interval = f"[{number(shape['lower'])}, {number(shape['upper'])}]" if shape["available"] else "Not bounded"
    scope = "SYNTHETIC CONTROL ONLY" if profile.scope == "synthetic_only" else (
        "OBSERVED GEOMETRY · NO LOCALIZATION CALIBRATION" if not data["calibrated"] else "CONDITIONAL REFERENCE-SET BOUNDS")
    html = '''<section id="intrinsic" aria-labelledby="intrinsic-heading">
<style>
#intrinsic{margin-top:38px;border-top:1px solid var(--line);padding-top:26px}
#intrinsic .scope{font:11px ui-monospace,monospace;letter-spacing:.1em;color:#edc787}
#intrinsic .cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}
#intrinsic .stat{border:1px solid var(--line);background:var(--panel);padding:15px}
#intrinsic .stat strong{display:block;font:23px ui-monospace,monospace;margin:8px 0;overflow-wrap:anywhere}
#intrinsic .stat small{line-height:1.5}
#intrinsic .charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}
#intrinsic svg{width:100%;display:block;background:#111d29;border:1px solid var(--line);aspect-ratio:1.2}
#intrinsic .point{fill:#dae8ee;stroke:#111d29;stroke-width:.01}
#intrinsic .ref-point{fill:#ffc875;stroke:#111d29;stroke-width:.003}
#intrinsic .point-label{fill:#c6d3de;font:.045px ui-monospace,monospace}
#intrinsic .uncertainty{fill:#ffc87514;stroke:#daa553;stroke-width:.005}
#intrinsic .relation-line{fill:none;stroke:#6ee0d3;stroke-width:.014}
#intrinsic .relation-face{fill:#6ee0d320;stroke:#6ee0d3;stroke-width:.012}
#intrinsic .axis{stroke:#304455;stroke-width:.004;stroke-dasharray:.025 .025}
#intrinsic .toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:9px;margin:15px 0}
#intrinsic button,#intrinsic select,#intrinsic input{background:#142535;color:var(--ink);border:1px solid #3d576b;border-radius:4px;padding:9px 11px;font:13px system-ui}
#intrinsic button{cursor:pointer}#intrinsic button[aria-pressed=true]{background:#24494d;border-color:#6ee0d3}
#intrinsic button:focus-visible,#intrinsic select:focus-visible,#intrinsic input:focus-visible{outline:2px solid #6ee0d3;outline-offset:3px}
#intrinsic input[type=search]{width:min(100%,270px)}#intrinsic .result-count{color:var(--muted);font-size:13px}
#intrinsic .relation-table{max-height:490px;overflow:auto;border:1px solid var(--line)}
#intrinsic .relation-table thead{position:sticky;top:0;z-index:1}
#intrinsic .relation-table td{padding:10px 12px}#intrinsic .relation-table tr[aria-selected=true]{background:#244047}
#intrinsic .pick{padding:5px 8px;text-align:left;background:transparent;border-color:transparent;font:12px ui-monospace,monospace}
#intrinsic .status-in{color:#78ded0}#intrinsic .status-out{color:#ffbd83}
#intrinsic .selected-detail{padding:15px 18px;background:var(--panel);border-left:3px solid #6ee0d3;margin:15px 0}
#intrinsic .selected-detail code{display:block;margin-top:8px}#intrinsic .unit{font-size:12px;color:var(--muted)}
#intrinsic .method-detail{margin:20px 0;border:1px solid var(--line);padding:15px}
@media(max-width:720px){#intrinsic .cards{grid-template-columns:1fr}#intrinsic .charts{grid-template-columns:1fr}#intrinsic .relation-table{max-height:390px}#intrinsic .toolbar label{width:100%}}
</style>
<div class="scope">''' + escape(scope) + '''</div>
<h2 id="intrinsic-heading">Shape relations without a frozen alignment.</h2>
<p>Eight reference landmarks per source. Each relationship keeps its own geometry and unit.
The error sets propagate through the <strong>uncertain reference eye span</strong>; moving the display or fitting the pair cannot erase that uncertainty.</p>
<div class="cards">
<div class="stat"><small>Observed proper-rotation shape RMS</small><strong>''' + number(shape["observed_rms"]) + '''</strong><small>Own-eye-span-normalized shape. Not the existing registered landmark residual.</small></div>
<div class="stat"><small>Refitted reference-shape outer range</small><strong>''' + interval + '''</strong><small>Bounds include normalization uncertainty and a fresh optimal proper rotation for every admissible shape.</small></div>
<div class="stat"><small>Relation inventory</small><strong>28 / 168 / 56 / 4</strong><small>Lengths / angles / signed areas / asymmetries. Dependent descriptions, never 256 independent votes.</small></div>
</div>
<div class="toolbar"><label>Display <select id="intrinsic-display"><option value="raw">Observed landmarks</option><option value="reference">Reference-set centers and ellipses</option></select></label>
<label>Chart zoom <input id="intrinsic-zoom" type="range" min="1" max="6" step=".25" value="1" aria-label="Intrinsic chart zoom"></label>
<span class="unit">Source-eye charts. Zoom affects display only. Point indices are zero-based.</span></div>
<div class="charts"><figure><svg id="intrinsic-a" role="img" aria-label="Source A reference geometry"></svg><figcaption>Source A · same observation used by the primary instrument</figcaption></figure>
<figure><svg id="intrinsic-b" role="img" aria-label="Source B reference geometry"></svg><figcaption>Source B · same observation used by the primary instrument</figcaption></figure></div>
<div class="toolbar" id="intrinsic-families" role="group" aria-label="Relation family"></div>
<div class="toolbar"><label>Filter <input id="intrinsic-search" type="search" placeholder="Landmark name or relation key" aria-label="Search intrinsic relations"></label>
<label>Status <select id="intrinsic-status"><option value="all">All outcomes</option><option value="within_tolerance_under_envelope">Within tolerance</option><option value="outside_tolerance_under_envelope">Outside tolerance</option><option value="unresolved_at_tolerance">Unresolved</option><option value="unbounded_or_unobserved_dependency">Unavailable bound</option><option value="no_localization_calibration">Uncalibrated</option></select></label>
<button id="intrinsic-csv" type="button">Export selected rows</button><span class="result-count" id="intrinsic-count" aria-live="polite"></span></div>
<div id="intrinsic-selected" class="selected-detail" aria-live="polite"></div>
<div class="relation-table"><table><thead><tr><th>Relationship</th><th>Observed A</th><th>Observed B</th><th>Reference-gap range</th><th>Status</th></tr></thead><tbody id="intrinsic-rows"></tbody></table></div>
<details class="method-detail"><summary>What this range does—and does not—establish</summary>
<p>All bounds describe reference positions within the supplied eight-point error sets. No coordinates are corrected and no error model is fitted here.
A lower bound above tolerance establishes separation <em>under those sets</em>; an upper bound below tolerance establishes agreement under them.
An unavailable or unresolved relation is not counted as agreement. A matched-relation fraction is not facial resemblance probability.</p>
<p>Distances, unsigned angles and bilateral contrasts are reflection-insensitive. Oriented areas and the proper-rotation shape distance retain reflection sensitivity under fixed point correspondence.
The finite directional calculations use conservative corrections and float64 padding; they are not exact extrema or formally verified interval arithmetic.</p>
<p>Statistical coverage transfers only under the original source-calibration assumptions. A new acquisition domain or a poorly generalized detector is not repaired by these geometric bounds.</p>
<p><a href="intrinsic.json">Complete intrinsic JSON</a> · <a href="intrinsic.csv">All 256 relation rows</a> · <a href="arrays.npz">Raw arrays and masks</a></p></details>
<script id="intrinsic-data" type="application/json">''' + encoded + '''</script>
<script>
(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('intrinsic-data').textContent);
  const el=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
  const labels={distance:'Lengths · 28',angle:'Angles · 168',signed_double_area:'Oriented areas · 56',asymmetry:'Asymmetry · 4'};
  const statuses={within_tolerance_under_envelope:'Within tolerance',outside_tolerance_under_envelope:'Outside tolerance',unresolved_at_tolerance:'Unresolved',unbounded_or_unobserved_dependency:'Unavailable',no_localization_calibration:'Uncalibrated'};
  let family='distance', current=data.rows[0], visible=[];
  const num=x=>x===null?'—':Number(x).toPrecision(5);
  function node(tag,attrs,text){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
  function draw(){
    const reference=el('intrinsic-display').value==='reference';
    const points=reference?data.display.centers:data.display.raw;
    const ext=[];
    points.forEach((p,s)=>p.forEach((xy,i)=>{if(data.display.support[s][i]){const r=reference?data.display.ellipse[s][i][0]:0;ext.push([xy[0]-r,xy[1]-r],[xy[0]+r,xy[1]+r]);}}));
    const xs=ext.map(p=>p[0]),ys=ext.map(p=>p[1]);
    let x0=Math.min(...xs)-.18,y0=Math.min(...ys)-.18,w=Math.max(...xs)-x0+.18,h=Math.max(...ys)-y0+.18;
    const extent=Math.max(w,h);w=extent;h=extent;
    const zoom=Number(el('intrinsic-zoom').value);
    if(zoom>1){const focus=[];if(current)points.forEach((p,s)=>current.indices.forEach(i=>{if(data.display.support[s][i])focus.push(p[i]);}));
      const cx=focus.length?focus.reduce((t,p)=>t+p[0],0)/focus.length:x0+w/2;
      const cy=focus.length?focus.reduce((t,p)=>t+p[1],0)/focus.length:y0+h/2;
      w/=zoom;h/=zoom;x0=cx-w/2;y0=cy-h/2;}
    ['intrinsic-a','intrinsic-b'].forEach((id,s)=>{
      const svg=el(id);svg.replaceChildren();svg.setAttribute('viewBox',`${x0} ${y0} ${w} ${h}`);
      svg.append(node('line',{x1:x0,x2:x0+w,y1:0,y2:0,class:'axis'}));
      const p=points[s];
      if(current){const ids=current.indices.filter(i=>data.display.support[s][i]);
        if(ids.length===current.indices.length){
          if(current.family==='signed_double_area')svg.append(node('polygon',{points:ids.map(i=>p[i].join(',')).join(' '),class:'relation-face'}));
          else if(current.family==='asymmetry'){
            const mid=(is)=>[is.reduce((t,i)=>t+p[i][0],0)/is.length,is.reduce((t,i)=>t+p[i][1],0)/is.length];
            let pairs;
            if(current.key.endsWith('eye_width'))pairs=[[p[0],p[1]],[p[2],p[3]]];
            else if(current.key.endsWith('nose_to_mouth'))pairs=[[p[4],p[5]],[p[4],p[6]]];
            else {const anchor=current.key.endsWith('nose_to_eye')?p[4]:p[7];pairs=[[anchor,mid([0,1])],[anchor,mid([2,3])]];}
            pairs.forEach(([a,b])=>svg.append(node('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],class:'relation-line'})));
          }else svg.append(node('polyline',{points:ids.map(i=>p[i].join(',')).join(' '),class:'relation-line'}));
        }
      }
      p.forEach((xy,i)=>{if(!data.display.support[s][i])return;
        if(reference && data.calibrated){const [rx,ry,angle]=data.display.ellipse[s][i];svg.append(node('ellipse',{cx:xy[0],cy:xy[1],rx,ry,transform:`rotate(${angle} ${xy[0]} ${xy[1]})`,class:'uncertainty'}));}
        svg.append(node('circle',{cx:xy[0],cy:xy[1],r:reference?.006:.016,class:reference?'ref-point':'point'}));
        svg.append(node('text',{x:xy[0]+.035,y:xy[1]-.025,class:'point-label'},String(i)));
      });
    });
  }
  function detail(){const d=el('intrinsic-selected');d.replaceChildren();if(!current){d.textContent='No relation matches the selected filters.';draw();return;}
    const strong=document.createElement('strong');strong.textContent=current.point_names.map(x=>x.replaceAll('_',' ')).join(' → ');d.append(strong);
    const p=document.createElement('p');p.textContent=`${current.description}. Unit: ${current.unit}. Frozen tolerance: ${num(current.tolerance)}. Observed absolute difference: ${num(current.observed_absolute_gap)}.`;d.append(p);
    const p2=document.createElement('p');p2.textContent=`Reference-set gap: [${num(current.reference_gap_lower)}, ${num(current.reference_gap_upper)}]. ${statuses[current.status]}. Source A: ${current.source_intervals[0].reason}; B: ${current.source_intervals[1].reason}.`;d.append(p2);
    const code=document.createElement('code');code.textContent=current.array_columns.observed+' | '+current.array_columns.bounds;d.append(code);draw();}
  function select(row){current=row;for(const tr of el('intrinsic-rows').children)tr.setAttribute('aria-selected',String(Number(tr.dataset.index)===row.index));detail();}
  function render(){
    const query=el('intrinsic-search').value.toLowerCase().trim(), status=el('intrinsic-status').value;
    visible=data.rows.filter(r=>r.family===family && (status==='all'||r.status===status) && (!query||(r.key+' '+r.point_names.join(' ')).toLowerCase().includes(query)));
    el('intrinsic-count').textContent=`${visible.length} displayed / ${data.families.find(f=>f.family===family).prespecified_relations} prespecified`;
    if(!visible.some(r=>current&&r.index===current.index))current=visible[0]||null;
    el('intrinsic-rows').replaceChildren();
    for(const row of visible){const tr=document.createElement('tr');tr.dataset.index=row.index;tr.setAttribute('aria-selected',String(current&&current.index===row.index));
      const first=document.createElement('td'),button=document.createElement('button');button.type='button';button.className='pick';button.textContent=row.key;button.setAttribute('aria-label',row.point_names.join(' to '));button.addEventListener('click',()=>select(row));first.append(button);tr.append(first);
      const text=[num(row.observed_a),num(row.observed_b),row.reference_gap_lower===null?'Unavailable':`[${num(row.reference_gap_lower)}, ${num(row.reference_gap_upper)}]`,statuses[row.status]];
      text.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===3)td.className=row.status.startsWith('within')?'status-in':row.status.startsWith('outside')?'status-out':'';tr.append(td);});el('intrinsic-rows').append(tr);
    }
    for(const b of el('intrinsic-families').children)b.setAttribute('aria-pressed',String(b.dataset.family===family));detail();
  }
  for(const [key,label] of Object.entries(labels)){const b=document.createElement('button');b.type='button';b.textContent=label;b.dataset.family=key;b.setAttribute('aria-pressed',String(key===family));b.addEventListener('click',()=>{family=key;render();});el('intrinsic-families').append(b);}
  el('intrinsic-search').addEventListener('input',render);el('intrinsic-status').addEventListener('change',render);el('intrinsic-display').addEventListener('change',draw);el('intrinsic-zoom').addEventListener('input',draw);
  el('intrinsic-csv').addEventListener('click',()=>{
    const fields=['key','family','unit','tolerance','observed_a','observed_b','observed_absolute_gap','reference_set_center_a','reference_set_center_b','reference_gap_lower','reference_gap_upper','status'];
    const cell=value=>{let t=value===null?'':String(value);if(typeof value==='string'&&/^[\\s]*[=+@-]/.test(t))t="'"+t;return '"'+t.replaceAll('"','""')+'"';};
    const content=[fields.map(cell).join(','),...visible.map(r=>fields.map(f=>cell(r[f])).join(','))].join('\\r\\n')+'\\r\\n';
    const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'})),link=document.createElement('a');link.href=url;link.download='intrinsic-selected.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  if(data.calibration_bound_state!=='finite')el('intrinsic-display').querySelector('option[value=reference]').disabled=true;
  render();
})();
</script></section>'''

    from .joint_report import render_joint_section
    return html + render_joint_section(profile)
