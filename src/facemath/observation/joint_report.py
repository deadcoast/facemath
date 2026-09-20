"""Offline display of retained joint sets; sliders NEVER recalculate scores."""
from __future__ import annotations

import json
from .intrinsic import IntrinsicProfile
from .reference import REFERENCE_NAMES


def render_joint_section(profile: IntrinsicProfile) -> str:
    audit = profile.joint
    if audit is None:
        return ""
    data = {"names": list(REFERENCE_NAMES), "centers": profile.source_centers.tolist(),
            "support": profile.source_supported.tolist(), "q": profile.quantile,
            "covariance": audit.covariance.tolist(), "eigenvalues": audit.mode_eigenvalues.tolist(),
            "modes": audit.mode_vectors.tolist(), "joint_shape": profile.shape.to_dict(),
            "marginal_shape": audit.marginal_shape.to_dict(), "audit": audit.to_dict(),
            "arrays": {"covariance": "intrinsic_joint_covariance[source,16,16]",
                       "modes": "intrinsic_joint_mode_vectors[source,coordinate,mode]",
                       "eigenvalues": "intrinsic_joint_mode_eigenvalues[source,mode]"}}
    encoded = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return '''<section id="joint" aria-labelledby="joint-heading">
<style>
#joint{margin-top:38px;border-top:1px solid var(--line);padding-top:24px}#joint .kicker{color:#6de2c4;font:11px ui-monospace,monospace;letter-spacing:.15em}#joint h2{margin:10px 0;font-size:28px;letter-spacing:-.025em}#joint .joint-intro{max-width:95ch;color:#b4c6d3}#joint .joint-grid{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;margin:20px 0}#joint .joint-panel{background:#111d29;border:1px solid #304553;padding:18px;min-width:0}#joint h3{margin:0 0 12px;font-size:16px}#joint .controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end}#joint label{font-size:12px;color:#b4c6d3;display:block}#joint select,#joint button{display:block;padding:10px;color:#edf5fb;background:#152431;border:1px solid #415769;font:12px system-ui}#joint input[type=range]{display:block;width:220px;max-width:100%;accent-color:#6de2c4}#joint svg{display:block;width:100%;background:#0c1620;margin:16px 0}#joint .joint-meta{font:12px/1.65 ui-monospace,monospace;color:#b7cbd6;overflow-wrap:anywhere;min-height:42px}#joint .joint-card-row{display:grid;grid-template-columns:1fr 1fr;gap:14px}#joint .joint-card{border:1px solid #304553;padding:16px;background:#131e28}#joint .joint-card strong{display:block;font:23px ui-monospace,monospace;color:#edf5fb;margin:8px 0}#joint .joint-card small,#joint .explanation{font:12px/1.7 system-ui;color:#b4c6d3}#joint .callout{border-left:3px solid #6de2c4;padding:12px 18px;background:#152431;font:13px/1.7 system-ui}#joint button{cursor:pointer}#joint :focus-visible{outline:2px solid #6de2c4;outline-offset:3px}#joint .heat-cell{cursor:crosshair}#joint code{font-size:12px}#joint .joint-links{display:flex;flex-wrap:wrap;gap:15px;font-size:13px;margin:16px 0}#joint a{color:#6de2c4}
@media(max-width:800px){#joint .joint-grid,#joint .joint-card-row{grid-template-columns:1fr}#joint .joint-panel{padding:12px}#joint h2{font-size:24px}}
</style>
<div class="kicker">WAVE 07 / CORRELATED REFERENCE ERROR</div>
<h2 id="joint-heading">One error field. Eight connected landmarks.</h2>
<p class="joint-intro">A shared localization shift should not become eight unrelated anatomical changes. This inspector displays the frozen full-shape error set and its marginal-product relaxation. It does not move the observations or update any measurement.</p>
<div class="joint-card-row"><div class="joint-card"><small>Shared full16 set · proper-rotation shape RMS</small><strong id="joint-bound"></strong><small>Joint dependencies retained through normalization and refitting.</small></div><div class="joint-card"><small>Same set → product of marginal ellipses</small><strong id="joint-relaxed-bound"></strong><small>Same centers, marginal blocks and q. This discards dependence; it is not a separately fitted baseline.</small></div></div>
<div class="joint-grid"><div class="joint-panel"><h3>Coherent displacement mode</h3><div class="controls"><label>Source<select id="joint-source"><option value="0">A</option><option value="1">B</option></select></label><label>Mode<select id="joint-mode"></select></label><label>Position on admitted mode<input id="joint-amplitude" type="range" min="-1" max="1" step="0.01" value="0.75"></label></div><svg id="joint-shape" viewBox="0 0 460 410" role="img" aria-label="Coherent displacement of eight source reference points"></svg><div class="joint-meta" id="joint-mode-detail" role="status"></div><p class="explanation">One shared latent coefficient moves all points together. The solid outline is the set center; displaced markers are a selected admissible error, not a new prediction. Eigenmode signs are conventional; tied eigenbases are not unique physical mechanisms.</p></div>
<div class="joint-panel"><h3>Full coordinate correlation</h3><svg id="joint-correlation" viewBox="0 0 420 410" role="img" aria-label="Sixteen by sixteen coordinate correlation matrix"></svg><div class="joint-meta" id="joint-cell-detail" role="status">Select a cell to inspect its exact covariance and correlation.</div><div class="controls"><button type="button" id="joint-covariance-csv">Export source matrix · CSV</button></div><p class="explanation">Coordinate order is point 0 x,y through point 7 x,y. Teal is positive, amber is negative. Correlation describes the fitted error model—not similarity between people or confidence in a match.</p></div></div>
<div class="callout" id="joint-common-mode"></div>
<div class="joint-links"><a href="arrays.npz">Raw arrays</a><a href="localization_calibration.json">Development fit + calibration</a><a href="intrinsic.json">Full relation inventory</a></div><p class="explanation">The plotted matrix is the development-derived error covariance transported into this source chart. The original perturbation covariance remains separately exported. Narrower algebraic bounds do not repair dataset shift or establish human detector accuracy.</p>
<script id="joint-data" type="application/json">'''+encoded+'''</script><script>
(()=>{'use strict';const d=JSON.parse(document.getElementById('joint-data').textContent),el=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';let active=[0,0];
function node(tag,attrs,text){let n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
function interval(s){return s.available?'['+s.lower.toFixed(6)+', '+s.upper.toFixed(6)+']':'Not bounded';}
el('joint-bound').textContent=interval(d.joint_shape);el('joint-relaxed-bound').textContent=interval(d.marginal_shape);
for(let i=0;i<16;i++){let o=document.createElement('option');o.value=i;o.textContent=String(i+1).padStart(2,'0');el('joint-mode').append(o);}
const label=i=>d.names[Math.floor(i/2)]+(i%2?'.y':'.x');
function inspect(s,i,j){const c=d.covariance[s],den=Math.sqrt(Math.max(0,c[i][i]*c[j][j]));el('joint-cell-detail').textContent=label(i)+' × '+label(j)+' | covariance '+c[i][j].toPrecision(9)+' | correlation '+(den?c[i][j]/den:0).toFixed(7);active=[i,j];}
function heat(s){const svg=el('joint-correlation');svg.replaceChildren();const c=d.covariance[s],size=19,x0=62,y0=35;
for(let i=0;i<16;i++){svg.append(node('text',{x:x0+i*size+size/2,y:y0-8,'text-anchor':'middle',fill:'#b4c6d3','font-size':10},i));svg.append(node('text',{x:x0-9,y:y0+i*size+13,'text-anchor':'end',fill:'#b4c6d3','font-size':10},i));for(let j=0;j<16;j++){let den=Math.sqrt(Math.max(0,c[i][i]*c[j][j])),r=den?Math.max(-1,Math.min(1,c[i][j]/den)):0;let cell=node('rect',{x:x0+j*size,y:y0+i*size,width:size-1,height:size-1,fill:r>=0?'#6de2c4':'#f3bd77',opacity:0.12+0.88*Math.abs(r),class:'heat-cell','data-i':i,'data-j':j,tabindex:0,role:'button','aria-label':label(i)+' with '+label(j)});cell.addEventListener('click',()=>inspect(s,i,j));cell.addEventListener('focus',()=>inspect(s,i,j));cell.addEventListener('mouseenter',()=>inspect(s,i,j));svg.append(cell);}}
svg.append(node('text',{x:210,y:374,'text-anchor':'middle',fill:'#b4c6d3','font-size':11},'0x 0y 1x 1y … 7x 7y · reference8 order'));inspect(s,...active);}
function draw(){const s=+el('joint-source').value,k=+el('joint-mode').value,a=+el('joint-amplitude').value,q=d.q||0,lam=Math.max(0,d.eigenvalues[s][k]),base=d.centers[s],v=d.modes[s].map(row=>row[k]),scale=q*Math.sqrt(lam);let p=base.map((x,i)=>[x[0]+a*scale*v[2*i],x[1]+a*scale*v[2*i+1]]);
const all=base.flatMap((x,i)=>[x,[x[0]+scale*v[2*i],x[1]+scale*v[2*i+1]],[x[0]-scale*v[2*i],x[1]-scale*v[2*i+1]]]);let xs=all.map(x=>x[0]),ys=all.map(x=>x[1]),xmin=Math.min(...xs)-.2,xmax=Math.max(...xs)+.2,ymin=Math.min(...ys)-.2,ymax=Math.max(...ys)+.2,z=Math.min(390/(xmax-xmin),345/(ymax-ymin)),px=x=>[230+(x[0]-(xmin+xmax)/2)*z,195+(x[1]-(ymin+ymax)/2)*z];let svg=el('joint-shape');svg.replaceChildren();
for(const [i,j]of [[0,1],[2,3],[0,4],[3,4],[4,5],[4,6],[5,6],[5,7],[6,7]]){let b=px(base[i]),c=px(base[j]);svg.append(node('line',{x1:b[0],y1:b[1],x2:c[0],y2:c[1],stroke:'#607d90','stroke-width':1.3}));}
for(let i=0;i<8;i++){let b=px(base[i]),c=px(p[i]);svg.append(node('line',{x1:b[0],y1:b[1],x2:c[0],y2:c[1],stroke:'#6de2c4','stroke-width':1.5,'stroke-dasharray':'3 3'}));svg.append(node('circle',{cx:b[0],cy:b[1],r:4,fill:'#b4c6d3'}));svg.append(node('circle',{cx:c[0],cy:c[1],r:5,fill:'#6de2c4','data-point':i}));svg.append(node('text',{x:c[0]+9,y:c[1]-7,fill:'#ecf6fb','font-size':11},i));}
const total=d.eigenvalues[s].reduce((x,y)=>x+y,0);el('joint-mode-detail').textContent='Source '+(s?'B':'A')+' · mode '+(k+1)+' · amplitude '+a.toFixed(2)+' · '+(100*lam/total).toFixed(2)+'% of model variance (not resemblance)';el('joint-common-mode').textContent='Common-translation subspace contains '+(100*d.audit.common_translation_variance_fraction[s]).toFixed(2)+'% of this source’s fitted error variance. Zero-sum relative expressions cancel coherent translation before uncertainty is bounded.';
heat(s);}
['joint-source','joint-mode'].forEach(id=>el(id).addEventListener('change',draw));el('joint-amplitude').addEventListener('input',draw);
el('joint-covariance-csv').addEventListener('click',()=>{const s=+el('joint-source').value,rows=['coordinate_i,coordinate_j,covariance'];for(let i=0;i<16;i++)for(let j=0;j<16;j++)rows.push([label(i),label(j),d.covariance[s][i][j]].join(','));let url=URL.createObjectURL(new Blob([rows.join('\\r\\n')+'\\r\\n'],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='joint-source-'+(s?'b':'a')+'-covariance.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});draw();})();
</script></section>'''
