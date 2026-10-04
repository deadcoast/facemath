from pathlib import Path
path=Path(__file__).resolve().parents[2]/'src/facemath/observation/annotation_ui.py'
text=path.read_text(encoding='utf-8')
replacements={
 "FACEMATH / WAVE 04 / INDEPENDENT REFERENCE":"FACEMATH / SOURCE REFERENCE",
 "function draw(){if(!image.complete)return;const r=canvas.getBoundingClientRect();":"function draw(){if(!image.complete)return;const r=canvasBox();",
 "const x=p.xy[0]*view.scale+view.tx,y=p.xy[1]*view.scale+view.ty;":"const [x,y]=canvasAt(...p.xy);",
 "$('zoom-state').textContent=(view.scale/baseScale).toFixed(1)+'×';}":"$('zoom-state').textContent=(view.scale/baseScale).toFixed(1)+'×';if(lastPointer)inspectAt(...lastPointer);}",
 "$('export').disabled=count!==8||!$('rater').value.trim();":"$('export').disabled=count!==8||!activeRater||activeRater!==$('rater').value.trim()||!!pendingDraft;",
 "if(p[0]<0||p[1]<0||p[0]>width-1||p[1]>height-1)":"if(p[0]<-1e-7||p[1]<-1e-7||p[0]>width-1+1e-7||p[1]>height-1+1e-7)",
 "xy:p.map(v=>Math.round(v*1000)/1000)":"xy:p.map((v,i)=>Math.round(Math.max(0,Math.min((i?height:width)-1,v))*1000)/1000)",
 "canvas.addEventListener('pointermove',e=>{const [x,y]=pointer(e);lastPointer=[x,y];if(drag){view.tx=drag.tx+x-drag.x;view.ty=drag.ty+y-drag.y;draw();}const p=sourceAt(x,y);":"function inspectAt(x,y){const p=sourceAt(x,y);",
 "lens.drawImage(image,p[0]-12,p[1]-12,24,24,0,0,148,148);":"lens.drawImage(image,p[0]+.5-12,p[1]+.5-12,24,24,0,0,148,148);",
 "lens.lineTo(148,74);lens.stroke();});":"lens.lineTo(148,74);lens.stroke();}\ncanvas.addEventListener('pointermove',e=>{const [x,y]=pointer(e);lastPointer=[x,y];if(drag){view.tx=drag.tx+x-drag.x;view.ty=drag.ty+y-drag.y;draw();}inspectAt(x,y);});",
 "if(e.target.matches('input,textarea'))return;":"if(e.target.matches('input,textarea'))return;if(e.altKey&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();view.tx+=(e.key==='ArrowRight'?32:e.key==='ArrowLeft'?-32:0);view.ty+=(e.key==='ArrowDown'?32:e.key==='ArrowUp'?-32:0);draw();return;}",
 "if(points.some(p=>p===null)||!$('rater').value.trim())return null;":"if(points.some(p=>p===null)||!activeRater||activeRater!==$('rater').value.trim()||pendingDraft)return null;",
 "interface:'facemath.reference-workbench.v4',":"interface:'facemath.reference-workbench.v20',coordinate_capture:data.coordinate_capture,",
}
for old,new in replacements.items():
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
path.write_text(text,encoding='utf-8',newline='\n')
