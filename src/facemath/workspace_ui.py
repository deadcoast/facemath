"""Image selection and inspection shell; all numerical readouts remain shared."""
from .observation.artifacts import REPORT_TOKENS


def page() -> str:
    return HTML.replace("__TOKENS__", REPORT_TOKENS)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FACEMATH · Compare two images</title><style>
__TOKENS__
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:30px 30px 70px}header{border-bottom:1px solid var(--line);padding-bottom:18px}
h1{font-size:32px;letter-spacing:-.025em;line-height:1.2;margin:8px 0 14px}h2{font-size:21px;margin:30px 0 12px}h3{font-size:17px}
p{color:var(--muted);max-width:92ch}.eyebrow{color:var(--accent);font:12px ui-monospace,monospace;letter-spacing:.12em}
a{color:var(--accent)}a:hover{color:var(--ink)}a:active{color:var(--accent)}button,a,input,summary{touch-action:manipulation}
button,input{font:inherit}button,.action{cursor:pointer;border:1px solid var(--line);padding:10px 16px;background:var(--panel);color:var(--ink);min-height:44px}
button:hover{border-color:var(--accent)}button:active{background:var(--line)}button:disabled{cursor:wait;opacity:.55}
.primary{background:var(--accent);color:var(--bg);font-weight:650}.primary:hover{background:var(--ink)}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}fieldset{border:0;margin:0;padding:0;min-width:0}legend{font-size:21px;font-weight:650;padding:20px 0 12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.source{background:var(--panel);border:1px solid var(--line);padding:16px;min-width:0}
.source label{font-size:19px;font-weight:650;display:block;margin-bottom:10px}.source input{display:block;width:100%;max-width:100%;color:var(--muted)}
input::file-selector-button{font:inherit;padding:8px 12px;margin-right:10px;background:var(--bg);border:1px solid var(--line);color:var(--accent);cursor:pointer}
.source img{display:block;width:100%;height:240px;object-fit:contain;margin-top:16px;background:var(--bg)}[hidden]{display:none!important}
.filename{overflow-wrap:anywhere;color:var(--muted);margin:8px 0 0;font-size:13px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0}
input[type=number]{background:var(--panel);color:var(--ink);border:1px solid var(--line);padding:8px;width:110px;min-height:44px}
details{margin:14px 0}summary{cursor:pointer;color:var(--accent)}#message{border-left:3px solid var(--accent);background:var(--panel);padding:16px;margin:16px 0;overflow-wrap:anywhere}
#history{padding:0;list-style:none}#history li{display:flex;align-items:center;gap:16px;border-top:1px solid var(--line);padding:12px 0;flex-wrap:wrap}
#history button{text-align:left;max-width:100%;overflow-wrap:anywhere}#history small{color:var(--muted)}.result-nav{display:flex;gap:18px;flex-wrap:wrap}
.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0;background:var(--panel);padding:12px;border:1px solid var(--line)}
figure img{width:100%;display:block}figcaption,small{color:var(--muted);font-size:13px}small{display:block}.blend{position:relative}.blend img:last-child{position:absolute;inset:0;opacity:.5}
input[type=range]{width:100%;cursor:pointer;min-height:32px}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
.scroll{overflow:auto;border:1px solid var(--line)}th,td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{background:var(--panel)}
#structure table{min-width:790px}#edge-comparison table{min-width:1120px}caption{text-align:left;padding:12px 14px;color:var(--muted)}
#bilateral table{min-width:1000px}#bilateral summary{min-height:44px;padding:10px 0}#bilateral .bilateral-gallery{grid-template-columns:repeat(2,minmax(0,1fr));max-width:850px}#bilateral svg{display:block;width:100%}
@media(max-width:760px){#bilateral .bilateral-gallery{grid-template-columns:1fr}}
.number{white-space:nowrap}.panel{background:var(--panel);border-left:3px solid var(--accent);padding:18px}code{overflow-wrap:anywhere}
*::-webkit-scrollbar{width:12px;height:12px}*::-webkit-scrollbar-track{background:var(--scroll-track)}*::-webkit-scrollbar-thumb{background:var(--scroll-thumb);border:3px solid var(--scroll-track)}
*::-webkit-scrollbar-thumb:hover{background:var(--scroll-hover)}*::-webkit-scrollbar-thumb:active{background:var(--scroll-active)}
@media(forced-colors:active){:root{scrollbar-color:auto}:focus-visible{outline-color:Highlight}}
@media(max-width:760px){main{padding:20px 14px}.pair,.gallery{grid-template-columns:1fr}h1{font-size:28px}.source img{height:210px}}
</style></head><body><main>
<header><div class="eyebrow">FACEMATH / IMAGE COMPARISON</div><h1>Compare two portraits</h1>
<p>Inspect differences in eye spacing, facial proportions, regional shape and image edges. Measurements describe the visible geometry; pose and expression affect the result.</p></header>
<form id="compare"><fieldset id="inputs"><legend>Choose your image pair</legend><div class="pair">
<div class="source"><label for="image-a">A · reference image</label><input id="image-a" type="file" accept="image/png,image/jpeg" required><img id="preview-a" alt="Selected image A" hidden><p class="filename" id="name-a">PNG or JPEG · up to 16 MiB</p></div>
<div class="source"><label for="image-b">B · comparison image</label><input id="image-b" type="file" accept="image/png,image/jpeg" required><img id="preview-b" alt="Selected image B" hidden><p class="filename" id="name-b">PNG or JPEG · up to 16 MiB</p></div></div>
<p>Use one clearly visible face per image. Analysis uses the stored image orientation; selection previews may reflect your browser's orientation handling.</p>
<details><summary>Comparison settings</summary><label for="tolerance">Landmark and edge tolerance (eye spans)</label> <input id="tolerance" type="number" min="0.001" max="1" step="0.001" value="0.05"><p>Affects within-tolerance counts. All structural, regional, edge, optical-flow and sensitivity calculations remain enabled. Research calibration and qualification can be attached with the existing CLI.</p></details>
<div class="controls"><button class="primary" type="submit" id="start">Compare images</button></div></fieldset></form>
<div id="message" role="status" aria-live="polite" hidden></div><div class="controls"><button id="cancel" type="button" hidden>Cancel comparison</button><button id="reconnect" type="button" hidden>Reconnect</button></div>
<section id="results" aria-labelledby="result-title" hidden><h2 id="result-title" tabindex="-1">Comparison results</h2><p id="pair-label"></p>
<nav class="result-nav" aria-label="Comparison results"><a id="full-report" target="_blank" rel="noopener">Open complete report</a><a id="download">Download comparison</a><a id="raw" target="_blank" rel="noopener">Measurement data</a><a href="#structure">Facial structure</a><a href="#bilateral">Bilateral configuration</a><a href="#edge-comparison">Image edges</a></nav>
<h3>Registered images</h3><div class="gallery"><figure><img id="registered-a" alt="Registered image A"><figcaption>A · reference</figcaption></figure><figure><img id="registered-b" alt="Registered image B"><figcaption>B · comparison</figcaption></figure><figure><div class="blend"><img id="blend-a" alt="Registered A in blend"><img id="blend-b" alt="Registered B in blend"></div><figcaption><label for="opacity">Blend A → B</label><input id="opacity" type="range" min="0" max="100" value="50"></figcaption></figure></div>
<div id="result-readout"></div></section>
<section aria-labelledby="saved-title"><h2 id="saved-title">Saved comparisons</h2><p>Images and results stay in the local comparison directory. Failed and cancelled runs keep their inputs and logs.</p><ul id="history"><li>No comparisons yet. Choose two images to begin.</li></ul></section>
</main><script>
'use strict';
const $ = id => document.getElementById(id);
let current = null, timer = null, generation = 0, busy = false, previewURLs = {};
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
  return data;
}
function message(text) { $('message').hidden = false; $('message').textContent = text; }
function setBusy(value) { busy = value; $('inputs').disabled = value; $('start').textContent = value ? 'Comparing…' : 'Compare images'; $('cancel').hidden = !value; }
for (const side of ['a','b']) {
  $('image-' + side).addEventListener('change', () => {
    const file = $('image-' + side).files[0];
    if (previewURLs[side]) URL.revokeObjectURL(previewURLs[side]);
    $('preview-' + side).hidden = !file;
    $('name-' + side).textContent = file ? file.name : 'PNG or JPEG · up to 16 MiB';
    if (file) { previewURLs[side] = URL.createObjectURL(file); $('preview-' + side).src = previewURLs[side]; }
  });
}
function encoded(file) {
  if (!file || file.size > 16 * 1024 * 1024) return Promise.reject(Error('Choose two images, each at most 16 MiB.'));
  return new Promise((resolve, reject) => { const reader = new FileReader();
    reader.onerror = () => reject(Error('Could not read ' + file.name));
    reader.onload = () => resolve({name:file.name, data:reader.result.split(',')[1]}); reader.readAsDataURL(file);
  });
}
async function history() {
  const rows = await api('api/jobs'); $('history').replaceChildren();
  if (!rows.length) { const li = document.createElement('li'); li.textContent = 'No comparisons yet. Choose two images to begin.'; $('history').append(li); }
  for (const row of rows) {
    const li = document.createElement('li'), button = document.createElement('button'), detail = document.createElement('small');
    button.type = 'button'; button.textContent = row.inputs.map(x => x.name).join(' / ');
    button.addEventListener('click', () => { if (busy && current !== row.id) { message('Wait for the current comparison or cancel it before opening another.'); return; } openRun(row.id); });
    detail.textContent = row.status + ' · ' + new Date(row.created).toLocaleString(); li.append(button, detail); $('history').append(li);
  }
  return rows;
}
async function showResult(record, revision) {
  const root = 'runs/' + record.id + '/';
  const response = await fetch(root + 'readout.html');
  if (!response.ok) throw Error('The saved result could not be loaded. Reconnect to try again.');
  const html = await response.text(); if (revision !== generation) return;
  $('result-readout').innerHTML = html;
  $('pair-label').textContent = record.inputs.map(x => x.name).join(' compared with ');
  $('full-report').href = root + 'result/report.html'; $('download').href = root + 'comparison.zip'; $('raw').href = root + 'result/summary.json';
  for (const side of ['a','b']) { $('registered-' + side).src = root + 'result/canonical_' + side + '.png'; $('blend-' + side).src = root + 'result/canonical_' + side + '.png'; }
  $('opacity').value = 50; $('blend-b').style.opacity = .5; $('results').hidden = false;
}
async function poll(revision) {
  if (revision !== generation || !current) return;
  try {
    const record = await api('api/jobs/' + current); if (revision !== generation) return;
    message(record.message); $('message').dataset.state = record.status; $('reconnect').hidden = true;
    if (record.status === 'running') { setBusy(true); timer = setTimeout(() => poll(revision), 1000); return; }
    setBusy(false);
    if (record.status === 'complete') { await showResult(record, revision); }
    else {
      $('results').hidden = true;
      const link = document.createElement('a'); link.href = 'runs/' + record.id + '/stderr.log'; link.target = '_blank'; link.rel = 'noopener'; link.textContent = 'Open execution log';
      $('message').append(document.createTextNode(' '), link);
    }
    await history();
  } catch (error) { if (revision !== generation) return; message('Connection interrupted: ' + error.message + ' Your run is retained. Reconnect to check it before starting another.'); $('reconnect').hidden = false; }
}
function openRun(id) { clearTimeout(timer); current = id; generation++; $('results').hidden = true; location.hash = 'run=' + id; poll(generation); }
$('compare').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return;
  const revision = ++generation; clearTimeout(timer); current = null; setBusy(true); $('results').hidden = true; $('cancel').hidden = true; message('Reading and saving the selected images…');
  try {
    const [a,b] = await Promise.all(['a','b'].map(s => encoded($('image-' + s).files[0])));
    const record = await api('api/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({a,b,tolerance:Number($('tolerance').value)})});
    for (const side of ['a','b']) $('preview-' + side).src = 'runs/' + record.id + '/preview_' + side + '.png';
    openRun(record.id); await history();
  } catch (error) { if (revision !== generation) return; message(error.message); setBusy(false); await history().catch(() => { $('reconnect').hidden = false; }); }
});
$('cancel').addEventListener('click', async () => {
  if (!current) return; $('cancel').disabled = true;
  try { await api('api/jobs/' + current + '/cancel', {method:'POST', headers:{'Content-Type':'application/json'},body:'{}'}); await poll(generation); }
  catch (error) { message(error.message); } finally { $('cancel').disabled = false; }
});
$('reconnect').addEventListener('click', async () => {
  try { const rows = await history(); const active = rows.find(r => r.status === 'running'); if (active || current) openRun(active ? active.id : current); else { setBusy(false); $('reconnect').hidden = true; message('Connected. Choose two images to compare.'); } }
  catch (error) { message(error.message); }
});
$('opacity').addEventListener('input', event => { $('blend-b').style.opacity = event.target.value / 100; });
for (const link of document.querySelectorAll('.result-nav a[href^="#"]')) {
  link.addEventListener('click', event => { event.preventDefault(); document.querySelector(link.getAttribute('href'))?.scrollIntoView(); });
}
history().then(rows => { const active = rows.find(r => r.status === 'running'); const id = /^#run=([0-9a-f]{32})$/.exec(location.hash)?.[1]; if (active || id) openRun(active ? active.id : id); }).catch(error => { message(error.message); $('reconnect').hidden = false; });
</script></body></html>'''
