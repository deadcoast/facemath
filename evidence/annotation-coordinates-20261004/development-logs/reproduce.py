from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parents[2]
out=root/'evidence/annotation-coordinates-20261004/parent'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1280,'height':1000},device_scale_factor=1)
    page.goto((out/'workbench.html').as_uri())
    page.wait_for_function('image.complete && image.naturalWidth > 0')
    page.locator('#rater').fill('synthetic-coordinate-reproducer');page.locator('#rater').press('Tab')
    # Locate the actual painted source pixel center using the Canvas CTM and
    # DOM content box, independently of the application's inverse transform.
    oracle='''([x,y])=>{const r=canvas.getBoundingClientRect(),s=getComputedStyle(canvas),t=ctx.getTransform();
      const left=parseFloat(s.borderLeftWidth)+parseFloat(s.paddingLeft),top=parseFloat(s.borderTopWidth)+parseFloat(s.paddingTop);
      const cw=r.width-left-parseFloat(s.borderRightWidth)-parseFloat(s.paddingRight),ch=r.height-top-parseFloat(s.borderBottomWidth)-parseFloat(s.paddingBottom);
      return [r.left+left+(view.tx+(x+.5)*view.scale)*t.a*cw/canvas.width,r.top+top+(view.ty+(y+.5)*view.scale)*t.d*ch/canvas.height];}'''
    expected=[[8,8],[12,8],[28,8],[36,8],[24,15],[15,21],[32,21],[24,28]]
    for i,xy in enumerate(expected):
        page.locator('#point-list button').nth(i).click()
        page.mouse.click(*page.evaluate(oracle,xy))
    with page.expect_download() as dl:page.locator('#export').click()
    dl.value.save_as(out/'actual-reference.json')
    actual=json.loads((out/'actual-reference.json').read_text())
    errors=[[b-a for a,b in zip(a,b)] for a,b in zip(expected,actual['xy'])]
    result={'expected_source_centers':expected,'exported':actual['xy'],'errors_px':errors,
      'maximum_absolute_error_px':max(abs(v) for row in errors for v in row),
      'browser':browser.version,'passed':all(abs(v)<=.002 for row in errors for v in row),
      'kind':'automated indexed-raster control, not human annotation'}
    (out/'coordinate-failure.json').write_text(json.dumps(result,indent=2))
    page.screenshot(path=str(out/'coordinate-failure.png'),full_page=True)
    browser.close()
print(json.dumps(result,indent=2))
assert not result['passed'], 'Expected the inherited coordinate defect to reproduce'
