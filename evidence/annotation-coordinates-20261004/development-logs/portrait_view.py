from pathlib import Path
import json
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2]
out=root/'evidence/annotation-coordinates-20261004/portrait-view'
out.mkdir(exist_ok=False)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1400,'height':1050},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((out.parent/'portrait-workbench.html').as_uri())
    page.wait_for_function('image.complete && image.naturalWidth>0')
    assert page.locator('#progress-label').inner_text()=='0 / 8 resolved'
    assert page.locator('#export').is_disabled()
    assert page.evaluate('points.every(p=>p===null)')
    page.locator('#zoom-in').click();page.locator('#fit').click()
    page.screenshot(path=str(out/'blank-portrait.png'),full_page=True)
    assert not errors
    record={'image_shape':page.evaluate('data.image_shape'),'reference_kind':page.evaluate('data.kind'),
            'points_entered':0,'native_predictions_shown':False,'fit_and_zoom_exercised':True,
            'scope':'blank real-portrait workspace; no annotation or anatomical validation',
            'browser':browser.version,'errors':errors}
    (out/'view.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2));browser.close()
