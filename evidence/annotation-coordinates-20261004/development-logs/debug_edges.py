from pathlib import Path
import json,sys
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'))
from browser_qa_annotation import ORACLE
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=3)
    page.goto((root/'evidence/annotation-coordinates-20261004/corrected-01/workbench.html').as_uri())
    page.wait_for_function('image.complete && image.naturalWidth>0')
    page.locator('#rater').fill('edge-repro');page.locator('#rater').press('Tab')
    page.evaluate('''()=>{window.clicks=[];canvas.addEventListener('pointerdown',e=>clicks.push({client:[e.clientX,e.clientY],source:sourceAt(...pointer(e)),box:canvasBox(),view:{...view}}),{capture:true});}''')
    for i,xy in enumerate([[0,0],[47,0],[0,31],[47,31]]):
        page.locator('#point-list button').nth(i).click();page.locator('#stage').scroll_into_view_if_needed()
        page.mouse.click(*page.evaluate(ORACLE,xy))
    result=page.evaluate('({clicks,points})')
    (root/'evidence/annotation-coordinates-20261004/edge-rounding-failure.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    browser.close()
