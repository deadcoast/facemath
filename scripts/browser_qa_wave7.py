"""Execute the actual emitted joint-error workbench at desktop and mobile sizes."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def run(source: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    result={'format':'facemath.browser_qa.v7','started_utc':datetime.now(timezone.utc).isoformat(),
            'source':str(source),'checks':{},'viewports':[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        context=browser.new_context(accept_downloads=True);page=context.new_page()
        errors,requests=[],[]
        def attach(page):
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
        attach(page)
        try:
            page.goto(source.as_uri(),wait_until='load',timeout=10000)
            result['loading']='file URL'
        except Exception as exc:
            page.close();page=context.new_page();attach(page)
            page.set_content(source.read_text(),wait_until='load')
            result.update(loading='emitted HTML in memory; local file navigation blocked',navigation_error=str(exc))
        data=json.loads(page.locator('#joint-data').text_content())
        immutable=page.locator('#intrinsic-data').text_content()
        joint_immutable=page.locator('#joint-data').text_content()
        assert page.locator('#joint-mode option').count()==16
        assert page.locator('#joint-correlation rect').count()==256
        assert page.locator('#joint-shape circle[data-point]').count()==8
        assert data['joint_shape']['upper']<data['marginal_shape']['upper']
        assert 'not a separately fitted baseline' in page.locator('#joint').inner_text()
        result['checks']['retained_mode_and_covariance_inventory']=[16,256,8]
        for viewport in [{'width':1440,'height':1150},{'width':390,'height':844}]:
            page.set_viewport_size(viewport)
            page.locator('#joint').evaluate('(e)=>e.scrollIntoView({block:"start"})')
            page.locator('#joint-source').select_option('0')
            page.locator('#joint-mode').select_option('0')
            page.locator('#joint-amplitude').fill('0');page.locator('#joint-amplitude').dispatch_event('input')
            at_zero=page.locator('#joint-shape circle[data-point]').evaluate_all('(xs)=>xs.map(x=>[+x.getAttribute("cx"),+x.getAttribute("cy")])')
            page.locator('#joint-amplitude').fill('1');page.locator('#joint-amplitude').dispatch_event('input')
            at_one=page.locator('#joint-shape circle[data-point]').evaluate_all('(xs)=>xs.map(x=>[+x.getAttribute("cx"),+x.getAttribute("cy")])')
            assert at_zero!=at_one
            page.locator('#joint-mode').select_option('3');assert 'mode 4' in page.locator('#joint-mode-detail').inner_text()
            page.locator('#joint-source').select_option('1');assert 'Source B' in page.locator('#joint-mode-detail').inner_text()
            page.locator('#joint-correlation rect[data-i="0"][data-j="2"]').click()
            assert data['names'][0]+'.x' in page.locator('#joint-cell-detail').inner_text()
            with page.expect_download() as downloaded:page.locator('#joint-covariance-csv').click()
            name=out/('mobile-covariance.csv' if viewport['width']<800 else 'desktop-covariance.csv')
            downloaded.value.save_as(name)
            rows=list(csv.DictReader(name.open()))
            assert len(rows)==256
            for i,r in enumerate(rows):assert float(r['covariance'])==data['covariance'][1][i//16][i%16]
            page.locator('#joint-source').select_option('0');page.locator('#joint-mode').select_option('0')
            page.locator('#joint-amplitude').fill('0.75');page.locator('#joint-amplitude').dispatch_event('input')
            page.locator('#joint').evaluate('(e)=>e.scrollIntoView({block:"start"})')
            page.wait_for_timeout(80)
            assert page.evaluate('document.documentElement.scrollWidth')<=viewport['width']
            filename='mobile_joint.png' if viewport['width']<800 else 'desktop_joint.png'
            page.screenshot(path=str(out/filename),full_page=False)
            result['viewports'].append({'size':viewport,'csv_rows':len(rows),'interactive':True,'screenshot':filename})
        assert immutable==page.locator('#intrinsic-data').text_content()
        assert joint_immutable==page.locator('#joint-data').text_content()
        result['checks'].update(display_operations_preserve_measurements=True,csv_exact_full_covariance=True,
            local_evidence_links=all((source.parent/name).is_file() for name in ('arrays.npz','intrinsic.json','localization_calibration.json')),
            no_horizontal_overflow=True)
        result['errors']=errors;result['remote_requests']=requests
        assert not errors,errors
        assert not requests,requests
        browser.close()
    result['finished_utc']=datetime.now(timezone.utc).isoformat();result['passed']=True
    (out/'browser_validation.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(run(Path(a.report).resolve(),Path(a.out)),indent=2))
