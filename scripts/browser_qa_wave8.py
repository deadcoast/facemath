"""Exercise emitted category calibration and source-bound pair reports without recomputation."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def run(study: Path, pair: Path, out: Path) -> dict:
    out.mkdir(parents=True,exist_ok=False)
    result={'format':'facemath.browser_qa.v8','started_utc':datetime.now(timezone.utc).isoformat(),
            'reports':{},'checks':{},'viewports':[]}
    errors,requests=[],[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(accept_downloads=True)
        def load(source,label):
            page=context.new_page()
            def attach(page):
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('request',lambda r:requests.append(r.url)if r.url.startswith(('http:','https:'))else None)
            attach(page)
            try:
                page.goto(source.as_uri(),wait_until='load',timeout=10000)
                result['reports'][label]={'loading':'file URL','source':str(source)}
            except Exception as exc:
                page.close();page=context.new_page();attach(page)
                page.set_content(source.read_text(),wait_until='load')
                result['reports'][label]={'loading':'emitted HTML in memory after local file navigation was blocked',
                    'source':str(source),'navigation_error':str(exc)}
            return page
        page=load(study,'study');data=json.loads(page.locator('#routing-data').text_content())
        frozen=page.locator('#data').text_content();routing_frozen=page.locator('#routing-data').text_content()
        count=len(data['categories']);assert page.locator('#routing-summary tbody tr').count()==count
        assert page.locator('#routing-category option').count()==count
        for viewport in({'width':1440,'height':1050},{'width':390,'height':844}):
            page.set_viewport_size(viewport);label='mobile'if viewport['width']<800 else'desktop'
            for category in data['categories']:
                page.locator('#routing-category').select_option(category['stratum'])
                assert page.locator('#routing-scores tbody tr').count()==len(category['group_scores'])
                detail=page.locator('#routing-detail').inner_text()
                assert category['status'].replace('_',' ')in detail
                with page.expect_download()as download:page.locator('#routing-export').click()
                path=out/f'{label}-{category["stratum"]}-scores.csv';download.value.save_as(path)
                rows=list(csv.DictReader(path.open()));assert len(rows)==len(category['group_scores'])
                lookup=dict(category['group_scores'])
                for row in rows:
                    expected=lookup[row['group_id']]
                    assert (row['score']==''and row['score_available']=='false')if expected is None else float(row['score'])==expected
                    assert row['deployed_q']==''if category['quantile'] is None else float(row['deployed_q'])==category['quantile']
            page.locator('#routing-category').select_option('degraded')
            page.locator('#acquisition-routing').evaluate('(e)=>e.scrollIntoView({block:"start"})')
            page.wait_for_timeout(50)
            assert page.evaluate('document.documentElement.scrollWidth')<=viewport['width']
            filename=f'{label}_acquisition.png';page.screenshot(path=str(out/filename))
            result['viewports'].append({'report':'study','viewport':viewport,'screenshot':filename,'categories_checked':count})
        assert frozen==page.locator('#data').text_content() and routing_frozen==page.locator('#routing-data').text_content()
        result['checks']['all_retained_group_scores_export_exactly']=True
        # Existing attempted-case filter remains operational, with excluded enrollment absent.
        page.locator('#split').select_option('evaluation');page.locator('#stratum').select_option('unseen')
        n=sum(c['split']=='evaluation'and c['stratum']=='unseen'for c in json.loads(frozen)['cases'])
        assert page.locator('#cases tbody tr:visible').count()==n
        result['checks']['unsupported_case_rows_remain_visible']=n
        page.close()
        page=load(pair,'pair');q=json.loads(page.locator('#pair-routing-data').text_content())
        immutable=page.locator('#intrinsic-data').text_content();intrinsic=json.loads(immutable)
        assert len(q['sources'])==2 and q['sources'][0]['quantile']!=q['sources'][1]['quantile']
        assert [s['quantile']for s in q['sources']]==intrinsic['source_quantiles']
        assert page.locator('#pair-routing select').count()==0
        assert not page.locator('#intrinsic-display option[value=reference]').is_disabled()
        page.locator('#intrinsic-display').select_option('reference')
        for viewport in({'width':1440,'height':1050},{'width':390,'height':844}):
            page.set_viewport_size(viewport);label='mobile'if viewport['width']<800 else'desktop'
            assert page.locator('#pair-routing .route-card').count()==2
            page.locator('#pair-routing').evaluate('(e)=>e.scrollIntoView({block:"start"})')
            page.wait_for_timeout(50)
            assert page.evaluate('document.documentElement.scrollWidth')<=viewport['width']
            filename=f'{label}_source_radii.png';page.screenshot(path=str(out/filename))
            result['viewports'].append({'report':'pair','viewport':viewport,'screenshot':filename,'query_radii':[s['quantile']for s in q['sources']]})
        assert immutable==page.locator('#intrinsic-data').text_content()
        result['checks'].update(no_source_route_selector=True,reference_geometry_enabled_with_two_radii=True,
            display_preserves_measurement_payload=True,local_evidence_paths_exist=all((pair.parent/n).is_file()for n in('arrays.npz','localization_calibration.json','intrinsic.json')))
        browser.close()
    result['errors']=errors;result['remote_requests']=requests
    assert not errors,errors
    assert not requests,requests
    result['finished_utc']=datetime.now(timezone.utc).isoformat();result['passed']=True
    (out/'browser_validation.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('pair');p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(run(Path(a.study).resolve(),Path(a.pair).resolve(),Path(a.out)),indent=2))
