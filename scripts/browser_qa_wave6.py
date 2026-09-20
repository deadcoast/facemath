"""Execute the emitted intrinsic workbench; no reconstructed screenshots."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def run(report_path: Path, destination: Path) -> dict:
    destination.mkdir(parents=True,exist_ok=False)
    report={'format':'facemath.browser_qa.v6','report':str(report_path),'started_at':datetime.now(timezone.utc).isoformat(),
            'checks':{},'viewports':[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        context=browser.new_context(accept_downloads=True)
        page=context.new_page();errors=[];network=[]
        def attach(page):
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('request',lambda r:network.append(r.url) if r.url.startswith(('https:','http:')) else None)
        attach(page)
        try:
            page.goto(report_path.as_uri(),wait_until='load',timeout=10000)
            report['loading']='file URL with source-relative artifacts'
        except Exception as exc:
            page.close();page=context.new_page();attach(page)
            page.set_content(report_path.read_text(),wait_until='load')
            report['loading']='in-memory emitted HTML; local navigation unavailable'
            report['navigation_error']=str(exc)
        data=json.loads(page.locator('#intrinsic-data').text_content())
        assert page.locator('#intrinsic-rows tr').count()==28
        for family,count in [('distance',28),('angle',168),('signed_double_area',56),('asymmetry',4)]:
            page.locator(f'button[data-family="{family}"]').click()
            assert page.locator('#intrinsic-rows tr').count()==count
            report['checks'][family+'_rows']=count
        page.locator('button[data-family="angle"]').click()
        page.locator('#intrinsic-search').fill('nose_tip')
        expected=[r for r in data['rows'] if r['family']=='angle' and 'nose_tip' in r['point_names']]
        assert page.locator('#intrinsic-rows tr').count()==len(expected)
        report['checks']['anatomical_name_filter_rows']=len(expected)
        selected=expected[-1]
        page.locator(f'#intrinsic-rows tr[data-index="{selected["index"]}"] button').click()
        text=page.locator('#intrinsic-selected').inner_text()
        assert selected['array_columns']['observed'] in text
        assert page.locator('#intrinsic-a .relation-line').count()==1
        report['checks']['selection_links_same_raw_array_row']=True
        with page.expect_download() as saved:
            page.locator('#intrinsic-csv').click()
        filename=destination/'filtered-relations.csv';saved.value.save_as(filename)
        rows=list(csv.DictReader(io.StringIO(filename.read_text())))
        assert [r['key'] for r in rows]==[r['key'] for r in expected]
        for got,want in zip(rows,expected):
            for key in ('observed_a','observed_b','reference_gap_lower','reference_gap_upper'):
                assert (got[key]=='' if want[key] is None else float(got[key])==want[key]), (key,got[key],want[key])
        report['checks']['csv_matches_exact_selected_frozen_values']=True
        page.locator('#intrinsic-search').fill('')
        page.locator('#intrinsic-status').select_option('outside_tolerance_under_envelope')
        expected=[r for r in data['rows'] if r['family']=='angle' and r['status']=='outside_tolerance_under_envelope']
        assert page.locator('#intrinsic-rows tr').count()==len(expected)
        report['checks']['status_filter_rows']=len(expected)
        page.locator('#intrinsic-status').select_option('all')
        page.locator('#intrinsic-display').select_option('reference')
        assert page.locator('#intrinsic-a .uncertainty').count()==8
        assert page.locator('#intrinsic-b .uncertainty').count()==8
        report['checks']['full_covariance_ellipse_display']=True
        box_before=page.locator('#intrinsic-a').get_attribute('viewBox')
        page.locator('#intrinsic-zoom').fill('3')
        page.locator('#intrinsic-zoom').dispatch_event('input')
        box_after=page.locator('#intrinsic-a').get_attribute('viewBox')
        assert float(box_after.split()[2])==float(box_before.split()[2])/3
        report['checks']['display_zoom_not_measurement_rescale']=True
        page.locator('#intrinsic-zoom').fill('1')
        page.locator('#intrinsic-zoom').dispatch_event('input')
        # Select a visibly altered mouth relation, not a UI-computed value.
        page.locator('button[data-family="distance"]').click()
        row=next(r for r in data['rows'] if r['key']=='distance.5.6')
        page.locator(f'#intrinsic-rows tr[data-index="{row["index"]}"] button').click()
        page.locator('#intrinsic-search').fill('mouth')
        report['checks']['local_evidence_links_exist']=all((report_path.parent/name).is_file()
            for name in ('intrinsic.json','intrinsic.csv','arrays.npz'))
        for size in [{'width':1440,'height':1150},{'width':390,'height':844}]:
            page.set_viewport_size(size)
            page.locator('#intrinsic').evaluate('(el)=>el.scrollIntoView({block:"start"})')
            page.wait_for_timeout(80)
            width=page.evaluate('document.documentElement.scrollWidth')
            assert width<=size['width'],(width,size)
            name='desktop' if size['width']>600 else 'mobile'
            page.screenshot(path=str(destination/f'{name}.png'))
            report['viewports'].append({**size,'document_width':width,'body_overflow':False})
        page.locator('#intrinsic-selected').scroll_into_view_if_needed()
        page.screenshot(path=str(destination/'mobile_detail.png'))
        # UI changes cannot mutate the embedded numerical evidence.
        assert json.loads(page.locator('#intrinsic-data').text_content())==data
        report['checks']['controls_do_not_change_measurements']=True
        assert not errors and not network,(errors,network)
        report['page_errors']=errors;report['remote_requests']=network
        browser.close()
    report['passed']=True;report['finished_at']=datetime.now(timezone.utc).isoformat()
    (destination/'browser_validation.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report');parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(run(Path(args.report).resolve(),Path(args.out).resolve()),indent=2))
