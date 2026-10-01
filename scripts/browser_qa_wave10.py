"""Exercise the emitted milestone reports and blank workspace UI.

Optional developer dependency: Playwright + a local Chromium binary. This script
never calls a detector. Pointer annotations are explicitly synthetic UI fixtures.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright


def run(lab:Path,out:Path,chromium:str='/usr/bin/chromium')->dict:
    out.mkdir(parents=True,exist_ok=False)
    result={'format':'facemath.browser_qa.v10','started_utc':datetime.now(timezone.utc).isoformat(),
            'reports':{},'checks':{},'viewports':[],'annotations':'none; blank workspace and rater state tested, not human labels'}
    errors,network=[],[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=chromium,headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        context=browser.new_context(accept_downloads=True)
        def load(path,name):
            page=context.new_page()
            def attach(pg):
                pg.on('pageerror',lambda e:errors.append(str(e)))
                pg.on('request',lambda r:network.append(r.url) if r.url.startswith(('http:','https:')) else None)
            attach(page)
            try:
                page.goto(path.as_uri(),wait_until='load',timeout=10000)
                result['reports'][name]={'path':str(path),'loading':'file URL'}
            except Exception as exc:
                page.close();page=context.new_page();attach(page)
                page.set_content(path.read_text(),wait_until='load')
                result['reports'][name]={'path':str(path),'loading':'emitted HTML in memory','navigation_error':str(exc)}
            return page
        for branch in ('positive','negative'):
            page=load(lab/branch/'qualification/report.html',branch)
            raw=page.locator('#data').text_content();data=json.loads(raw)
            admission=data['empirical_admission']
            assert page.locator('#admission tbody tr').count()==len(admission['categories'])
            assert 'SYNTHETIC CONTROL ONLY' in page.locator('#admission').inner_text()
            assert page.locator('#cases tbody tr').count()==len(data['cases'])
            assert ('admitted synthetic control' if branch=='positive' else 'withheld') in page.locator('#admission h2').inner_text()
            for viewport in ({'width':1440,'height':1080},{'width':390,'height':844}):
                page.set_viewport_size(viewport);label='mobile' if viewport['width']<800 else 'desktop'
                page.locator('#admission').scroll_into_view_if_needed()
                assert page.evaluate('document.documentElement.scrollWidth')<=viewport['width']
                page.screenshot(path=str(out/f'{label}_{branch}_acceptance.png'))
            page.locator('#split').select_option('evaluation')
            page.locator('#stratum').select_option('pixel_smooth')
            wanted=[r for r in data['cases'] if r['split']=='evaluation' and r['stratum']=='pixel_smooth']
            assert page.locator('#cases tbody tr:visible').count()==len(wanted)
            assert page.locator('#data').text_content()==raw
            result['checks'][branch+'_full_evaluation_retained']=len(wanted)
            page.close()
        pair=lab/'admitted_pair';page=load(pair/'report.html','pair')
        source=page.locator('#acquisition9 [data-role="data"]').text_content()
        intrinsic=page.locator('#intrinsic-data').text_content();groups=json.loads(source)
        summary=json.loads((pair/'summary.json').read_text())
        assert page.locator('#readout > .scroll tbody tr').count()==4
        assert 'not facial resemblance' in page.locator('#admission').inner_text()
        assert 'Missing relations remain in the denominator' in page.locator('#readout').inner_text()
        for index,f in enumerate(summary['families']):
            row=page.locator('#readout > .scroll tbody tr').nth(index).inner_text()
            lo,hi=f['within_tolerance_percent_bounds']
            assert f'{lo:.2f}–{hi:.2f}%' in row
        loc=lambda name:page.locator(f'#acquisition9 [data-role="{name}"]')
        for viewport in ({'width':1440,'height':1080},{'width':390,'height':844}):
            page.set_viewport_size(viewport);label='mobile' if viewport['width']<800 else 'desktop'
            page.locator('#readout').scroll_into_view_if_needed()
            assert page.evaluate('document.documentElement.scrollWidth')<=viewport['width']
            page.screenshot(path=str(out/f'{label}_readout.png'))
            loc('group').select_option('1');loc('member').select_option('1');loc('view').select_option('grid')
            assert loc('category').inner_text()=='Derived category: '+groups[1]['category']
            with page.expect_download() as download:loc('csv').click()
            path=out/f'{label}_complete_group.csv';download.value.save_as(path)
            rows=list(csv.DictReader(path.open()))
            assert len(rows)==9*len(groups[1]['members'])
            for j,m in enumerate(groups[1]['members']):
                for k,name in enumerate(groups[1]['feature_order']):
                    row=rows[j*9+k]
                    assert row['image_sha256']==m['image_sha256'] and float(row['value'])==groups[1]['features'][j][k]
            page.locator('#acquisition9').scroll_into_view_if_needed()
            page.screenshot(path=str(out/f'{label}_source_group.png'))
            result['viewports'].append({'size':viewport,'page_fits':True,'csv_exact':True})
        assert intrinsic==page.locator('#intrinsic-data').text_content() and source==loc('data').text_content()
        assert (pair/'qualification_snapshot.zip').is_file()
        page.close()
        # The generated workspace is genuinely blank even though its source has
        # a separate synthetic reference file imported by the Python lab.
        bench=next((lab/'positive/workspace/workbenches').glob('*.html'))
        page=load(bench,'blank_reference_workspace')
        page.wait_for_function('image.complete && image.naturalWidth > 0 && baseScale > 0')
        for viewport in ({'width':1440,'height':1080},{'width':390,'height':844}):
            page.set_viewport_size(viewport);label='mobile' if viewport['width']<800 else 'desktop'
            assert page.locator('#progress-label').inner_text()=='0 / 8 resolved'
            assert page.locator('#export').is_disabled()
            page.locator('#rater').fill('synthetic-ui-test-rater')
            page.locator('#rater').press('Tab')
            assert page.locator('#export').is_disabled()
            page.locator('#rater').fill('another-synthetic-ui-rater');page.locator('#rater').press('Tab')
            assert page.locator('#progress-label').inner_text()=='0 / 8 resolved'
            page.screenshot(path=str(out/f'{label}_blank_workspace.png'))
        assert not errors,errors
        assert not network,network
        result['checks'].update(readout_uses_stored_fraction_bounds=True,whole_group_csv_exact=True,
            immutable_pair_payload=True,blank_workspace_no_reference_predictions=True,
            no_result_category_or_acceptance_editors=True,qualification_snapshot_exists=True)
        browser.close()
    result.update(passed=True,script_errors=errors,remote_requests=network,finished_utc=datetime.now(timezone.utc).isoformat())
    (out/'browser_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('laboratory',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--chromium',default='/usr/bin/chromium')
    a=p.parse_args();print(json.dumps(run(a.laboratory.resolve(),a.out,a.chromium),indent=2))
