"""Interact with the actual emitted measured-source and qualification reports.

The fallback renders the same HTML in memory if local file navigation is blocked;
it does not bypass browser policy or claim native file navigation succeeded.
"""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright


def run(study: Path, pair: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    result = {'format': 'facemath.browser_qa.v9', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'reports': {}, 'viewports': [], 'checks': {}}
    errors, remote = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                                   args=['--no-sandbox', '--disable-dev-shm-usage'])
        context = browser.new_context(accept_downloads=True)
        def load(source, name):
            def attach(page):
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.on('request', lambda r: remote.append(r.url) if r.url.startswith(('http:', 'https:')) else None)
            page=context.new_page(); attach(page)
            try:
                page.goto(source.as_uri(), wait_until='load', timeout=10000)
                result['reports'][name]={'path':str(source),'loading':'file URL'}
            except Exception as e:
                page.close(); page=context.new_page(); attach(page)
                page.set_content(source.read_text(),wait_until='load')
                result['reports'][name]={'path':str(source),'loading':'emitted HTML in memory',
                                          'file_navigation_error':str(e)}
            return page
        page=load(study,'qualification')
        frozen=page.locator('#data').text_content(); report=json.loads(frozen)
        routing=page.locator('#routing-data').text_content(); categories=json.loads(routing)['categories']
        assert len(report['cases'])==page.locator('#cases tbody tr').count()
        assert len(categories)==page.locator('#routing-summary tbody tr').count()
        assert 'Source-measured acquisition assignment' in page.inner_text('body')
        for viewport in ({'width':1440,'height':1080},{'width':390,'height':844}):
            page.set_viewport_size(viewport);label='mobile'if viewport['width']<800 else'desktop'
            for category in categories:
                name=category['stratum'];page.locator('#routing-category').select_option(name)
                assert page.locator('#routing-scores tbody tr').count()==len(category['group_scores'])
                assert category['status'].replace('_',' ') in page.locator('#routing-detail').inner_text()
                with page.expect_download() as download:page.locator('#routing-export').click()
                path=out/f'{label}-scores-{name}.csv';download.value.save_as(path)
                rows=list(csv.DictReader(path.open()));lookup=dict(category['group_scores'])
                assert len(rows)==len(lookup)
                for row in rows:
                    score=lookup[row['group_id']]
                    assert (row['score']=='' and row['score_available']=='false') if score is None else float(row['score'])==score
                    assert row['deployed_q']=='' if category['quantile'] is None else float(row['deployed_q'])==category['quantile']
            page.locator('#routing-category').select_option('pixel_smooth')
            page.locator('#acquisition-routing').scroll_into_view_if_needed()
            assert page.evaluate('document.documentElement.scrollWidth')<=viewport['width']
            page.screenshot(path=str(out/f'{label}_category_support.png'))
        page.locator('#split').select_option('evaluation');page.locator('#stratum').select_option('__unmatched__')
        rejected=sum(c['split']=='evaluation'and c['stratum']=='__unmatched__'for c in report['cases'])
        assert page.locator('#cases tbody tr:visible').count()==rejected and rejected>0
        assert frozen==page.locator('#data').text_content() and routing==page.locator('#routing-data').text_content()
        result['checks'].update(attempted_case_rows=len(report['cases']),rejected_evaluation_rows_retained=rejected,
            category_score_exports_exact=True,qualification_payload_unchanged=True)
        page.close()
        page=load(pair,'pair')
        scope='#acquisition9 ';loc=lambda n:page.locator(scope+f'[data-role="{n}"]')
        raw=loc('data').text_content(); groups=json.loads(raw)
        intrinsic=page.locator('#intrinsic-data').text_content()
        bound=page.locator('#pair-routing-data').text_content();routes=json.loads(bound)['sources']
        assert len(groups)==2 and [g['category']for g in groups]==[s['stratum']for s in routes]
        assert routes[0]['quantile']!=routes[1]['quantile']
        assert page.locator('#pair-routing select').count()==0
        for viewport in ({'width':1440,'height':1080},{'width':390,'height':844}):
            page.set_viewport_size(viewport);label='mobile'if viewport['width']<800 else'desktop'
            for i,g in enumerate(groups):
                loc('group').select_option(str(i))
                assert loc('member').locator('option').count()==len(g['members'])
                for j,member in enumerate(g['members']):
                    loc('member').select_option(str(j))
                    assert loc('features').locator('tr').count()==9
                    assert loc('predicates').locator('tr').count()==len(g['predicate_truth'])
                    assert loc('category').inner_text()=='Derived category: '+g['category']
                    for view in ('source','grid'):
                        loc('view').select_option(view)
                        shape=g['analyzed_patches'][j]['shape'] if view=='grid'else member['image_shape']
                        assert loc('preview').get_attribute('viewBox')==f'0 0 {shape[1]} {shape[0]}'
                        assert loc('image').get_attribute('href').startswith('data:image/png;base64,')
                        assert loc('category').inner_text()=='Derived category: '+g['category']
                with page.expect_download()as download:loc('csv').click()
                path=out/f'{label}-group-{i}.csv';download.value.save_as(path)
                rows=list(csv.DictReader(path.open()));assert len(rows)==len(g['members'])*9
                for j,member in enumerate(g['members']):
                    for k,name in enumerate(g['feature_order']):
                        row=rows[j*9+k]
                        assert row['image_sha256']==member['image_sha256'] and row['metric']==name
                        assert row['unit']==g['feature_units'][k]
                        assert float(row['value'])==g['features'][j][k]
                        assert float(row['group_min'])==g['aggregate'][0][k]
                        assert float(row['group_max'])==g['aggregate'][1][k]
            loc('group').select_option('1');loc('member').select_option('0');loc('view').select_option('source')
            page.locator('#acquisition9').evaluate('(e)=>e.scrollIntoView({block:"start"})')
            assert page.evaluate('document.documentElement.scrollWidth')<=viewport['width']
            page.screenshot(path=str(out/f'{label}_measured_source.png'))
            loc('view').select_option('grid')
            page.screenshot(path=str(out/f'{label}_analysis_grid.png'))
            result['viewports'].append({'viewport':viewport,'source_counts':[len(g['members'])for g in groups],
                'source_screenshot':f'{label}_measured_source.png','grid_screenshot':f'{label}_analysis_grid.png',
                'page_fits_viewport':True})
        assert raw==loc('data').text_content() and intrinsic==page.locator('#intrinsic-data').text_content() and bound==page.locator('#pair-routing-data').text_content()
        result['checks'].update(all_source_members_displayed=True,whole_group_category_unchanged_by_selected_member=True,
            all_original_and_analysis_dimensions_exact=True,all_group_measurements_export_exactly=True,
            all_routing_predicates_retained=True,no_category_aperture_threshold_editors=True,pair_payload_unchanged=True,
            source_sidecar_links_exist=all((pair.parent/(f'acquisition_{x}_'+n)).is_file()for x in('a','b')for n in('policy.json','measurements.json','arrays.npz','source_0000.png','report.html')))
        browser.close()
    assert not errors,errors
    assert not remote,remote
    result.update(errors=errors,remote_requests=remote,passed=True,finished_utc=datetime.now(timezone.utc).isoformat())
    (out/'browser_validation.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('pair');p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(run(Path(a.study).resolve(),Path(a.pair).resolve(),Path(a.out)),indent=2))
