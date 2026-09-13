"""Exercise the emitted HTML and retain screenshots and actual browser results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile

from playwright.sync_api import sync_playwright


def run(report_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    result = {"format": "facemath.browser_qa.v5", "started_at": datetime.now(timezone.utc).isoformat(),
              "report": str(report_path), "engine": "Chromium / Playwright", "checks": {}, "viewports": []}
    with sync_playwright() as driver:
        browser = driver.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                        args=["--no-sandbox"])
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        errors, network = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: network.append(request.url) if request.url.startswith(("http:","https:")) else None)
        try:
            page.goto(report_path.as_uri(), wait_until="load", timeout=15000)
            result["loading"] = "file URL with actual bundle-relative assets"
        except Exception as exc:
            page.close()
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: network.append(request.url) if request.url.startswith(("http:","https:")) else None)
            page.set_content(report_path.read_text(), wait_until="load")
            result["loading"] = "in-memory HTML; local navigation unavailable"
            result["navigation_error"] = str(exc)
        data = page.locator("#data").text_content()
        data = json.loads(data)
        rows = data["cases"]
        count = lambda: page.locator("#cases tbody tr:visible").count()
        assert count() == len(rows)
        result["checks"]["all_cases_rendered"] = count()
        for viewport in [{"width":1440,"height":1100},{"width":390,"height":844}]:
            page.set_viewport_size(viewport)
            page.locator("#split").select_option("all")
            page.locator("#stratum").select_option("all")
            page.locator("#outcome").select_option("all")
            page.evaluate("window.scrollTo(0,0)")
            body_width = page.evaluate("document.documentElement.scrollWidth")
            assert body_width <= viewport["width"], (body_width,viewport)
            name="desktop" if viewport['width']>600 else "mobile"
            page.screenshot(path=str(output/f"{name}.png"))
            result["viewports"].append({**viewport,"document_width":body_width,"body_overflow":False})
        page.locator("#split").select_option("evaluation")
        page.locator("#stratum").select_option("unseen_shift")
        expected=[r for r in rows if r['split']=='evaluation' and r['stratum']=='unseen_shift']
        assert count()==len(expected)
        result['checks']['heldout_stratum_filter_rows']=count()
        with page.expect_download() as dl:
            page.locator('#export').click()
        download=dl.value
        csv_path=output/'filtered-cases.csv';download.save_as(csv_path)
        exported=list(csv.DictReader(io.StringIO(csv_path.read_text())))
        assert [r['case_id'] for r in exported]==[r['case_id'] for r in expected]
        result['checks']['export_exact_selected_cases']=True
        page.locator("#stratum").select_option("all")
        page.locator("#outcome").select_option("failed")
        expected=[r for r in rows if r['split']=='evaluation' and r['attempt_outcome']!='observed']
        assert count()==len(expected)
        result['checks']['failed_outcome_filter_rows']=count()
        page.locator('#cases').scroll_into_view_if_needed()
        page.screenshot(path=str(output/'mobile_filtered.png'))
        # Direct links refer to disk artifacts, not manufactured UI-only detail.
        result['checks']['local_evidence_links_exist']=all((report_path.parent/r[key]).is_file()
            for r in rows for key in ('image','receipt'))
        result['checks']['browser_navigation_to_evidence'] = result['loading'].startswith('file')
        # Ensure headers and code identifiers escape markup, and CSV text does
        # not gain spreadsheet formula semantics. Alter ONLY a separate QA page.
        from facemath.observation.study_report import render_study_report
        altered=json.loads(json.dumps(data))
        altered['cases'][0]['case_id']='=2+3'
        altered['domain']='<script>window.__bad=true</script>'
        qa=context.new_page();qa.set_content(render_study_report(altered),wait_until='load')
        assert qa.evaluate('window.__bad===undefined')
        with qa.expect_download() as dl:
            qa.locator('#export').click()
        path=output/'formula-escape-control.csv';dl.value.save_as(path)
        value=list(csv.DictReader(io.StringIO(path.read_text())))[0]['case_id']
        assert value=="'=2+3"
        result['checks']['markup_and_csv_formula_escaped']=True
        result['page_errors']=errors
        result['remote_requests']=network
        assert not errors and not network
        browser.close()
    result['finished_at']=datetime.now(timezone.utc).isoformat()
    result['passed']=True
    (output/'browser_validation.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('report');parser.add_argument('--out',required=True)
    args=parser.parse_args()
    print(json.dumps(run(Path(args.report).resolve(),Path(args.out).resolve()),indent=2))
