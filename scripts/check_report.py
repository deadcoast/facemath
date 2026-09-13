"""Browser smoke test for a generated self-contained evidence report.

Requires Playwright and a Chromium executable supplied explicitly. This is an
optional release-development tool, not a runtime dependency of FACEMATH.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('report')
    parser.add_argument('--chromium', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    errors = []
    requests = []
    records = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium,
                                             headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        page.on('request', lambda request: requests.append(request.url))
        # The managed development browser blocks file:// navigation. Render the
        # exact self-contained HTML bytes without a local or external server.
        page.set_content(Path(args.report).read_text(encoding='utf-8'), wait_until='load')
        for width, height, label in ((1440, 1100, 'desktop'), (390, 844, 'mobile')):
            page.set_viewport_size({'width': width, 'height': height})
            page.screenshot(path=str(out / f'{label}.png'), full_page=False)
            records.append({
                'viewport': label,
                'body_fits_viewport': page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),
                'images_loaded': page.locator('img').evaluate_all('(imgs) => imgs.every(i => i.complete && i.naturalWidth > 0)'),
                'measurement_rows': page.locator('table').nth(0).locator('tbody tr').count(),
            })
        page.locator('details summary').click()
        slider = page.locator('#opacity')
        slider.focus()
        slider.press('End')
        for _ in range(20):
            slider.press('ArrowLeft')
        opacity = page.locator('#blend-b').evaluate('(e) => getComputedStyle(e).opacity')
        browser.close()
    result = {'rendering_method': 'exact_HTML_bytes_via_set_content_due_to_managed_file_URL_policy', 'checks': records, 'script_errors': errors,
              'remote_requests': [u for u in requests if u.startswith(('http:', 'https:'))],
              'slider_80_percent_opacity': opacity}
    (out / 'browser-check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    assert all(r['body_fits_viewport'] and r['images_loaded'] for r in records)
    assert not errors and not result['remote_requests'] and opacity == '0.8'


if __name__ == '__main__':
    main()
