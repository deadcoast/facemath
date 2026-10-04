"""Real-browser checks of emitted pair reports; no inference or score editing."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def run(pair: Path, out: Path, executable: str, unavailable: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'pair': str(pair), 'checks': [],
              'scope': 'emitted native pair HTML; browser interactions do not validate anatomical accuracy'}
    errors, remote = [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=executable, headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('request', lambda r: remote.append(r.url) if r.url.startswith(('https:', 'http:')) else None)
            for branch in ('nominal', 'calibrated'):
                root = pair/branch
                summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))['structural_comparison']
                page.goto((root/'report.html').as_uri(), wait_until='load')
                rows = page.locator('#structure tbody tr')
                assert rows.count() == 14
                for i, row in enumerate(summary['rows']):
                    cells = rows.nth(i).locator('td')
                    assert cells.nth(0).inner_text() == format(row['observed_a'], '.5g')
                    assert cells.nth(1).inner_text() == format(row['observed_b'], '.5g')
                    assert cells.nth(2).inner_text().splitlines()[0] == format(row['observed_delta_b_minus_a'], '+.5g')
                    if branch == 'nominal':
                        assert 'Unavailable' in cells.nth(3).inner_text()
                    else:
                        lo, hi = row['reference_delta_b_minus_a']
                        assert f'[{lo:+.5g}, {hi:+.5g}]' in cells.nth(3).inner_text()
                record['checks'].append(branch+': fourteen rows equal exported measurements')
                edge_data = json.loads((root/'summary.json').read_text(encoding='utf-8')).get('edge_comparison')
                if edge_data:
                    edge_rows = page.locator('#edge-comparison tbody tr')
                    assert edge_rows.count() == 8
                    for i, row in enumerate(edge_data['rows']):
                        cells = edge_rows.nth(i).locator('td')
                        assert cells.nth(0).inner_text().splitlines()[0] == f"{row['supported']} / {row['total']}"
                        stats = row['eye_spans'] or {}
                        for column, key, fmt in ((1,'mean_signed_offset','+.5g'),(2,'rms_offset','.5g'),(3,'p90_absolute_offset','.5g')):
                            value = stats.get(key)
                            expected = 'Unavailable' if value is None else format(value,fmt)
                            assert cells.nth(column).inner_text() == expected
                    page.locator('#edge-comparison').scroll_into_view_if_needed()
                    page.screenshot(path=str(out/f'{branch}-edges-desktop.png'))
                    record['checks'].append(branch+': eight paired edge rows and unavailable values equal exported measurements')
                    edge_images = page.locator('.edge-gallery img')
                    assert edge_images.count() == 2
                    assert edge_images.evaluate_all('(images) => images.every(i => i.complete && i.naturalWidth > 0)')
                    page.locator('.edge-gallery').scroll_into_view_if_needed()
                    page.screenshot(path=str(out/f'{branch}-edge-locations.png'))
                    record['checks'].append(branch+': both anchored edge location views decode')
                evidence = json.loads((root/'evidence.json').read_text(encoding='utf-8'))
                regional = [row for row in evidence['measurement_profile'] if row['name'].startswith('regional.')]
                if regional:
                    assert len(regional) == 56
                    for measured in regional:
                        row = page.get_by_text(measured['name'], exact=True).locator('xpath=ancestor::tr')
                        expected = 'WITHHELD' if measured['value'] is None else format(measured['value'], '.6g')
                        assert row.locator('td').nth(1).inner_text() == expected
                    page.get_by_text(regional[0]['name'], exact=True).scroll_into_view_if_needed()
                    page.screenshot(path=str(out/f'{branch}-regional.png'))
                    record['checks'].append(branch+': all 56 regional component ledger values equal exported measurements')
                page.locator('#structure').scroll_into_view_if_needed()
                page.screenshot(path=str(out/f'{branch}-desktop.png'))
                # Open the actual disclosure, then operate the native slider by keyboard.
                page.get_by_text('Interactive opacity inspection', exact=True).click()
                slider = page.locator('#opacity')
                slider.focus(); slider.press('End')
                assert slider.input_value() == '100'
                assert page.locator('#blend-b').evaluate('(e) => getComputedStyle(e).opacity') == '1'
                slider.press('Home')
                assert page.locator('#blend-b').evaluate('(e) => getComputedStyle(e).opacity') == '0'
                record['checks'].append(branch+': keyboard slider changes only overlay opacity')
                page.locator('button[data-family="angle"]').click()
                assert page.locator('#intrinsic-rows tr').count() == 168
                search = page.locator('#intrinsic-search')
                search.fill('nose_tip')
                assert 0 < page.locator('#intrinsic-rows tr').count() < 168
                search.fill('no_such_landmark')
                assert page.locator('#intrinsic-rows tr').count() == 0
                search.fill('')
                assert page.locator('#intrinsic-rows tr').count() == 168
                record['checks'].append(branch+': sibling relation filter empty and recovery states')
                with page.expect_download() as pending:
                    page.locator('#intrinsic-csv').click()
                export = out/f'{branch}-relations.csv'
                pending.value.save_as(export)
                with export.open(encoding='utf-8-sig', newline='') as stream:
                    exported_rows = list(csv.DictReader(stream))
                assert len(exported_rows) == 168
                record['checks'].append(branch+': selected relation button downloads all 168 filtered rows')
                if branch == 'calibrated':
                    with page.expect_download() as pending:
                        page.locator('#joint-covariance-csv').click()
                    export = out/'source-matrix.csv'
                    pending.value.save_as(export)
                    with export.open(encoding='utf-8-sig', newline='') as stream:
                        matrix_rows = list(csv.DictReader(stream))
                    assert len(matrix_rows) == 256
                    record['checks'].append(branch+': source covariance button downloads 256 entries')
                page.set_viewport_size({'width': 390, 'height': 844})
                table = page.get_by_role('region', name='Structural comparison table')
                table.scroll_into_view_if_needed(); table.focus()
                assert table.evaluate('(e) => e.scrollWidth > e.clientWidth')
                table.press('End')
                table.press('ArrowRight')
                page.wait_for_function("document.querySelector('#structure .scroll').scrollLeft > 0")
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert table.evaluate('(e) => getComputedStyle(e).outlineStyle') != 'none'
                page.locator('#structure h2').scroll_into_view_if_needed()
                page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                page.screenshot(path=str(out/f'{branch}-narrow.png'))
                record['checks'].append(branch+': narrow table scrolls by keyboard without page overflow')
                if edge_data:
                    edge_table = page.get_by_role('region', name='Image-edge comparison table')
                    edge_table.scroll_into_view_if_needed(); edge_table.focus()
                    assert edge_table.evaluate('(e) => e.scrollWidth > e.clientWidth')
                    edge_table.press('ArrowRight')
                    page.wait_for_function("document.querySelector('#edge-comparison .scroll').scrollLeft > 0")
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    assert edge_table.evaluate('(e) => getComputedStyle(e).outlineStyle') != 'none'
                    page.screenshot(path=str(out/f'{branch}-edges-narrow.png'))
                    record['checks'].append(branch+': paired edge table supports narrow keyboard scrolling and visible focus')
                page.set_viewport_size({'width': 1440, 'height': 1000})
            if unavailable is not None:
                page.goto((unavailable/'report.html').as_uri(), wait_until='load')
                rows = page.locator('#edge-comparison tbody tr')
                assert rows.count() == 8
                for i in range(8):
                    cells = rows.nth(i).locator('td')
                    assert cells.nth(0).inner_text().startswith('0 / 32')
                    assert all(cells.nth(c).inner_text() == 'Unavailable' for c in (1,2,3))
                    assert cells.nth(4).inner_text() == '0.0–100.0%'
                page.locator('#edge-comparison').scroll_into_view_if_needed()
                page.screenshot(path=str(out/'unavailable-edges.png'))
                record['checks'].append('flat synthetic image control: all eight rows remain unavailable with full denominators')
            assert not errors, errors
            assert not remote, remote
            record['browser_version'] = browser.version
            browser.close()
        record['passed'] = True
    except Exception as exc:
        record.update(passed=False, error=repr(exc))
        raise
    finally:
        record.update(script_errors=errors, remote_requests=remote, finished_utc=datetime.now(timezone.utc).isoformat())
        (out/'browser.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--browser', required=True)
    parser.add_argument('--unavailable', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.pair.resolve(), args.out.resolve(), args.browser,
                         args.unavailable.resolve() if args.unavailable else None), indent=2))
