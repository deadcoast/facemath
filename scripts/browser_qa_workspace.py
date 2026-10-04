"""Exercise real file selection through the local native comparison application."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import struct
import zlib
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("image_a", type=Path)
    parser.add_argument("image_b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--browser", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.locator("#image-a").set_input_files({"name": "invalid.png", "mimeType": "image/png", "buffer": b"invalid image"})
        page.locator("#image-b").set_input_files(str(args.image_b.resolve()))
        page.get_by_role("button", name="Compare images", exact=True).click()
        page.wait_for_function("() => document.querySelector('#message').textContent.includes('PNG or JPEG')")
        assert page.locator("#start").is_enabled()
        page.locator("#image-a").set_input_files(str(args.image_a.resolve()))
        page.locator("#image-b").set_input_files(str(args.image_b.resolve()))
        page.screenshot(path=str(args.out / "selected.png"), full_page=True)
        page.get_by_role("button", name="Compare images", exact=True).click()
        page.wait_for_function("() => location.hash.startsWith('#run=')", timeout=30000)
        print("Native analysis submitted: " + page.url.split("#")[-1], flush=True)
        page.wait_for_function("() => !document.querySelector('#results').hidden || ['failed','cancelled','interrupted'].includes(document.querySelector('#message').dataset.state)", timeout=600000)
        assert page.locator("#results").is_visible(), page.locator("#message").inner_text()
        assert page.locator("#structure tbody tr").count() == 14
        assert page.locator("#edge-comparison tbody tr").count() == 8
        page.wait_for_function("() => [...document.querySelectorAll('#results img')].every(i => i.complete && i.naturalWidth > 0)")
        page.locator("#opacity").focus()
        page.keyboard.press("ArrowRight")
        assert page.locator("#blend-b").evaluate("e => Number(e.style.opacity)") == .51
        run_id = page.url.split("#run=")[1]
        summary = page.request.get(args.url + f"runs/{run_id}/result/summary.json").json()
        for row, actual in zip(summary["structural_comparison"]["rows"], page.locator("#structure tbody tr").all()):
            assert row["label"] in actual.inner_text()
            for index, field in enumerate(("observed_a", "observed_b", "observed_delta_b_minus_a")):
                value = row[field]
                expected = "Unavailable" if value is None else format(value, "+.5g" if index == 2 else ".5g")
                assert actual.locator("td").nth(index).inner_text().splitlines()[0] == expected
        bilateral = summary["bilateral_comparison"]
        assert page.locator("#bilateral tbody tr").count() == 5
        for row, actual in zip(bilateral["rows"], page.locator("#bilateral tbody tr").all()):
            assert row["label"] in actual.inner_text()
            for index, field in enumerate(("observed_a", "observed_b", "observed_delta_b_minus_a")):
                value = row[field]
                expected = "Unavailable" if value is None else '(' + ', '.join(format(v, '+.5g') for v in value) + ')'
                assert actual.locator("td").nth(index).inner_text().splitlines()[0] == expected
            disclosure = page.locator(f'details[data-bilateral="{row["key"]}"]')
            disclosure.locator('summary').focus()
            page.keyboard.press('Enter')
            assert disclosure.get_attribute('open') is not None
            for s, svg in enumerate(disclosure.locator('svg').all()):
                segment = row['display_segments'][s]
                assert float(svg.locator('circle').get_attribute('cx')) == segment[0][0]
                assert float(svg.locator('circle').get_attribute('cy')) == segment[0][1]
                href = svg.locator('image').get_attribute('href')
                response = page.request.get(args.url + href)
                assert response.ok and response.headers['content-type'] == 'image/png'
        page.locator('#bilateral').scroll_into_view_if_needed()
        page.screenshot(path=str(args.out / 'bilateral.png'), full_page=True)
        regional = summary['regional_inspection']
        assert page.locator('#regional [data-region-row]').count() == 7
        regional_points = 0
        regional_triangles = 0
        for row in regional['rows']:
            disclosure = page.locator(f'details[data-region="{row["region"]}"]')
            disclosure.locator(':scope > summary').focus()
            page.keyboard.press('Enter')
            assert disclosure.get_attribute('open') is not None
            for point in row['landmarks']:
                if point['supported']:
                    marker = disclosure.locator(f'g[data-landmark="{point["index"]}"]')
                    assert float(marker.locator('circle').get_attribute('cx')) == point['a'][0]
                    assert float(marker.locator('circle').get_attribute('cy')) == point['a'][1]
                    regional_points += 1
            for m in row['maps'] if row['maps_available'] else []:
                img = disclosure.locator(f'[data-regional-map="{m["key"]}"] img')
                assert img.is_visible() and img.evaluate('e => e.complete && e.naturalWidth > 0')
                assert m['image'] in img.get_attribute('src')
            for nested in disclosure.locator(':scope > details').all():
                nested.locator(':scope > summary').focus()
                page.keyboard.press('Enter')
            for point in row['landmarks']:
                cell = disclosure.locator(f'[data-landmark-row="{point["index"]}"] td').nth(2)
                expected = 'Unavailable' if point['delta'] is None else '(' + ', '.join(format(v, '+.5g') for v in point['delta']) + ')'
                assert cell.inner_text() == expected
            for t in row['triangles']:
                cells = disclosure.locator(f'[data-triangle="{t["index"]}"] td')
                assert int(cells.nth(1).inner_text()) == t['pixels']
                assert cells.nth(2).inner_text() == format(t['rotation_deg'], '+.5g')
                assert cells.nth(3).inner_text() == format(t['area_ratio'], '.5g')
                regional_triangles += 1
            # Keep the overview compact after exercising each region.
            if row['region'] != 'jaw':
                disclosure.locator(':scope > summary').click()
        page.locator('details[data-region="jaw"]').scroll_into_view_if_needed()
        page.screenshot(path=str(args.out / 'regional.png'))
        with page.expect_download() as downloaded:
            page.locator("#download").click()
        downloaded.value.save_as(str(args.out / "comparison.zip"))
        page.screenshot(path=str(args.out / "results.png"), full_page=True)
        page.reload()
        page.locator("#results").wait_for(state="visible")
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator('details[data-bilateral="mouth"] summary').click()
        assert page.locator('details[data-bilateral="mouth"] svg').count() == 2
        page.locator('details[data-bilateral="mouth"]').scroll_into_view_if_needed()
        page.screenshot(path=str(args.out / 'bilateral-narrow.png'))
        page.locator('details[data-region="jaw"] > summary').click()
        assert page.locator('details[data-region="jaw"] [data-regional-map]').count() == 3
        page.locator('details[data-region="jaw"]').scroll_into_view_if_needed()
        page.screenshot(path=str(args.out / 'regional-narrow.png'))
        regional_table = page.get_by_role('region', name='Regional comparison table', exact=True)
        regional_table.focus()
        page.keyboard.press('ArrowRight')
        page.wait_for_function("() => document.querySelector('#regional .scroll').scrollLeft > 0")
        table = page.get_by_role("region", name="Image-edge comparison table", exact=True)
        table.scroll_into_view_if_needed()
        table.focus()
        page.keyboard.press("ArrowRight")
        page.wait_for_function("() => document.querySelector('#edge-comparison .scroll').scrollLeft > 0")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.out / "narrow.png"))
        # A real no-face observer failure stays visible and does not replace success.
        def chunk(kind, data):
            return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
        flat = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 256, 256, 8, 2, 0, 0, 0))
        flat += chunk(b'IDAT', zlib.compress((b'\0' + b'\x80' * 768) * 256)) + chunk(b'IEND', b'')
        page.locator("#image-a").set_input_files({"name": "no-face.png", "mimeType": "image/png", "buffer": flat})
        page.locator("#image-b").set_input_files(str(args.image_b.resolve()))
        page.locator("#start").click()
        page.wait_for_function("() => document.querySelector('#message').dataset.state === 'failed'", timeout=90000)
        assert page.get_by_role("link", name="Open execution log").is_visible()
        failure_id = page.url.split("#run=")[1]
        assert failure_id != run_id
        page.screenshot(path=str(args.out / "no-face.png"), full_page=True)
        page.goto(args.url + '#run=' + run_id)
        page.reload()
        page.locator("#results").wait_for(state="visible")
        page.locator("#image-a").set_input_files(str(args.image_a.resolve()))
        page.locator("#image-b").set_input_files(str(args.image_b.resolve()))
        page.locator("#start").click()
        page.locator("#cancel").wait_for(state="visible")
        page.locator("#cancel").click()
        page.wait_for_function("() => document.querySelector('#message').dataset.state === 'cancelled'")
        cancelled_id = page.url.split("#run=")[1]
        assert not errors, errors
        (args.out / "browser.json").write_text(json.dumps({"passed": True, "run_id": run_id,
            "browser": browser.version, "source": "real browser uploads into native analyze", "structure_rows": 14,
            "edge_rows": 8, "numeric_cells_compared": 42, "reload": True, "download": True,
            "bilateral_rows": 5, "bilateral_numeric_components_compared": 30,
            "bilateral_keyboard_disclosures": 5, "bilateral_marker_pairs_compared": 10,
            "regional_keyboard_disclosures": 7, "regional_maps": 21,
            "regional_landmark_coordinates_compared": regional_points,
            "regional_triangle_rows_compared": regional_triangles,
            "keyboard_blend": True, "narrow_keyboard_scroll": True, "page_errors": errors,
            "invalid_upload": True, "no_face_run": failure_id, "cancelled_run": cancelled_id,
            "success_reopened_after_failure": True}, indent=2), encoding="utf-8")
        print(f"Complete: native results, {regional_points} regional markers, {regional_triangles} triangle rows, download, reload and keyboard access", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
