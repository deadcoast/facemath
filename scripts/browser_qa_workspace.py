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
        with page.expect_download() as downloaded:
            page.locator("#download").click()
        downloaded.value.save_as(str(args.out / "comparison.zip"))
        page.screenshot(path=str(args.out / "results.png"), full_page=True)
        page.reload()
        page.locator("#results").wait_for(state="visible")
        page.set_viewport_size({"width": 390, "height": 844})
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
            "keyboard_blend": True, "narrow_keyboard_scroll": True, "page_errors": errors,
            "invalid_upload": True, "no_face_run": failure_id, "cancelled_run": cancelled_id,
            "success_reopened_after_failure": True}, indent=2), encoding="utf-8")
        print("Complete: real native results, 42 numeric cells, download, reload, keyboard blend and narrow scrolling", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
