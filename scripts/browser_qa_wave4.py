"""Browser acceptance test; all automated annotations are SYNTHETIC fixtures.

Usage: python scripts/browser_qa_wave4.py PATH_TO_VALIDATION_LAB OUT_DIRECTORY
Requires optional Playwright and a local Chromium binary. No model downloads.
"""

import argparse
import json
import re
from base64 import b64encode
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from facemath.observation.reference import REFERENCE_NAMES, ReferenceAnnotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("laboratory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=args.chromium, headless=True, args=["--no-sandbox"]
        )
        for name, viewport in [
            ("desktop", dict(width=1440, height=1060)),
            ("mobile", dict(width=390, height=844)),
        ]:
            context = browser.new_context(
                viewport=viewport, accept_downloads=True, device_scale_factor=1
            )
            page = context.new_page()
            errors = []
            remote = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "request",
                lambda r: (
                    remote.append(r.url)
                    if r.url.startswith(("http://", "https://"))
                    else None
                ),
            )
            page.set_content(
                (args.laboratory / "reference_workbench.html").read_text(),
                wait_until="load",
            )
            page.wait_for_function(
                "image.complete && image.naturalWidth > 0 && baseScale > 0"
            )
            assert page.locator("#progress-label").inner_text() == "0 / 8 resolved"
            assert page.locator("#export").is_disabled()
            page.locator("#rater").fill("automated-synthetic-browser-qa")
            if name == "desktop":
                page.locator("#zoom-in").click()
            coords = [
                [91, 80],
                [110, 80],
                [145, 81],
                [162, 80],
                [126, 114],
                [112, 138],
                [145, 138],
                [128, 168],
            ]
            for i, (x, y) in enumerate(coords):
                page.get_by_role(
                    "button", name=f"Select {REFERENCE_NAMES[i]}", exact=True
                ).click()
                page.locator("#stage").scroll_into_view_if_needed()
                # Test source->view->actual pointer event->export, including zoom.
                pos = page.evaluate(
                    "([x,y]) => {const r=canvas.getBoundingClientRect();return [r.left+view.tx+x*view.scale,r.top+view.ty+y*view.scale]}",
                    [x, y],
                )
                page.mouse.click(*pos)
            assert not page.locator("#export").is_disabled()
            page.get_by_role("button", name="Select chin", exact=True).click()
            page.locator("#unobservable").click()
            assert page.evaluate("exportReference().visible[7]") is False
            page.locator("#undo").click()
            assert page.evaluate("exportReference().visible[7]") is True
            page.locator("#redo").click()
            assert page.evaluate("exportReference().visible[7]") is False
            page.locator("#undo").click()
            page.locator("#stage").focus()
            page.keyboard.press("ArrowRight")
            assert abs(page.evaluate("exportReference().xy[7][0]") - 128.25) < 0.1
            # Return chin to the test target before exporting.
            page.keyboard.press("ArrowLeft")
            with page.expect_download() as dl:
                page.locator("#export").click()
            target = args.output / f"{name}_synthetic_reference.json"
            dl.value.save_as(target)
            exported = json.loads(target.read_text())
            reference = ReferenceAnnotation.from_dict(exported)
            assert reference.kind == "synthetic_fixture"
            assert reference.prediction_exposure == "none_declared"
            err = float(np.max(np.abs(reference.xy - np.array(coords))))
            assert err <= 0.1, err
            page.evaluate("window.scrollTo(0,0)")
            page.screenshot(
                path=str(args.output / f"{name}_workbench.png"), full_page=True
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.locator("#rater").fill("different-declared-rater")
            page.locator("#rater").press("Tab")
            assert page.locator("#progress-label").inner_text() == "0 / 8 resolved"
            assert page.locator("#export").is_disabled()
            views = []
            for relative, label in [
                ("validation/report.html", "validation"),
                ("qualified_pair/report.html", "pair"),
            ]:
                path = args.laboratory / relative
                html = path.read_text()

                # Render exact report text, replacing only local raster URLs with
                # their byte-identical data URIs: browser navigation is restricted.
                def inline_image(match):
                    if match[1].startswith("data:"):
                        return match[0]
                    asset = path.parent / match[1]
                    if asset.is_file() and asset.suffix == ".png":
                        return f'src="data:image/png;base64,{b64encode(asset.read_bytes()).decode()}"'
                    return match[0]

                html = re.sub(r'src="([^"]+)"', inline_image, html)
                page.set_content(html, wait_until="load")
                page.wait_for_load_state("load")
                page.screenshot(
                    path=str(args.output / f"{name}_{label}.png"), full_page=True
                )
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth"
                )
                views.append(
                    dict(
                        view=label,
                        tables=page.locator("table").count(),
                        rows=page.locator("tr").count(),
                    )
                )
            assert not errors, errors
            assert not remote, remote
            reports.append(
                dict(
                    viewport=name,
                    export_roundtrip=True,
                    max_source_coordinate_error_px=err,
                    missing_undo_redo_checked=True,
                    keyboard_subpixel_nudge_checked=True,
                    rater_switch_clears_previous_labels=True,
                    console_errors=errors,
                    remote_requests=remote,
                    report_views=views,
                )
            )
            context.close()
        browser.close()
    output = {
        "format": "facemath.browser_qa.v4",
        "render_mode": "in-memory set_content; exact local PNG bytes inlined; external/file navigation blocked",
        "annotations": "automated synthetic fixture; not independent human labels",
        "results": reports,
    }
    (args.output / "browser_validation.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
