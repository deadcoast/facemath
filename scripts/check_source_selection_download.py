"""Open an actual downloaded comparison and inspect its source selection offline."""
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import zipfile
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('download', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--browser', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    extracted = args.out/'download'; extracted.mkdir()
    with zipfile.ZipFile(args.download) as archive:
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            assert not name.is_absolute() and '..' not in name.parts
            target = extracted.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
    manifest = json.loads((extracted/'result/manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert sha256((extracted/'result'/name).read_bytes()).hexdigest() == record['sha256']
    request = json.loads((extracted/'request.json').read_text())
    for side, record in zip(('a', 'b'), request['inputs']):
        assert sha256((extracted/f'input_{side}.image').read_bytes()).hexdigest() == record['sha256']
        assert sha256((extracted/record['analysis']['file']).read_bytes()).hexdigest() == record['analysis']['sha256']
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1050})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((extracted/'selection.html').resolve().as_uri())
        assert page.locator('section').count() == 2
        for img, record in zip(page.locator('figure > img').all(), request['inputs']):
            assert img.evaluate('e => e.complete && e.naturalWidth > 0')
            assert img.evaluate('e => [e.naturalWidth,e.naturalHeight]') == [record['analysis']['width'], record['analysis']['height']]
        for img in page.locator('svg image').all():
            assert page.evaluate('async src => { const i = new Image(); i.src = src; await i.decode(); return i.naturalWidth > 0; }', img.get_attribute('href'))
        page.screenshot(path=str(args.out/'selection.png'), full_page=True)
        page.get_by_role('link', name='Open complete comparison', exact=True).click()
        summary = json.loads((extracted/'result/summary.json').read_text())
        count = 0
        for row in summary['regional_inspection']['rows']:
            detail = page.locator(f'details[data-region="{row["region"]}"]')
            detail.locator(':scope > summary').focus(); page.keyboard.press('Enter')
            assert detail.get_attribute('open') is not None
            for img in detail.locator('[data-regional-map] img').all():
                assert img.is_visible() and img.evaluate('e => e.complete && e.naturalWidth > 0')
                count += 1
            detail.locator(':scope > summary').click()
        assert not errors, errors
        record = {'passed': True, 'browser': browser.version, 'download_sha256': sha256(args.download.read_bytes()).hexdigest(),
            'manifest_files_checked': len(manifest['files']), 'input_hashes_checked': 4,
            'selection_images_loaded': 4, 'comparison_link_followed': True,
            'keyboard_region_disclosures': 7, 'map_images_loaded': count, 'page_errors': errors}
        (args.out/'portable.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2)); browser.close()


if __name__ == '__main__':
    main()
