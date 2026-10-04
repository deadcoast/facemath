import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
import zipfile
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent
out = root/'portable'; out.mkdir(exist_ok=False)
with zipfile.ZipFile(root/'browser-01/comparison.zip') as archive:
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        if path.parent == PurePosixPath('result'):
            (out/path.name).write_bytes(archive.read(item))
manifest = json.loads((out/'manifest.json').read_text())
for name, record in manifest['files'].items():
    data = (out/name).read_bytes()
    assert sha256(data).hexdigest() == record['sha256']
summary = json.loads((out/'summary.json').read_text())
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1050})
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto((out/'report.html').as_uri())
    count = 0
    for row in summary['regional_inspection']['rows']:
        detail = page.locator(f'details[data-region="{row["region"]}"]')
        detail.locator(':scope > summary').focus(); page.keyboard.press('Enter')
        assert detail.get_attribute('open') is not None
        for img in detail.locator('[data-regional-map] img').all():
            assert img.is_visible() and img.evaluate('e => e.complete && e.naturalWidth > 0')
            count += 1
        if row['region'] != 'jaw':
            detail.locator(':scope > summary').click()
    page.locator('details[data-region="jaw"] > summary').evaluate('e => e.scrollIntoView({block:"start"})')
    page.screenshot(path=str(root/'portable-regional.png'))
    assert not errors, errors
    record = {'passed': True, 'download_manifest_files_checked': len(manifest['files']),
              'keyboard_region_disclosures': 7, 'map_images_loaded_from_download': count,
              'page_errors': errors, 'browser': browser.version,
              'source': 'Downloaded native comparison.zip, opened local report.html with relative assets'}
    (root/'portable.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    browser.close()
