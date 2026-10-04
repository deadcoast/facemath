"""Use pointer and numeric source selection through actual native analysis."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('url'); parser.add_argument('fixtures',type=Path)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--browser',required=True); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False); errors=[]; matrix=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.browser,headless=True)
        for width,dpr in ((1440,1),(390,2)):
            context=browser.new_context(viewport={'width':width,'height':1000},device_scale_factor=dpr)
            page=context.new_page(); page.on('pageerror',lambda error:errors.append(str(error))); page.goto(args.url)
            page.locator('#image-a').set_input_files(str((args.fixtures/'indexed.png').resolve()))
            page.locator('#crop-editor-a').wait_for(state='visible')
            canvas=page.locator('#crop-canvas-a'); canvas.scroll_into_view_if_needed(); box=canvas.bounding_box()
            page.mouse.move(box['x']+7.25*box['width']/60,box['y']+11.25*box['height']/40); page.mouse.down()
            page.mouse.move(box['x']+38.75*box['width']/60,box['y']+34.75*box['height']/40,steps=5); page.mouse.up()
            values={k:int(page.locator('#crop-'+k+'-a').input_value()) for k in ('left','top','width','height')}
            assert values=={'left':7,'top':11,'width':32,'height':24},values
            for k,v in (('left',9),('top',12),('width',10),('height',11)):
                page.locator('#crop-'+k+'-a').fill(str(v))
            assert 'columns [9, 19), rows [12, 23)' in page.locator('#crop-note-a').inner_text()
            page.locator('#crop-width-a').fill('59'); assert page.locator('#crop-note-a').get_attribute('data-error')=='true'
            page.get_by_role('button',name='Use whole image A',exact=True).click()
            assert page.locator('#crop-width-a').input_value()=='60'
            page.locator('#image-a').set_input_files(str((args.fixtures/'exif.jpg').resolve()))
            page.wait_for_function("() => document.querySelector('#crop-height-a').value === '30' && !document.querySelector('#crop-editor-a').hidden")
            assert page.locator('#crop-width-a').input_value()=='60'
            assert page.locator('#crop-canvas-a').evaluate('c => c.getContext("2d").getImageData(10,15,1,1).data[0]')>240
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(args.out/f'controls-{width}.png'))
            matrix.append({'viewport_width':width,'device_scale_factor':dpr,'pointer_bounds':values,'numeric_edit':True,'invalid_crop':True,'reset':True,'exif_stored_orientation':True})
            context.close()
        page=browser.new_page(viewport={'width':1440,'height':1050}); page.on('pageerror',lambda error:errors.append(str(error))); page.goto(args.url)
        page.locator('#image-a').set_input_files(str((args.fixtures/'two-faces.png').resolve()))
        page.locator('#image-b').set_input_files(str((args.fixtures/'single.png').resolve()))
        page.get_by_role('button',name='Compare images',exact=True).click()
        page.wait_for_function("() => document.querySelector('#message').dataset.state === 'failed'",timeout=120000)
        assert 'multiple_faces' in page.locator('#message').inner_text(),page.locator('#message').inner_text()
        failed=page.url.split('#run=')[1]; print('Retained full-image failure: '+failed,flush=True)
        bounds=json.loads((args.fixtures/'construction.json').read_text())['crop']; l,t,r,b=bounds
        canvas=page.locator('#crop-canvas-a'); canvas.scroll_into_view_if_needed(); box=canvas.bounding_box()
        source_w=int(page.locator('#crop-width-a').input_value()); source_h=int(page.locator('#crop-height-a').input_value())
        page.mouse.move(box['x']+(l+.5)*box['width']/source_w,box['y']+(t+.5)*box['height']/source_h); page.mouse.down()
        page.mouse.move(box['x']+(r-.5)*box['width']/source_w,box['y']+(b-.5)*box['height']/source_h,steps=8); page.mouse.up()
        for key,value in (('left',l),('top',t),('width',r-l),('height',b-t)):
            page.locator('#crop-'+key+'-a').fill(str(value))
        page.screenshot(path=str(args.out/'selected-face.png'),full_page=True)
        page.get_by_role('button',name='Compare images',exact=True).click()
        page.wait_for_function("previous => location.hash.startsWith('#run=') && location.hash !== '#run=' + previous",arg=failed,timeout=30000)
        page.wait_for_function("() => !document.querySelector('#results').hidden || ['failed','cancelled'].includes(document.querySelector('#message').dataset.state)",timeout=600000)
        assert page.locator('#results').is_visible(),page.locator('#message').inner_text()
        run_id=page.url.split('#run=')[1]; assert run_id!=failed
        record=page.request.get(args.url+f'api/jobs/{run_id}').json()
        assert record['inputs'][0]['crop']==bounds and record['inputs'][1]['crop'] is None
        assert record['inputs'][0]['analysis']['decoded_rgb_sha256']==record['inputs'][1]['analysis']['decoded_rgb_sha256']
        summary=page.request.get(args.url+f'runs/{run_id}/result/summary.json').json()
        assert summary['registered_all_rms']['value']<1e-10
        assert page.locator('#regional [data-region-row]').count()==7
        with page.expect_popup() as popup:
            page.get_by_role('link',name='Input selection',exact=True).click()
        selected=popup.value; selected.wait_for_load_state()
        assert selected.locator('svg rect').first.get_attribute('x')==str(l-.5)
        assert 'Original uploaded bytes remain' in selected.locator('body').inner_text()
        selected.screenshot(path=str(args.out/'input-selection.png'),full_page=True); selected.close()
        with page.expect_download() as download:
            page.locator('#download').click()
        download.value.save_as(str(args.out/'comparison.zip'))
        page.reload(); page.locator('#results').wait_for(state='visible')
        assert page.locator('#selection').is_visible() and 'crop [' in page.locator('#pair-label').inner_text()
        page.screenshot(path=str(args.out/'comparison.png'),full_page=True)
        assert not errors,errors
        result={'passed':True,'browser':browser.version,'controls':matrix,'full_image_multiple_faces_run':failed,'cropped_native_run':run_id,
            'crop':bounds,'analysis_pixel_identity':True,'registered_rms':summary['registered_all_rms']['value'],
            'download':True,'reload':True,'source_selection_page':True,'page_errors':errors,
            'scope':'Native workflow on explicitly composited copies of one portrait; not independent reference validation'}
        (args.out/'browser.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2)); browser.close()


if __name__=='__main__':main()
