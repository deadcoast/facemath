"""Exercise actual pixel-center clicks, downloads and draft boundaries.

The oracle locates painted pixels from the DOM content box and Canvas CTM;
it does not call sourceAt/canvasAt or replay the application's inverse mapping.
All clicks are automated synthetic controls, never independent human labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from playwright.sync_api import sync_playwright


ORACLE = """([x,y])=>{const r=canvas.getBoundingClientRect(),s=getComputedStyle(canvas),t=ctx.getTransform();
  const n=k=>parseFloat(s[k])||0,left=n('borderLeftWidth')+n('paddingLeft'),right=n('borderRightWidth')+n('paddingRight'),top=n('borderTopWidth')+n('paddingTop'),bottom=n('borderBottomWidth')+n('paddingBottom');
  const cw=n('width')-(s.boxSizing==='border-box'?left+right:0),ch=n('height')-(s.boxSizing==='border-box'?top+bottom:0),sx=r.width/(cw+left+right),sy=r.height/(ch+top+bottom);
  return [r.left+left*sx+(view.tx+(x+.5)*view.scale)*t.a*cw*sx/canvas.width,r.top+top*sy+(view.ty+(y+.5)*view.scale)*t.d*ch*sy/canvas.height];}"""


def run(workbench: Path, out: Path, browser_path: str) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    result = {'format':'facemath.annotation_browser.v20','started_utc':datetime.now(timezone.utc).isoformat(),
              'workbench':str(workbench),'annotations':'automated indexed-raster fixtures, not human labels',
              'oracle':'painted pixel centers through actual DOM content box and Canvas CTM', 'views':[], 'passed':False}
    errors, requests = [], []
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=browser_path,headless=True)
            result['browser']=browser.version
            variants=[('desktop',1280,1000,1,''),('fractional-dpr',1280,1000,1.25,''),
                      ('retina',1280,1000,2,''),('narrow',390,844,3,''),
                      ('border-padding',1280,1000,1,'border:7px solid cyan;padding:5px 9px;'),
                      ('css-scale',1280,1000,1.5,'transform:scale(.83);transform-origin:top left;')]
            for name,w,h,dpr,style in variants:
                context=browser.new_context(viewport={'width':w,'height':h},device_scale_factor=dpr,accept_downloads=True)
                page=context.new_page()
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('request',lambda r:requests.append(r.url) if r.url.startswith(('http:','https:')) else None)
                page.on('dialog',lambda d:(_ for _ in ()).throw(AssertionError('Unexpected native dialog: '+d.message)))
                page.goto(workbench.as_uri(),wait_until='load')
                page.wait_for_function('image.complete && image.naturalWidth===48 && image.naturalHeight===32')
                if style:
                    page.locator('#stage').evaluate('(el,style)=>{el.style.cssText=style;window.dispatchEvent(new Event("resize"));}',style)
                assert page.locator('#progress-label').inner_text()=='0 / 8 resolved'
                assert page.locator('#export').is_disabled()
                page.locator('#rater').fill('synthetic-'+name);page.locator('#rater').press('Tab')
                expected=[[0,0],[47,0],[0,31],[47,31],[24.25,15.5],[15,21],[32,21],[24,28]]

                def click(i, xy):
                    page.locator('#point-list button').nth(i).click()
                    page.locator('#stage').scroll_into_view_if_needed()
                    page.mouse.click(*page.evaluate(ORACLE,xy))

                for i,xy in enumerate(expected):click(i,xy)
                assert not page.locator('#export').is_disabled(),(name,page.evaluate('points'),page.locator('#status').inner_text())
                actual=page.evaluate('exportReference()')
                assert actual['metadata']['coordinate_capture']['format']=='facemath.reference_canvas.v20'
                error=max(abs(x-y) for a,b in zip(expected,actual['xy']) for x,y in zip(a,b))
                assert error<=.002,(name,error,actual['xy'])
                click(0,[-.01,0])
                assert page.evaluate('points[0].xy')==[0,0]
                assert 'inside the original source raster' in page.locator('#status').inner_text()
                # Trace drawing commands to verify marker and magnifier positions,
                # independently of the application mapping helpers.
                drawing=page.evaluate('''()=>{const arcs=[];const original=ctx.arc.bind(ctx);ctx.arc=(...a)=>{arcs.push(a.slice(0,2));original(...a);};draw();ctx.arc=original;
                  window.lensArguments=[];const originalLens=lens.drawImage.bind(lens);lens.drawImage=(...a)=>{window.lensArguments=a.slice(1);originalLens(...a);};return {arcs,view:{...view}};}''')
                for xy,arc in zip(expected,drawing['arcs']):
                    assert max(abs(arc[i]-(drawing['view']['t'+('x' if i==0 else 'y')]+(xy[i]+.5)*drawing['view']['scale'])) for i in (0,1))<.002
                page.mouse.move(*page.evaluate(ORACLE,[24.25,15.5]))
                lens=page.evaluate('window.lensArguments')
                assert abs(lens[0]+lens[2]/2-.5-24.25)<.002 and abs(lens[1]+lens[3]/2-.5-15.5)<.002
                page.locator('#zoom-in').click()
                click(4,[24.25,15.5])
                # Chromium WheelEvent coordinates are integer CSS pixels. Anchor
                # at that actual event location, not a fractional requested move.
                pos=[round(v) for v in page.evaluate(ORACLE,[24.25,15.5])]
                zero=page.evaluate(ORACLE,[0,0]);one=page.evaluate(ORACLE,[1,1])
                anchor=[(v-a)/(b-a) for v,a,b in zip(pos,zero,one)]
                old_scale=page.evaluate('view.scale')
                page.evaluate("()=>{canvas.addEventListener('wheel',e=>{window.lastWheel=[e.clientX,e.clientY];},{capture:true});}")
                page.mouse.move(*pos);page.mouse.wheel(0,-100)
                page.wait_for_function('(old)=>view.scale>old',arg=old_scale)
                assert page.evaluate('window.lastWheel')==pos
                anchored=page.evaluate(ORACLE,anchor)
                assert max(abs(a-b) for a,b in zip(pos,anchored))<.002,(name,pos,anchored)
                page.keyboard.down('Alt');page.mouse.move(*anchored);page.mouse.down()
                page.mouse.move(anchored[0]+20,anchored[1]+12,steps=3);page.mouse.up();page.keyboard.up('Alt')
                click(4,[24.25,15.5]);page.locator('#stage').focus()
                before=page.evaluate('JSON.stringify(points)');view_before=page.evaluate('view.tx')
                page.keyboard.press('Alt+ArrowRight')
                assert page.evaluate('JSON.stringify(points)')==before
                assert abs(page.evaluate('view.tx')-view_before-32)<1e-9
                page.keyboard.press('ArrowRight');expected[4][0]+=.25
                page.keyboard.press('Shift+ArrowDown');expected[4][1]+=1
                page.locator('#undo').click();expected[4][1]-=1
                page.locator('#redo').click();expected[4][1]+=1
                page.locator('#unobservable').click()
                assert page.evaluate('points[4].visible') is False
                page.locator('#undo').click()
                page.locator('#fit').click()
                with page.expect_download() as download:page.locator('#export').click()
                target=out/(name+'.reference.json');download.value.save_as(target)
                exported=json.loads(target.read_text(encoding='utf-8'))
                assert exported['kind']=='synthetic_fixture'
                error=max(error,max(abs(x-y) for a,b in zip(expected,exported['xy']) for x,y in zip(a,b)))
                assert error<=.002,(name,error)
                legacy=page.evaluate('''()=>{const key='facemath.ref4:'+data.image_sha256+':legacy',raw=JSON.stringify({points,events,note:'old convention'});localStorage.setItem(key,raw);return {key,raw};}''')
                page.locator('#rater').fill('legacy');page.locator('#rater').press('Tab')
                assert page.locator('#progress-label').inner_text()=='0 / 8 resolved'
                assert 'Legacy draft retained' in page.locator('#status').inner_text()
                assert page.evaluate('(k)=>localStorage.getItem(k)',legacy['key'])==legacy['raw']
                page.locator('#rater').fill('synthetic-'+name);page.locator('#rater').press('Tab')
                assert page.locator('#draft-recovery').is_visible() and page.locator('#export').is_disabled()
                page.locator('#restore-draft').press('Enter')
                assert page.evaluate('exportReference().xy')==exported['xy']
                assert page.locator('#stage').evaluate('(el)=>document.activeElement===el')
                # A reload restores only after an explicit choice; cancel preserves
                # the current blank state. Invalid bindings stay unimported.
                page.reload();page.wait_for_function('image.complete && image.naturalWidth>0')
                page.locator('#rater').fill('synthetic-'+name);page.locator('#rater').press('Tab')
                assert page.locator('#draft-recovery').is_visible()
                page.locator('#keep-input').click()
                assert page.locator('#progress-label').inner_text()=='0 / 8 resolved'
                page.evaluate('''()=>{const key=draftKey('invalid');const draft=JSON.parse(localStorage.getItem(draftKey(activeRater)));draft.image_sha256='0'.repeat(64);draft.rater_id='invalid';localStorage.setItem(key,JSON.stringify(draft));}''')
                page.locator('#rater').fill('invalid');page.locator('#rater').press('Tab')
                assert 'Incompatible saved draft retained' in page.locator('#status').inner_text()
                assert page.locator('#progress-label').inner_text()=='0 / 8 resolved'
                page.evaluate("()=>{Storage.prototype.setItem=function(){throw new Error('controlled storage unavailable');};}")
                click(0,[8,8])
                assert 'storage unavailable' in page.locator('#save-state').inner_text()
                assert page.evaluate('points[0].xy')==[8,8]
                page.evaluate('window.scrollTo(0,0)')
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'),name
                page.screenshot(path=str(out/(name+'.png')),full_page=True)
                result['views'].append({'name':name,'device_scale_factor':dpr,'maximum_error_px':error,
                    'edge_centers':True,'zoom_pan_and_keyboard':True,'marker_and_lens':True,
                    'draft_restore_and_isolation':True,'invalid_draft_and_storage_failure':True,'file_navigation':True})
                context.close()
            browser.close()
        assert not errors and not requests,(errors,requests)
        result['passed']=True
    except Exception:
        result['failure']=traceback.format_exc()
        raise
    finally:
        result.update(console_errors=errors,remote_requests=requests,finished_utc=datetime.now(timezone.utc).isoformat())
        (out/'browser.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbench',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--browser',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.workbench.resolve(),args.out.resolve(),args.browser),indent=2))
