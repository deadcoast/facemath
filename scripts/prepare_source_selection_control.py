"""Create explicitly derived input-selection controls, not a new reference cohort."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('source',type=Path); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    raw=args.source.read_bytes(); image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    h,w=image.shape[:2]; face=cv2.resize(image,(round(w*600/h),600),interpolation=cv2.INTER_AREA)
    h,w=face.shape[:2]; composite=np.full((h+40,2*w+60,3),110,np.uint8)
    composite[20:20+h,20:20+w]=face; composite[20:20+h,40+w:40+2*w]=face
    for name,pixels in (('single.png',face),('two-faces.png',composite)):
        ok,data=cv2.imencode('.png',pixels); assert ok; (args.out/name).write_bytes(data.tobytes())
    yy,xx=np.indices((40,60)); indexed=np.stack((xx,yy,xx+yy),axis=-1).astype(np.uint8)
    ok,data=cv2.imencode('.png',indexed); assert ok; (args.out/'indexed.png').write_bytes(data.tobytes())
    rgb=np.zeros((30,60,3),np.uint8); rgb[:,:20,0]=255
    exif=Image.Exif(); exif[274]=6; Image.fromarray(rgb).save(args.out/'exif.jpg',exif=exif)
    shutil.copyfile(args.source,args.out/'original.image')
    record={'source_path':str(args.source),'source_sha256':sha256(raw).hexdigest(),
        'construction':'One retained portrait resized to height 600, duplicated into a two-face composite; not independent portraits or anatomical reference evidence.',
        'crop':[20,20,20+w,20+h],'source_dimensions':[image.shape[1],image.shape[0]],
        'analysis_dimensions':[w,h],'copied_original':'original.image'}
    (args.out/'construction.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
