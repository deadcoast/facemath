"""Explicit source selection in stored-raster pixel coordinates."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import cv2
import numpy as np

from .observation.contracts import array_digest

MAX_IMAGE_BYTES = 16 * 1024 * 1024


def decode_upload(item: dict) -> tuple[str, bytes, np.ndarray]:
    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
        raise ValueError("Each image needs its filename and encoded contents.")
    value = item.get("data")
    if not isinstance(value, str) or len(value) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError("Each image must be at most 16 MiB.")
    raw = base64.b64decode(value, validate=True)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Each image must be at most 16 MiB.")
    if not (raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8\xff")):
        raise ValueError("Choose a PNG or JPEG image.")
    pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if pixels is None or min(pixels.shape[:2]) < 8:
        raise ValueError("The image could not be decoded. Choose a valid PNG or JPEG.")
    if pixels.shape[0] * pixels.shape[1] > 30_000_000:
        raise ValueError("Images must contain at most 30 million pixels.")
    name = item["name"].replace("\\", "/").split("/")[-1][:180]
    return name, raw, pixels


def png(pixels: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError("Could not encode the selected source pixels.")
    return encoded.tobytes()


def source_preview(item: dict) -> dict:
    name, raw, image = decode_upload(item)
    h, w = image.shape[:2]
    scale = min(1., 1024 / max(h, w))
    small = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA) if scale < 1 else image
    return {"name": name, "width": w, "height": h, "sha256": sha256(raw).hexdigest(),
            "orientation": "stored_raster", "preview": base64.b64encode(png(small)).decode()}


@dataclass(frozen=True)
class SelectedInput:
    name: str
    original: bytes
    pixels: np.ndarray
    bounds: tuple[int, int, int, int]

    @property
    def cropped(self) -> bool:
        h, w = self.pixels.shape[:2]
        return self.bounds != (0, 0, w, h)

    @property
    def selected(self) -> np.ndarray:
        left, top, right, bottom = self.bounds
        return self.pixels[top:bottom, left:right]

    def record(self, side: str, analysis_bytes: bytes) -> dict:
        left, top, right, bottom = self.bounds
        h, w = self.pixels.shape[:2]
        return {"name": self.name, "sha256": sha256(self.original).hexdigest(), "width": w, "height": h,
            "decoded_rgb_sha256": array_digest(cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB)),
            "crop": list(self.bounds) if self.cropped else None,
            "analysis": {"file": f"analysis_{side}.png" if self.cropped else f"input_{side}.image",
                "width": right-left, "height": bottom-top, "sha256": sha256(analysis_bytes).hexdigest(),
                "decoded_rgb_sha256": array_digest(cv2.cvtColor(self.selected, cv2.COLOR_BGR2RGB)),
                "to_original": [[1, 0, left], [0, 1, top], [0, 0, 1]],
                "coordinates": "pixel centers in selected raster; original center = selected center + (left, top)"}}


def select_input(item: dict) -> SelectedInput:
    name, raw, image = decode_upload(item)
    h, w = image.shape[:2]
    crop = item.get("crop")
    bounds = (0, 0, w, h) if crop is None else crop
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or any(type(v) is not int for v in bounds):
        raise ValueError("Crop bounds must be four integer pixel indices: left, top, right, bottom.")
    left, top, right, bottom = bounds
    if not (0 <= left < right <= w and 0 <= top < bottom <= h and right-left >= 8 and bottom-top >= 8):
        raise ValueError("The crop must stay inside the image and be at least 8 × 8 pixels.")
    return SelectedInput(name, raw, image, tuple(bounds))


def selection_page(inputs: list[dict]) -> str:
    from .observation.artifacts import REPORT_TOKENS
    sections = []
    for side, item in zip(("a", "b"), inputs):
        w, h = item["width"], item["height"]
        left, top, right, bottom = item["crop"] or (0, 0, w, h)
        a = item["analysis"]
        selected = f"analysis_{side}.png" if item["crop"] else f"preview_{side}.png"
        sections.append(f'''<section><h2>{side.upper()} · {escape(item['name'])}</h2>
<p>Original {w} × {h} pixels. Selected columns [{left}, {right}) and rows [{top}, {bottom}): {a['width']} × {a['height']} pixels.</p>
<div class="pair"><figure><svg role="img" aria-label="Original {side.upper()} with selected crop" viewBox="-.5 -.5 {w} {h}">
<image href="preview_{side}.png" x="-.5" y="-.5" width="{w}" height="{h}"/>
<rect x="{left-.5}" y="{top-.5}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#ffca80" stroke-width="3" vector-effect="non-scaling-stroke"/></svg>
<figcaption>Original stored orientation · amber outline is the selected region.</figcaption></figure>
<figure><img src="{selected}" alt="Selected analysis image {side.upper()}"><figcaption>Actual analysis input · no resampling.</figcaption></figure></div>
<p>To recover original pixel centers, add ({left}, {top}) to the selected-raster coordinates.</p>
<p>Original bytes: <code>{item['sha256']}</code><br>Analysis RGB array: <code>{a['decoded_rgb_sha256']}</code></p></section>''')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FACEMATH · Source selection</title><style>{REPORT_TOKENS}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:24px}}a{{color:var(--accent)}}p,figcaption{{color:var(--muted)}}code{{overflow-wrap:anywhere}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0;background:var(--panel);padding:12px}}
img,svg{{display:block;width:100%;max-height:600px;object-fit:contain}}:focus-visible{{outline:2px solid var(--accent)}}
@media(max-width:760px){{.pair{{grid-template-columns:1fr}}main{{padding:14px}}}}</style><main>
<h1>Source selection</h1><p><a href="result/report.html">Open complete comparison</a> · <a href="request.json">Input records</a></p>
<p>Analysis coordinates belong to the selected raster. Original uploaded bytes remain in input_a.image and input_b.image. Previews use stored orientation, not EXIF display rotation. Selecting a crop does not establish anatomical accuracy.</p>
{''.join(sections)}</main></html>'''
