"""Self-contained inspection bundle with real arrays and byte-level verification.

Hashes establish consistency with the manifest, not external authorship or a
trusted timestamp. All paths are sandboxed to the selected output directory.
"""
from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any
import json
import os
import shutil
from .publication import publication_stage

import cv2
import numpy as np

from .contracts import ObservationError, array_digest, canonical_json
from .instrument import InstrumentResult
from .journal import verify_journal
from .regional_view import REGIONAL_CSS, regional_map_images


def _png(image: np.ndarray) -> bytes:
    encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, buf = cv2.imencode(".png", encoded_image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _overlay(result: InstrumentResult) -> np.ndarray:
    image = cv2.addWeighted(result.state.image_a, .5, result.state.image_b, .5, 0)
    pa, pb = result.state.xy_a, result.state.xy_b
    # Distinct marker forms, not an unlabeled interpolation portrayed as measurement.
    for a, b in zip(pa, pb):
        aa, bb = tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))
        cv2.line(image, aa, bb, (255, 200, 80), 1, cv2.LINE_AA)
        cv2.circle(image, aa, 1, (245, 245, 245), 1)
        cv2.drawMarker(image, bb, (30, 220, 225), cv2.MARKER_CROSS, 3, 1)
    return image


def _edge_overlay(result: InstrumentResult, anchor: str) -> np.ndarray:
    """Show accepted image-edge locations, never relocated model landmarks."""
    image = cv2.addWeighted(result.state.image_a, .5, result.state.image_b, .5, 0)
    if result.edges is not None:
        for key, pair in result.edges.pairs.items():
            if not key.startswith(anchor + "_"):
                continue
            a = pair.a.points + pair.a.normals*pair.a.best_offset_px[:, None]
            b = pair.b.points + pair.b.normals*pair.b.best_offset_px[:, None]
            for pa, pb in zip(a[pair.accepted], b[pair.accepted]):
                aa, bb = tuple(np.rint(pa).astype(int)), tuple(np.rint(pb).astype(int))
                cv2.line(image, aa, bb, (255, 200, 80), 1, cv2.LINE_AA)
                cv2.circle(image, aa, 2, (245, 245, 245), 1)
                cv2.drawMarker(image, bb, (30, 220, 225), cv2.MARKER_CROSS, 5, 1)
    return image


REPORT_TOKENS = ':root{--bg:#10141a;--panel:#192029;--ink:#e7eef4;--muted:#a2b1c1;--line:#354252;--accent:#71ddd6;--scroll-thumb:#a2b1c1;--scroll-track:#192029;--scroll-hover:#e7eef4;--scroll-active:#71ddd6;scrollbar-color:var(--scroll-thumb) var(--scroll-track);scrollbar-width:auto}'


def _html(result: InstrumentResult, arrays: dict[str, np.ndarray]) -> str:
    def uri(image: np.ndarray) -> str:
        return "data:image/png;base64," + b64encode(_png(image)).decode()
    ev = result.evidence()
    from .readout import render_pair_readout
    readout_section = render_pair_readout(ev)
    from .intrinsic_report import render_intrinsic_section
    intrinsic_section = render_intrinsic_section(result.intrinsic) if result.intrinsic is not None else ""
    rows = []
    for m in result.profile.measurements:
        value = "WITHHELD" if m.value is None else f"{m.value:.6g}"
        interval = "—" if m.interval is None else f"[{m.interval[0]:.5g}, {m.interval[1]:.5g}]"
        rows.append(f'<tr><td><code>{escape(m.name)}</code><small>{escape(m.interpretation)}</small></td>'
                    f'<td class="number">{value}</td><td>{escape(m.unit)}</td>'
                    f'<td class="number">{m.support:,} / {m.total:,}</td><td>{interval}'
                    f'<small>{escape(m.exclusion or "")}</small></td></tr>')
    array_rows = "".join(f'<tr><td><code>{escape(name)}</code></td><td>{list(a.shape)}</td><td>{a.dtype}</td>'
                         f'<td><code>{array_digest(a)[:18]}…</code></td></tr>' for name, a in sorted(arrays.items()))
    warnings = "".join(f"<li>{escape(w)}</li>" for w in ev["warnings"])
    limits = "".join(f"<li>{escape(w)}</li>" for w in ev["limits"])
    code = escape(result.environment["engine_code_sha256"])
    edge_section = ""
    if result.edges:
        summaries = result.edges.summary()
        erows = "".join(f"<tr><td>{escape(k)}</td><td>{v['accepted']} / {v['rays']}</td><td>{v['source_supported']}</td><td>{v['median_absolute_offset_px']:.3f}</td></tr>" if v['median_absolute_offset_px'] is not None else f"<tr><td>{escape(k)}</td><td>0 / {v['rays']}</td><td>{v['source_supported']}</td><td>withheld</td></tr>" for k, v in summaries['profiles'].items())
        edge_section = f"<h2>Image-edge support</h2><p>Landmark-guided normal-ray searches. Matching image gradients are diagnostic, not proof of correct anatomy. Landmarks are never moved to an edge candidate. Complete ray responses and rejection masks are in arrays.npz.</p><div class='scroll'><table><tr><th>Source / region</th><th>Accepted / attempted rays</th><th>Full source support</th><th>Median |normal offset| · px</th></tr>{erows}</table></div>"
        edge_views = "".join(f'<figure><img src="{uri(_edge_overlay(result, side))}" alt="Paired image-edge locations on {side.upper()}-anchored rays">'
            f'<figcaption>{side.upper()}-anchored rays · white circle: image A edge; cyan cross: image B edge; amber line: normal displacement.</figcaption></figure>'
            for side in ("a", "b"))
        edge_section += f'<h3>Locate the measured image edges</h3><p>Only rays localized in both images are drawn. Unresolved rays remain in the table denominator. These markers show selected gradients, not corrected anatomical landmarks.</p><div class="gallery edge-gallery">{edge_views}</div>'
    from .stratified_report import render_routing_pair
    routing_section = render_routing_pair(result.localization.calibration) if result.localization else ""
    acquisition_section = ""
    if result.acquisition_groups is not None:
        from .acquisition_report import acquisition_panel
        acquisition_section = acquisition_panel(result.acquisition_groups)
    localization_section = ""
    if result.localization:
        info = result.localization.to_dict()
        centering_note = ""
        if info.get("bias_sha256"):
            centering_note = "<p><strong>Development-qualified error centers.</strong> The observed gap remains the raw measurement. The bounds use reference-position centers shifted by the frozen development-only localization-bias model. The mesh, raster, registration and measured geometry have not moved. A raw gap need not lie inside an interval for the independently referenced gap.</p>"
        lrows = "".join(f"<tr><td>{escape(r['name'])}</td><td>{r['observed_gap_ipd']:.4f}</td><td>[{r['lower_ipd']:.4f}, {r['upper_ipd']:.4f}]</td><td>{escape(r['status'])}</td></tr>" if r['observed_gap_ipd'] is not None else f"<tr><td>{escape(r['name'])}</td><td colspan='3'>withheld</td></tr>" for r in info['rows'])
        localization_section = f"<h2>Reference-calibrated localization envelope</h2><div class='panel'>{centering_note}Scope: <strong>{escape(info['scope'])}</strong>. Acquisition domain: {escape(info['domain'])}. These bounds hold the observed transforms and normalization fixed; they are not confidence intervals for re-fitted true shape or identity.</div><div class='scroll'><table><tr><th>Annotated landmark</th><th>Observed gap</th><th>Error envelope · observed eye spans</th><th>At selected tolerance</th></tr>{lrows}</table></div><p><a href='localization_calibration.json'>Recorded error calibration</a> · all eight bounds retained in arrays.npz.</p>"
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FACEMATH · Anatomical observation instrument</title>
<style>
{REPORT_TOKENS}
{REGIONAL_CSS}
*::-webkit-scrollbar{{width:12px;height:12px}}*::-webkit-scrollbar-track{{background:var(--scroll-track)}}*::-webkit-scrollbar-thumb{{background:var(--scroll-thumb);border:3px solid var(--scroll-track)}}*::-webkit-scrollbar-thumb:hover{{background:var(--scroll-hover)}}*::-webkit-scrollbar-thumb:active{{background:var(--scroll-active)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:3px}}a:hover,summary:hover{{color:var(--ink)}}a:active,summary:active{{color:var(--accent)}}a,summary,input[type=range]{{cursor:pointer}}
#structure table{{min-width:790px;font-variant-numeric:tabular-nums}}#structure caption{{text-align:left;padding:12px 14px;color:var(--muted)}}#structure th[scope=row]{{font-weight:500;background:transparent}}#structure td{{white-space:nowrap}}#structure td small{{white-space:normal}}
#edge-comparison table{{min-width:1120px;font-variant-numeric:tabular-nums}}
#bilateral table{{min-width:1000px;font-variant-numeric:tabular-nums}}#bilateral caption{{text-align:left;padding:12px 14px;color:var(--muted)}}
#bilateral summary{{min-height:44px;padding:10px 0}}#bilateral .bilateral-gallery{{grid-template-columns:repeat(2,minmax(0,1fr));max-width:850px}}#bilateral svg{{display:block;width:100%}}
@media(max-width:760px){{#bilateral .bilateral-gallery{{grid-template-columns:1fr}}}}
@media(forced-colors:active){{:root{{scrollbar-color:auto}}:focus-visible{{outline-color:Highlight}}}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif;margin:0}}
main{{max-width:1320px;padding:38px 30px 80px;margin:auto}}.eyebrow{{font:12px/1.5 ui-monospace,monospace;letter-spacing:.15em;color:var(--accent)}}
h1{{font-size:34px;letter-spacing:-.025em;margin:10px 0}}h2{{font-size:21px;margin:35px 0 12px}}p{{max-width:92ch;color:var(--muted)}}
header{{border-bottom:1px solid var(--line);padding-bottom:24px}}.tag{{border:1px solid var(--line);padding:5px 10px;font:12px ui-monospace,monospace;display:inline-block;margin:0 8px 8px 0}}
.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}figure{{background:var(--panel);margin:0;padding:12px;border:1px solid var(--line)}}
.edge-gallery{{grid-template-columns:repeat(2,minmax(0,1fr))}}
figure img{{width:100%;display:block;image-rendering:auto}}figcaption{{padding:10px 0 0;color:var(--muted);font-size:13px}}
.panel{{padding:18px;background:var(--panel);border-left:3px solid var(--accent);margin:18px 0}}
.scroll{{overflow:auto;border:1px solid var(--line)}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;background:var(--panel)}}th,td{{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}}
small{{display:block;color:var(--muted);font-size:12px;max-width:68ch}}code{{font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}}.number{{font-variant-numeric:tabular-nums;white-space:nowrap}}
a{{color:var(--accent)}}summary{{cursor:pointer;font-weight:600}}input[type=range]{{width:min(100%,500px)}}.blend{{position:relative;max-width:360px}}.blend img{{width:100%;display:block}}.blend img:last-child{{position:absolute;inset:0;opacity:.5}}
footer{{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:18px}}
@media(max-width:760px){{.gallery{{grid-template-columns:1fr}}main{{padding:20px 14px}}h1{{font-size:28px}}}}
</style><main><header><div class="eyebrow">FACEMATH / WAVE 10 / OBSERVATION → FIELD → EVIDENCE</div>
<h1>Anatomical correspondence, exposed.</h1>
<span class="tag">{escape(result.state.observation_a.provider)}</span><span class="tag">{len(result.state.xy_a)} observed landmarks</span>
<span class="tag">No identity scoring</span><span class="tag">No population calibration attached</span>
<p>One canonical state binds observations, the registration gauge, regional contours, anatomical interpolation, and independent image-flow checks. Every reported value has a raw array path in this bundle.</p></header>
{readout_section}<h2>Registered views</h2><div class="gallery">
<figure><img src="{uri(result.state.image_a)}" alt="Canonical image A"><figcaption>A · source observation transformed once</figcaption></figure>
<figure><img src="{uri(result.state.image_b)}" alt="Canonical image B"><figcaption>B · reference display chart</figcaption></figure>
<figure><img src="{uri(_overlay(result))}" alt="Landmark residual overlay"><figcaption>White circle: A. Cyan cross: B. Amber line: positional residual. This overlay is a display, not the score.</figcaption></figure></div>
<details><summary>Interactive opacity inspection</summary><p>Changes only the display; it does not re-estimate landmarks or recompute any metric.</p>
<div class="blend"><img src="{uri(result.state.image_a)}" alt="A"><img id="blend-b" src="{uri(result.state.image_b)}" alt="B"></div>
<label>A → B <input id="opacity" type="range" min="0" max="100" value="50" aria-label="B opacity"></label></details>
<div class="panel"><strong>Read the units before the number.</strong><br>
Landmark residuals are normalized by the geometric mean of registered interocular spans. A within-tolerance fraction counts landmarks below the declared threshold ({result.state.config.tolerance_ipd:g} spans). Neither quantity is an identity probability. The intervals below measure conditional observation sensitivity, not established detector accuracy.</div>
{acquisition_section}{routing_section}{intrinsic_section}<h2>Measured profile</h2><div class="scroll"><table><thead><tr><th>Measurement / interpretation</th><th>Value</th><th>Unit</th><th>Support / denominator</th><th>95% sensitivity interval</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{edge_section}{localization_section}<h2>Raw numerical state</h2><p><a href="arrays.npz">arrays.npz</a> · <a href="evidence.json">evidence.json</a> · <a href="observations.json">observations.json</a> · <a href="events.jsonl">actual execution events</a> · <a href="manifest.json">byte manifest</a></p>
<div class="scroll"><table><thead><tr><th>Array</th><th>Shape</th><th>dtype</th><th>Content digest</th></tr></thead><tbody>{array_rows}</tbody></table></div>
<h2>Observation constraints</h2><ul>{warnings or '<li>No additional source warnings.</li>'}</ul><ul>{limits}</ul>
<footer>Recipe SHA-256: <code>{escape(result.state.recipe_sha256)}</code><br>Engine source SHA-256: <code>{code}</code><br>
This bundle verifies internal integrity, not external authorship or historical model telemetry. Report has no remote assets, analytics, or network calls.</footer>
<script>document.getElementById('opacity').addEventListener('input', e => {{document.getElementById('blend-b').style.opacity=e.target.value/100;}});</script>
</main></html>'''


def write_bundle(result: InstrumentResult, destination: str | Path,
                 source_a: np.ndarray, source_b: np.ndarray) -> Path:
    """Create-only, atomic directory publication. Never replaces an earlier run."""
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"evidence bundle already exists: {destination}")
    result.state.observation_a.require_image(source_a)
    result.state.observation_b.require_image(source_b)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = publication_stage(destination.parent, ".facemath-")
    try:
        arrays = result.arrays()
        if result.acquisition_groups is not None:
            from .acquisition import write_group_files
            for name, group in zip(("a", "b"), result.acquisition_groups):
                write_group_files(group, temp, f"acquisition_{name}_")
        if result.qualification_evidence is not None:
            (temp / "qualification_snapshot.zip").write_bytes(result.qualification_evidence.snapshot)
        np.savez_compressed(temp / "arrays.npz", **arrays)
        (temp / "evidence.json").write_text(json.dumps(result.evidence(), indent=2, allow_nan=False), encoding="utf-8")
        from .readout import pair_readout
        (temp / "summary.json").write_text(json.dumps(pair_readout(result.evidence()), indent=2, allow_nan=False), encoding="utf-8")
        (temp / "observations.json").write_text(json.dumps({"a": result.state.observation_a.to_dict(), "b": result.state.observation_b.to_dict()}, indent=2, allow_nan=False), encoding="utf-8")
        (temp / "environment.json").write_text(json.dumps(result.environment, indent=2, allow_nan=False), encoding="utf-8")
        (temp / "events.jsonl").write_text(result.journal_jsonl, encoding="utf-8")
        if result.localization is not None:
            (temp / "localization_calibration.json").write_text(json.dumps(result.localization.calibration.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        if result.intrinsic is not None:
            from .intrinsic_report import relation_csv
            (temp / "intrinsic.json").write_text(json.dumps(result.intrinsic.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
            (temp / "intrinsic.csv").write_text(relation_csv(result.intrinsic), encoding="utf-8", newline="")
        for name, image in (("source_a.png", source_a), ("source_b.png", source_b),
                            ("canonical_a.png", result.state.image_a), ("canonical_b.png", result.state.image_b),
                            ("landmark_overlay.png", _overlay(result))):
            (temp / name).write_bytes(_png(image))
        for name, image in regional_map_images(result):
            (temp / name).write_bytes(_png(image))
        (temp / "report.html").write_text(_html(result, arrays), encoding="utf-8")
        manifest: dict[str, Any] = {
            "format": "facemath.bundle.v3", "recipe_sha256": result.state.recipe_sha256,
            "files": {p.name: {"sha256": sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                      for p in sorted(temp.iterdir())},
            "arrays": {k: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": array_digest(a)} for k, a in sorted(arrays.items())},
            "array_semantics": {
                "coordinates": "xy pixel coordinates; rows use observation.names; image arrays are y,x",
                "anatomical_uv": "A-domain B-minus-A displacement; zero outside support is a sentinel, not agreement",
                "triangle_jacobians": "d(B_coordinates)/d(A_coordinates); piecewise-affine inferred anatomy",
                "triangle_green_strain": "0.5*(F.T@F-I); geometric distortion, not physical tissue deformation",
                "triangle_principal_log_stretch": "log singular values of F; invalid triangles excluded by triangle_valid",
                "heldout_residual_vectors": {"axis_0": result.profile.heldout_labels, "units": "registered_eye_span_geomean",
                    "missingness": "rows with heldout_fit_supported=false are zero sentinels; failures remain in evidence.json"},
                "sensitivity_draws": {"axis_1": result.profile.sensitivity_regions, "kind": "conditional_observation_sensitivity",
                                      "missingness": "rows with sensitivity_draw_supported=false or columns with sensitivity_region_supported=false contain zero sentinels; failed refits withhold intervals"},
                "regional_*": "seven regional rows in evidence.json regional_comparison; B-minus-A observed displacement; existing A-to-B polar maps; unsupported array slots are zero sentinels; no localization uncertainty propagation",
                "landmark_supported": "source-coordinate in-bounds intersection for both observations",
                "flow_admissible": "heuristic texture/cycle/overlap support, not correctness probability",
                "edge_*": "landmark-guided normal-ray gradient response; offsets are meaningful only where accepted; not independent anatomy or landmark corrections",
                "intrinsic_*": "source-chart reference8 relations with uncertain eye-span normalization; arrays use the 256-row registry in intrinsic.json; source axes A,B; missing bounds use explicit masks; no independent votes",
                "structure_*": "fourteen named rows in intrinsic.json structural_comparison; same source graph; nominal arrays use observed points; delta is B minus A; unavailable masks retain zero sentinels",
                "bilateral_*": "five fixed rows in intrinsic.json bilateral_comparison; horizontal/vertical components and source markers in intrinsic source charts; display segments separately map expected then observed points into existing canonical pixel centers; source bounds share the existing reference-error graph; false masks mark zero sentinels",
                "localization_*": "eight reference-protocol points only; source-ellipse-derived outer gap bounds in the fixed observed coordinate gauge; no extrapolation to 478 points",
            },
            "integrity_limit": "self-consistency hashes; not an externally signed execution attestation",
        }
        manifest["manifest_content_sha256"] = sha256(canonical_json(manifest).encode()).hexdigest()
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        verify_bundle(temp)
        # Directory rename is atomic on this filesystem; create-only checked again.
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temp, destination)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination


def verify_bundle(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_content_sha256")
    if sha256(canonical_json(manifest).encode()).hexdigest() != digest:
        raise ObservationError("manifest content hash mismatch")
    if manifest["format"] != "facemath.bundle.v3":
        raise ObservationError("unknown bundle format")
    from .study import verify_inventory
    for relative, info in manifest["files"].items():
        target = root / relative
        if target.is_symlink() or target.resolve().parent != root or Path(relative).name != relative:
            raise ObservationError("manifest path escapes bundle or is a symlink")
        content = target.read_bytes()
        if len(content) != info["bytes"] or sha256(content).hexdigest() != info["sha256"]:
            raise ObservationError(f"artifact mismatch: {relative}")
    verify_inventory(root, manifest["files"], exclude=("manifest.json",))
    with np.load(root / "arrays.npz", allow_pickle=False) as data:
        if set(data.files) != set(manifest["arrays"]):
            raise ObservationError("array inventory mismatch")
        for name, spec in manifest["arrays"].items():
            a = data[name]
            if list(a.shape) != spec["shape"] or a.dtype.str != spec["dtype"] or array_digest(a) != spec["sha256"]:
                raise ObservationError(f"array contract mismatch: {name}")
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    if "summary.json" in manifest["files"]:
        from .readout import pair_readout
        if json.loads((root / "summary.json").read_text()) != pair_readout(evidence):
            raise ObservationError("compact readout differs from authoritative instrument evidence")
    if evidence.get("intrinsic_profile") is not None:
        if "intrinsic.json" not in manifest["files"] or "intrinsic.csv" not in manifest["files"]:
            raise ObservationError("intrinsic profile lost its external data artifacts")
        if json.loads((root / "intrinsic.json").read_text(encoding="utf-8")) != evidence["intrinsic_profile"]:
            raise ObservationError("intrinsic report differs from instrument evidence")
    if evidence.get("measured_acquisition") is not None:
        from .acquisition import load_group_files
        groups = tuple(load_group_files(root, f"acquisition_{n}_") for n in ("a", "b"))
        if canonical_json([g.to_dict() for g in groups]) != canonical_json(evidence["measured_acquisition"]):
            raise ObservationError("pair measured acquisition differs from recomputed source sidecars")
        with np.load(root/"arrays.npz", allow_pickle=False) as arrays:
            for name, group in zip(("a", "b"), groups):
                for key, value in group.arrays(f"acquisition_{name}_").items():
                    if key not in arrays or not np.array_equal(arrays[key], value):
                        raise ObservationError("pair measured source array differs from pixel recomputation")
        from .bias import load_localization_calibration
        from .stratified import BoundStratifiedCalibration
        model = load_localization_calibration(json.loads((root/"localization_calibration.json").read_text()))
        if not isinstance(model, BoundStratifiedCalibration) or any(
                r.measured_group_sha256 != g.digest for r, g in zip(model.routes, groups)):
            raise ObservationError("measured source sidecars differ from bound calibration routes")
    admission = evidence.get("empirical_admission", {})
    if admission.get("format") == "facemath.validated_qualification_link.v10":
        from .deployment import load_snapshot
        qualified = load_snapshot(root / "qualification_snapshot.zip")
        if canonical_json(qualified.summary) != canonical_json(admission):
            raise ObservationError("pair empirical admission differs from replayed raw study")
        from .bias import load_localization_calibration
        from .acquisition import load_group_files
        model = load_localization_calibration(json.loads((root / "localization_calibration.json").read_text()))
        groups = tuple(load_group_files(root, f"acquisition_{n}_") for n in ("a", "b"))
        qualified.require_query(groups, model)
    elif "qualification_snapshot.zip" in manifest["files"]:
        raise ObservationError("orphan qualification snapshot has no bound admission")
    records = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    verify_journal(records)
    return {"verified": True, "files": len(manifest["files"]), "arrays": len(manifest["arrays"]),
            "events": len(records), "manifest_content_sha256": digest,
            "limit": manifest["integrity_limit"]}
