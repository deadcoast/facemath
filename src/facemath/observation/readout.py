"""A compact readout of existing evidence, never a second scoring pipeline."""
from __future__ import annotations

from html import escape
from pathlib import Path
import json

from .contracts import ObservationError
from .study import read_json


def pair_readout(evidence: dict) -> dict:
    intrinsic = evidence.get("intrinsic_profile") or {}
    bilateral = intrinsic.get("bilateral_comparison")
    display = evidence.get("bilateral_display")
    if bilateral and display:
        bilateral = {**bilateral, "display_axes": display["axes"], "canvas_size": evidence["configuration"]["canvas_size"],
            "rows": [{**row, "display_segments": [display["segments"][s][row["index"]]
                       if row["observed_"+side] is not None else None for s, side in enumerate(("a", "b"))]}
                     for row in bilateral["rows"]]}
    family_rows = []
    for family in intrinsic.get("families", []):
        fraction = family["within_tolerance_fraction_bounds"]
        family_rows.append({**family, "within_tolerance_percent_bounds": None if fraction is None else [100 * x for x in fraction]})
    admission = evidence.get("empirical_admission") or {"status": "not_requested"}
    if "admission" in admission:
        admission = admission["admission"]
    measured = {m["name"]: m for m in evidence.get("measurement_profile", [])}
    return {"format": "facemath.pair_readout.v10", "recipe_sha256": evidence["recipe_sha256"],
        "scope": "projected geometry, not identity", "empirical_admission": admission,
        "localization_calibrated": evidence.get("localization_envelope") is not None,
        "registered_all_rms": measured.get("geometry.all.rms"),
        "shape_distance": intrinsic.get("shape_distance"), "families": family_rows,
        "structural_comparison": intrinsic.get("structural_comparison"),
        "bilateral_comparison": bilateral,
        "regional_comparison": evidence.get("regional_comparison"),
        "edge_comparison": (evidence.get("edge_diagnostics") or {}).get("paired_comparison"),
        "registration_diagnostics": evidence.get("registration_diagnostics"),
        "percentage_meaning": "fraction of a prespecified, dependent relation inventory within explicit tolerances; NOT a probability of resemblance",
        "array_file": "arrays.npz", "exact_relation_rows": "intrinsic.json", "execution_journal": "events.jsonl",
        "source_measured": evidence.get("measured_acquisition") is not None,
        "warning_count": len(evidence.get("warnings", [])),
        "limits": evidence.get("limits", [])}


def admission_panel(report: dict | None) -> str:
    if not report or report.get("status") in ("not_prespecified", "not_requested"):
        return '<section id="admission"><h2>Empirical acceptance</h2><p>Research analysis. No prespecified held-out acceptance was asserted. A finite calibration radius is not an empirical deployment approval.</p></section>'
    if "admission" in report:
        report = report["admission"]
    status = escape(report["status"].replace("_", " "))
    rows = []
    for row in report.get("categories", []):
        def value(v):
            return "unavailable" if v is None else f"{100*v:.2f}%"
        rows.append(f'<tr><td>{escape(row["category"])}</td><td>{row["groups"]}</td>'
            f'<td>{row["delivered"]} · {value(row["delivery_lower"])}</td>'
            f'<td>{row["covered"]} · {value(row["coverage_lower"])}</td>'
            f'<td>{"pass" if row["passes"] else "withheld"}<small>{escape("; ".join(row["reasons"]))}</small></td></tr>')
    note = "SYNTHETIC CONTROL ONLY" if report.get("policy", {}).get("evidence_kind") == "synthetic_control" else "Declared native reference study; independent acquisition assumptions still apply"
    return f'''<section id="admission"><h2>Empirical acceptance · {status}</h2><div class="panel"><strong>{note}</strong><br>
All requested categories must pass. Thresholds and family error were frozen before acquisition; these percentages describe held-out group delivery/coverage, not facial resemblance.</div>
<div class="scroll"><table><thead><tr><th>Category</th><th>Groups</th><th>Delivered · lower bound</th><th>Covered · lower bound</th><th>Decision</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Independent comparable groups are declared, not proven by their identifiers. No guarantee against hidden shift or repeated cross-study model selection.</p></section>'''


def _value(value: float | None, *, signed: bool = False) -> str:
    return "Unavailable" if value is None else format(value, "+.5g" if signed else ".5g")


def render_structure(data: dict | None) -> str:
    if data is None:
        return ""
    rows = []
    for row in data["rows"]:
        interval = row["reference_delta_b_minus_a"]
        bounds = ("Unavailable" if interval is None else f"[{_value(interval[0], signed=True)}, {_value(interval[1], signed=True)}]")
        reasons = "; ".join(f"{s}: {r['reason'].replace('_', ' ')}" for s, r in zip(("A", "B"), row["source_intervals"]))
        unit = "contrast" if row["operation"] == "asymmetry" else "eye spans"
        status = row["reference_status"].replace("_", " ")
        rows.append(f'<tr><th scope="row">{escape(row["label"])}<small>{unit} · tolerance {_value(row["tolerance"])}</small></th>'
            f'<td>{_value(row["observed_a"])}</td><td>{_value(row["observed_b"])}</td>'
            f'<td>{_value(row["observed_delta_b_minus_a"], signed=True)}<small>{escape(row["observed_status"].replace("_", " "))}</small></td>'
            f'<td>{bounds}<small>{escape(status)}' + (f' · {escape(reasons)}' if interval is None else '') + '</small></td></tr>')
    counts = data["observed_counts"]
    return f'''<section id="structure"><h2>Facial structure · A compared with B</h2>
<p>Observed measurements in each image, normalized by its eye-midpoint span. Positive differences mean B is larger; lateral offsets increase toward viewer-right. Heights follow the eye line's clockwise perpendicular. Pose and expression affect these projected measurements.</p>
<div class="scroll" role="region" aria-label="Structural comparison table" tabindex="0"><table><caption>Fourteen fixed measurements · {counts['within_tolerance']} observed within tolerance · {counts['outside_tolerance']} outside · {counts['unavailable']} unavailable</caption>
<thead><tr><th scope="col">Measurement</th><th scope="col">Observed A</th><th scope="col">Observed B</th><th scope="col">B − A</th><th scope="col">Reference interval for B − A</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Observed differences use the measured coordinates. Reference intervals use the attached error model; an unresolved interval cannot establish agreement. Eye-width asymmetry is (left − right)/(left + right). These dependent measurements do not form an identity or resemblance probability.</p></section>'''


def render_edge_comparison(data: dict | None) -> str:
    if data is None:
        return ""
    rows = []
    for row in data["rows"]:
        stats = row["eye_spans"] or {}
        lo, hi = row["within_tolerance_fraction_range"]
        decisions = (escape(row["reason"].replace("_", " ")) if row["rejections"] is None else
            "".join(f'<small>{side.upper()}: ' + escape("; ".join(
                f'{reason.replace("_", " ")} {count}' for reason, count in counts.items() if count)) + '</small>'
                for side, counts in row["rejections"].items()))
        rows.append(f'<tr><th scope="row">{escape(row["region"].replace("_", " ").capitalize())}'
            f'<small>{row["anchor"].upper()}-anchored rays</small></th>'
            f'<td>{row["supported"]} / {row["total"]}<small>{row["unresolved"]} unresolved</small></td>'
            f'<td>{_value(stats.get("mean_signed_offset"), signed=True)}</td>'
            f'<td>{_value(stats.get("rms_offset"))}</td><td>{_value(stats.get("p90_absolute_offset"))}</td>'
            f'<td>{100*lo:.1f}–{100*hi:.1f}%</td><td>{decisions}</td></tr>')
    return f'''<section id="edge-comparison"><h2>Image-edge displacement · A compared with B</h2>
<p>Both images are sampled on the same rays around each contour. Values below are B minus A along the displayed anchor's normal, in registered eye spans. RMS and the 90th percentile retain mismatch that signed means can cancel.</p>
<div class="scroll" role="region" aria-label="Image-edge comparison table" tabindex="0"><table>
<caption>Four contour regions, sampled separately from A and B · tolerance {_value(data['tolerance_eye_spans'])} eye spans · no pooled score</caption>
<thead><tr><th scope="col">Contour / ray anchor</th><th scope="col">Paired support</th><th scope="col">Signed mean</th><th scope="col">RMS</th><th scope="col">90th absolute</th><th scope="col">Within-tolerance range</th><th scope="col">Ray decisions</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>The range keeps every requested ray: rejected and missing rays remain unresolved. It is a missingness range, not a confidence interval or resemblance probability. Each side records its first failed rule. Weak, ambiguous or out-of-window edges cannot establish agreement. These gradients may describe texture rather than anatomical boundaries; tangential motion is not measured.</p></section>'''


def render_bilateral(data: dict | None, *, image_prefix: str = "") -> str:
    if data is None:
        return ""
    def vector(value):
        return "Unavailable" if value is None else f"({_value(value[0], signed=True)}, {_value(value[1], signed=True)})"
    def interval(value):
        return "Unavailable" if value is None else f"[{_value(value[0])}, {_value(value[1])}]"
    rows = []
    locations = []
    for row in data["rows"]:
        rows.append(f'<tr><th scope="row">{escape(row["label"])}</th>'
            f'<td>{vector(row["observed_a"])}<small>magnitude {_value(row["observed_magnitude_a"])}</small></td>'
            f'<td>{vector(row["observed_b"])}<small>magnitude {_value(row["observed_magnitude_b"])}</small></td>'
            f'<td>{vector(row["observed_delta_b_minus_a"])}<small>change {_value(row["observed_configuration_change"])} · {escape(row["observed_status"].replace("_", " "))}</small></td>'
            f'<td>{interval(row["reference_configuration_change"])}<small>{escape(row["reference_status"].replace("_", " "))}</small></td></tr>')
        views = []
        for s, side in enumerate(("a", "b")):
            segment = row["display_segments"][s]
            if segment is None:
                views.append(f'<p>{side.upper()}: measurement unavailable because a required point is missing.</p>')
                continue
            expected, actual = segment
            start, end = data["display_axes"][s]
            n = data["canvas_size"]
            label = escape(row["label"] + " on image " + side.upper())
            href = escape(image_prefix + "canonical_" + side + ".png", quote=True)
            views.append(f'''<figure><svg role="img" aria-label="{label}" viewBox="-.5 -.5 {n} {n}">
<image href="{href}" x="-.5" y="-.5" width="{n}" height="{n}"/>
<g fill="none" stroke-width="1.2"><path stroke="#fff" stroke-dasharray="3 3" opacity=".7" d="M {start[0]} {start[1]} L {end[0]} {end[1]}"/>
<path stroke="#ffca80" d="M {expected[0]} {expected[1]} L {actual[0]} {actual[1]}"/>
<circle stroke="#fff" cx="{expected[0]}" cy="{expected[1]}" r="3"/>
<path stroke="#79d7d0" d="M {actual[0]-3} {actual[1]} h 6 M {actual[0]} {actual[1]-3} v 6"/></g></svg>
<figcaption>{side.upper()} · {vector(row['observed_'+side])} eye spans</figcaption></figure>''')
        sources = "; ".join(f"{side}: {source['reason'].replace('_', ' ')}" for side, source in zip(("A", "B"), row["source_intervals"]))
        component_bounds = row["reference_delta_b_minus_a"]
        bounds = "Unavailable" if component_bounds is None else f"horizontal {interval(component_bounds[0])}; vertical {interval(component_bounds[1])}"
        locations.append(f'<details data-bilateral="{escape(row["key"])}"><summary>Locate {escape(row["label"].lower())}</summary>'
            f'<div class="gallery bilateral-gallery">{"".join(views)}</div>'
            f'<p>Reference component changes: {bounds}. {escape(sources)}.</p></details>')
    counts = data["observed_counts"]
    return f'''<section id="bilateral"><h2>Bilateral configuration</h2>
<p>Compare horizontal and vertical imbalance around each image's eye-defined midline. Eye and mouth rows measure the right corner minus the mirrored left corner. Nose and chin rows measure signed midline offset. All values use each image's eye span.</p>
<div class="scroll" role="region" aria-label="Bilateral comparison table" tabindex="0"><table><caption>Five fixed rows · {counts['within_tolerance']} observed changes within tolerance · {counts['outside_tolerance']} outside · {counts['unavailable']} unavailable · tolerance {_value(data['tolerance'])}</caption>
<thead><tr><th scope="col">Feature</th><th scope="col">A · horizontal, vertical</th><th scope="col">B · horizontal, vertical</th><th scope="col">B − A · components</th><th scope="col">Reference change range</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Change is the length of the component-difference vector. Zero change means the same imbalance, even when both images are asymmetric. These dependent projected measurements vary with pose and expression.</p>
<p>Expand a feature to locate it. White circle: mirrored left corner or projected midline point. Cyan cross: observed right corner or central point. Amber segment: measured difference. The dashed midline is defined by the eyes; no new alignment is fitted.</p>
{''.join(locations)}</section>'''


def render_pair_readout(evidence: dict, *, image_prefix: str = "") -> str:
    data = pair_readout(evidence)
    shape = data["shape_distance"]
    shape_text = "Unavailable: no intrinsic shape profile"
    if shape:
        nominal = shape.get("observed_rms")
        shape_text = "Observed proper-rotation shape RMS: " + ("unavailable" if nominal is None else f"{nominal:.6g}")
        if shape.get("available"):
            shape_text += f' · reference-shape range [{shape["lower"]:.6g}, {shape["upper"]:.6g}]'
        else:
            shape_text += " · reference-shape range unavailable"
    rows = []
    for family in data["families"]:
        bounds = family["within_tolerance_percent_bounds"]
        text = "Unavailable — uncalibrated" if bounds is None else f"{bounds[0]:.2f}–{bounds[1]:.2f}%"
        counts = ["unavailable" if family[k] is None else str(family[k]) for k in
                  ("prespecified_relations", "within", "outside", "unresolved_or_unavailable")]
        cells = "".join(f"<td>{v}</td>" for v in counts)
        rows.append(f'<tr><td>{escape(family["family"])}</td>{cells}<td>{text}</td></tr>')
    rows = "".join(rows)
    return f'''{render_structure(data["structural_comparison"])}{render_bilateral(data["bilateral_comparison"], image_prefix=image_prefix)}{render_edge_comparison(data["edge_comparison"])}<section id="readout"><h2>Comparison at a glance</h2><p>{escape(shape_text)}</p>
<div class="scroll"><table><thead><tr><th>Relations</th><th>Declared total</th><th>Within</th><th>Outside</th><th>Unresolved</th><th>Within-tolerance range</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>{escape(data["percentage_meaning"])}. Missing relations remain in the denominator. Detailed rows and exact values remain below and in intrinsic.json.</p>
{admission_panel(evidence.get("empirical_admission"))}</section>'''


def inspect_readout(root: str | Path) -> dict:
    root = Path(root)
    kind = read_json(root / "manifest.json")["format"]
    if kind == "facemath.qualification_bundle.v5":
        from .study_io import verify_qualification_bundle
        verified = verify_qualification_bundle(root)
        report = read_json(root / "qualification.json")
        return {"kind": "observer_study", "integrity": verified, "enrollment": report["enrollment"],
            "empirical_admission": report.get("empirical_admission"),
            "scope": report["execution_kind"], "report": str(root / "report.html"),
            "limit": "overview verifies consistency; use replay for full numerical recomputation"}
    from .artifacts import verify_bundle
    verification = verify_bundle(root)
    return {"kind": "pair", "integrity": verification, **pair_readout(read_json(root / "evidence.json")),
            "report": str(root / "report.html")}


def terminal_readout(data: dict, *, encoding: str = "utf-8") -> str:
    lines = ["FACEMATH ◆ EVIDENCE", "│"]
    admission = data.get("empirical_admission") or {}
    lines.append(f"├── Empirical acceptance: {admission.get('status', 'not_requested')}")
    if data.get("kind") == "observer_study":
        e = data["enrollment"]
        lines.append(f"├── Acquisition: {e['committed']}/{e['eligible']} committed; {e['prospectively_excluded']} excluded")
        for row in admission.get("categories", []):
            lines.append(f"│   {row['category']}: {row['covered']}/{row['groups']} covered; " + ("pass" if row['passes'] else "WITHHELD"))
    else:
        bilateral = data.get("bilateral_comparison")
        if bilateral:
            lines.append("├── Bilateral configuration (horizontal, vertical; own eye spans)")
            for row in bilateral["rows"]:
                def components(value):
                    return "unavailable" if value is None else f"({_value(value[0], signed=True)}, {_value(value[1], signed=True)})"
                lines.append(f"│   {row['label']}: A={components(row['observed_a'])} B={components(row['observed_b'])}; "
                    f"delta={components(row['observed_delta_b_minus_a'])}; change={_value(row['observed_configuration_change'])}")
        edge_comparison = data.get("edge_comparison")
        if edge_comparison:
            lines.append("├── Image-edge displacement (B minus A along shared normals; eye spans)")
            for row in edge_comparison["rows"]:
                stats = row["eye_spans"] or {}
                lines.append(f"│   {row['region']} [{row['anchor'].upper()} rays]: signed={_value(stats.get('mean_signed_offset'), signed=True)}; "
                    f"RMS={_value(stats.get('rms_offset'))}; p90={_value(stats.get('p90_absolute_offset'))}; "
                    f"paired={row['supported']}/{row['total']}; unresolved={row['unresolved']}")
            lines.append("│   Image-gradient evidence; missing rays are unresolved, not anatomical agreement.")
        regional = data.get("regional_comparison")
        if regional:
            lines.append("├── Regional movement and deformation (observed geometry)")
            for row in regional["rows"]:
                movement, deformation = row["displacement"], row["deformation"]
                shift = movement["shift_b_minus_a"]
                shift_text = "unavailable" if shift is None else f"({_value(shift[0], signed=True)}, {_value(shift[1], signed=True)})"
                lines.append(f"│   {row['region']}: shift={shift_text}; centered RMS={_value(movement['centered_rms'])}; "
                    f"area ratio={_value(deformation['area_ratio_mean'])}; anisotropic stretch={_value(deformation['anisotropic_log_strain_rms'])}; "
                    f"landmarks={movement['supported']}/{movement['total']}, pixels={deformation['supported_pixels']}/{deformation['total_pixels']}")
        structure = data.get("structural_comparison")
        if structure:
            lines.append("├── Facial structure (B minus A; own eye spans unless marked contrast)")
            for row in structure["rows"]:
                interval = row["reference_delta_b_minus_a"]
                bounds = "unavailable" if interval is None else f"[{_value(interval[0], signed=True)}, {_value(interval[1], signed=True)}]"
                unit = " contrast" if row["operation"] == "asymmetry" else ""
                lines.append(f"│   {row['label']}: A={_value(row['observed_a'])} B={_value(row['observed_b'])} "
                    f"delta={_value(row['observed_delta_b_minus_a'], signed=True)}{unit}; reference delta {bounds} ({row['reference_status']})")
        shape = data.get("shape_distance") or {}
        n = shape.get("observed_rms")
        lines.append("├── Observed shape RMS: " + ("unavailable" if n is None else f"{n:.7g} eye spans"))
        if shape.get("available"):
            lines.append(f"├── Reference-shape bounds: [{shape['lower']:.7g}, {shape['upper']:.7g}]")
        for f in data["families"]:
            bounds = f["within_tolerance_percent_bounds"]
            if bounds is None:
                lines.append(f"│   {f['family']:<20} bounds unavailable; uncalibrated observations")
            else:
                lo, hi = bounds
                lines.append(f"│   {f['family']:<20} {lo:6.2f}–{hi:6.2f}% within tolerance; {f['unresolved_or_unavailable']} unresolved")
        lines.append("│   Fractions are dependent relation counts, not resemblance probabilities.")
    lines.append(f"└── Report: {data['report']}")
    text = "\n".join(lines)
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        # Keep values intact in redirected Windows/code-page streams. Replace
        # only decoration; escape unrepresentable source labels/paths reversibly.
        # Convert generated prefixes only: a source path may contain these glyphs.
        prefixes = {"FACEMATH ◆ EVIDENCE": "FACEMATH * EVIDENCE", "├──": "+--", "└──": "\\--", "│": "|"}
        ascii_lines = []
        for line in lines:
            for prefix, replacement in prefixes.items():
                if line.startswith(prefix):
                    line = replacement + line[len(prefix):]
                    break
            ascii_lines.append(line)
        text = "\n".join(ascii_lines)
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    return text
