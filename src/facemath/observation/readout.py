"""A compact readout of existing evidence, never a second scoring pipeline."""
from __future__ import annotations

from html import escape
from pathlib import Path
import json

from .contracts import ObservationError
from .study import read_json


def pair_readout(evidence: dict) -> dict:
    intrinsic = evidence.get("intrinsic_profile") or {}
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


def render_pair_readout(evidence: dict) -> str:
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
    return f'''<section id="readout"><h2>Comparison at a glance</h2><p>{escape(shape_text)}</p>
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


def terminal_readout(data: dict) -> str:
    lines = ["FACEMATH ◆ EVIDENCE", "│"]
    admission = data.get("empirical_admission") or {}
    lines.append(f"├── Empirical acceptance: {admission.get('status', 'not_requested')}")
    if data.get("kind") == "observer_study":
        e = data["enrollment"]
        lines.append(f"├── Acquisition: {e['committed']}/{e['eligible']} committed; {e['prospectively_excluded']} excluded")
        for row in admission.get("categories", []):
            lines.append(f"│   {row['category']}: {row['covered']}/{row['groups']} covered; " + ("pass" if row['passes'] else "WITHHELD"))
    else:
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
    return "\n".join(lines)
