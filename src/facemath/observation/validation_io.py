"""Replayable validation studies using the SAME Observation/provider contracts.

Images and references are captured into a create-only bundle. Failed inference
attempts are recorded before summaries, never filtered for convenient accuracy.
"""
from __future__ import annotations

from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any
import json
import os
import tempfile

import cv2
import numpy as np

from .contracts import Observation, ObservationError, LandmarkProvider, array_digest, canonical_json, object_digest
from .error_model import fit_localization_calibration, evaluate_coverage
from .journal import Journal, environment, verify_journal, code_fingerprint, numeric_environment
from .reference import REFERENCE_NAMES, ReferenceAnnotation
from .validation import ValidationCase, evaluate_study


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _local_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ObservationError("study assets must be explicit relative paths")
    path = root / value
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ObservationError("study path escapes root or is a symlink")
    if not path.is_file():
        raise ObservationError(f"study asset missing: {value}")
    return path


def _image(path: Path) -> np.ndarray:
    # One stored-raster coordinate policy for studies, annotations and queries.
    from .study import load_rgb
    return load_rgb(path)


def _png(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise ObservationError("failed to publish study source image")


def _html(report: dict, coverage: dict | None, domain: str) -> str:
    sections = []
    for split, s in report["splits"].items():
        nme = "withheld" if s["successful_case_mean_nme"] is None else f'{s["successful_case_mean_nme"]:.5f}'
        cards = f'''<div class="cards"><div><small>Attempted / complete</small><strong>{s['attempted_cases']} / {s['complete_cases']}</strong></div><div><small>Declared groups</small><strong>{s['groups']}</strong></div><div><small>Mean NME · complete cases</small><strong>{nme}</strong></div><div><small>PCK ≤ {s['pck_threshold_ipd']:.2f} reference spans</small><strong>{s['pck_including_failures']:.1%}</strong></div></div>'''
        rows = "".join(f"<tr><td><code>{escape(r['name'])}</code></td><td>{r['observed']} / {r['eligible']}</td><td>{r['within_tolerance']} / {r['eligible']}</td><td>{r['median_error_px']:.3f}</td></tr>" if r['median_error_px'] is not None else f"<tr><td>{escape(r['name'])}</td><td>0 / {r['eligible']}</td><td>0 / {r['eligible']}</td><td>withheld</td></tr>" for r in s['per_landmark'])
        sections.append(f"<h2>{escape(split.title())}</h2>{cards}<p>{s['failed_cases']} failed cases remain in the PCK and AUC denominators. Unconditional AUC to {s['auc_threshold_ipd']:.2f} spans: {s['unconditional_auc']:.4f}.</p><div class='scroll'><table><tr><th>Landmark</th><th>Observed / eligible</th><th>Within tolerance / eligible</th><th>Median error · source px</th></tr>{rows}</table></div>")
    cases = "".join(f"<tr><td>{escape(r['case_id'])}</td><td>{escape(r['group_id'])}</td><td>{escape(r['split'])}</td><td>{r['nme']:.5f}</td><td>complete</td></tr>" if r['nme'] is not None else f"<tr><td>{escape(r['case_id'])}</td><td>{escape(r['group_id'])}</td><td>{escape(r['split'])}</td><td>withheld</td><td>{escape(r['failure'] or 'unsupported')}</td></tr>" for r in report['cases'])
    cover = "<p>No localization calibration attached.</p>"
    if coverage:
        actual = "unbounded / not scored" if coverage["empirical_group_coverage"] is None else f"{coverage['empirical_group_coverage']:.1%}"
        cover = f"<div class='notice'><strong>Held-out group coverage: {actual}</strong><br>Nominal {coverage['nominal_coverage']:.1%}; {coverage['evaluation_groups']} declared groups; {coverage['failures']} failed or incompletely labeled cases. This is reference coverage, not a resemblance percentage.</div>"
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FACEMATH / Localization validation</title><style>
:root{{--bg:#10171e;--panel:#1a2731;--ink:#eaf1f5;--muted:#a5b6c2;--line:#344753;--accent:#83ddd1}}*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif;margin:0}}main{{max-width:1250px;margin:auto;padding:35px 24px 70px}}header{{border-bottom:1px solid var(--line);padding-bottom:22px}}.eyebrow{{color:var(--accent);font:11px ui-monospace,monospace;letter-spacing:.16em}}h1{{font-size:34px;letter-spacing:-.035em;margin:10px 0}}h2{{font-size:23px;margin-top:35px}}p,small{{color:var(--muted)}}code{{font:12px ui-monospace,monospace;overflow-wrap:anywhere}}a{{color:var(--accent)}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}.cards>div{{background:var(--panel);padding:15px;border:1px solid var(--line)}}small{{display:block;font-size:12px}}strong{{display:block;font-size:24px;font-variant-numeric:tabular-nums}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}}th{{background:var(--panel)}}.scroll{{overflow:auto;border:1px solid var(--line)}}.notice{{padding:16px;border-left:3px solid var(--accent);background:var(--panel);margin:20px 0}}.notice strong{{font-size:18px}}footer{{margin-top:32px;border-top:1px solid var(--line);padding-top:15px;color:var(--muted);font-size:12px}}@media(max-width:650px){{main{{padding:22px 12px}}.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}h1{{font-size:28px}}}}
</style><main><header><div class="eyebrow">FACEMATH / WAVE 04 / OBSERVATION VALIDATION</div><h1>Repeatable is not necessarily accurate.</h1><p>Acquisition domain: {escape(domain)}. Reference kind: {escape(report['cases'][0]['reference_kind'])}.</p><p>Predictions are compared in their original source-pixel coordinates. No alignment to reference labels is fitted.</p></header>{''.join(sections)}<h2>Localization envelope check</h2>{cover}<h2>Every attempted case</h2><div class="scroll"><table><tr><th>Case</th><th>Group</th><th>Split</th><th>NME</th><th>Status</th></tr>{cases}</table></div><h2>Underlying records</h2><p><a href="validation.json">All reference/prediction coordinates and errors</a> · <a href="arrays.npz">Numerical arrays</a> · <a href="study.lock.json">Replay inputs</a> · <a href="events.jsonl">Actual execution journal</a> · <a href="manifest.json">Byte manifest</a></p><footer>Source-image and declared-lineage leakage are checked. Group identities and annotation blinding are curator declarations, not inferred identities or externally attested facts. Annotation consensus is a reference, not guaranteed anatomical truth.</footer></main></html>'''


def run_validation_study(study_path: str | Path, destination: str | Path, *,
                         provider: LandmarkProvider | None = None, alpha: float = .1) -> Path:
    study_path, destination = Path(study_path).absolute(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    spec, root = _json(study_path), study_path.parent
    if spec.get("format") != "facemath.validation_study.v4":
        raise ObservationError("unsupported validation study format")
    domain = spec.get("domain", "")
    if not isinstance(domain, str) or not domain.strip() or not spec.get("cases"):
        raise ObservationError("study needs a named domain and cases")
    minimum_raters = spec.get("minimum_raters", 1)
    from .error_model import corrected_quantile
    from .reference import consensus
    corrected_quantile(np.zeros(1), alpha)  # validate protocol before invoking a runtime
    # Validate all references/source bindings and split plans BEFORE invoking an ML runtime.
    loaded, preflight = [], []
    for entry in spec["cases"]:
        image_path = _local_path(root, entry["image"])
        image = _image(image_path)
        refs = tuple(ReferenceAnnotation.from_dict(_json(_local_path(root, name))) for name in entry["references"])
        if any(r.image_sha256 != array_digest(image) or r.image_shape != image.shape[:2] for r in refs):
            raise ObservationError("study reference does not match decoded source pixels")
        consensus(refs, minimum_raters).eye_span_px
        preflight.append(ValidationCase(entry["case_id"], entry["group_id"], entry["split"], refs, None,
                                       "preflight_only", tuple(entry.get("lineage_ids", ()))))
        loaded.append((entry, image_path, image, refs))
    from .validation import validate_study
    validate_study(tuple(preflight))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".facemath-validation-", dir=destination.parent))
    journal = Journal()
    try:
        with journal.step("validation.preflight", {"study_file_sha256": sha256(study_path.read_bytes()).hexdigest()}) as output:
            output.update(cases=len(loaded), domain=domain, source_pixel_binding=True, split_leakage_checked=True)
        cases, lock_cases = [], []
        for i, (entry, image_path, image, refs) in enumerate(loaded):
            obs, failure = None, None
            with journal.step("validation.observe_case", {"case_id": entry["case_id"], "image_sha256": array_digest(image)}) as output:
                if entry.get("observation") and entry.get("failure"):
                    raise ObservationError("case cannot supply both observation and failure")
                if entry.get("observation"):
                    obs = Observation.from_dict(_json(_local_path(root, entry["observation"])))
                    obs.require_image(image)
                    output["origin"] = "imported_observation_no_new_inference"
                elif entry.get("failure"):
                    failure = str(entry["failure"])
                    output["origin"] = "imported_failed_attempt_no_new_inference"
                elif provider is not None:
                    output["origin"] = "provider_invocation"
                    try:
                        with journal.step("provider.observe", {"case_id": entry["case_id"]}):
                            obs = provider.observe(image)
                    except ObservationError as exc:
                        failure = str(exc)
                        obs = None
                    if obs is not None:
                        # Wrong image-bound output is a provider contract violation,
                        # not a legitimate nondetection that may be silently pooled.
                        obs.require_image(image)
                else:
                    raise ObservationError("study case needs a saved observation, recorded failure, or concrete provider")
                output.update(observation_sha256=obs.digest if obs else None, failure=failure)
            cases.append(ValidationCase(entry["case_id"], entry["group_id"], entry["split"], refs, obs,
                                        failure, tuple(entry.get("lineage_ids", ()))))
            base = f"case_{i:04d}"
            _png(temp / f"{base}.png", image)
            rpaths = []
            for j, ref in enumerate(refs):
                name = f"{base}_reference_{j:02d}.json"; _write(temp / name, ref.to_dict()); rpaths.append(name)
            lock = {"case_id": entry["case_id"], "group_id": entry["group_id"], "split": entry["split"],
                    "lineage_ids": list(entry.get("lineage_ids", ())), "image": f"{base}.png", "references": rpaths}
            if obs:
                lock["observation"] = f"{base}_observation.json"; _write(temp / lock["observation"], obs.to_dict())
            else:
                lock["failure"] = failure
            lock_cases.append(lock)
        with journal.step("validation.source_coordinate_errors", {"alignment_fitted_to_references": False}) as output:
            report, rows = evaluate_study(tuple(cases), minimum_raters)
            output.update(report_sha256=report["content_sha256"], splits=report["splits"])
        calibration, coverage = None, None
        if any(c.split == "calibration" for c in cases):
            with journal.step("validation.group_conformal_calibration", {"alpha": alpha, "domain": domain}) as output:
                calibration = fit_localization_calibration(rows, study_sha256=report["content_sha256"], alpha=alpha,
                                                          domain=domain, minimum_raters=minimum_raters)
                output.update(calibration_sha256=calibration.digest, finite=calibration.quantile is not None)
            _write(temp / "localization_calibration.json", calibration.to_dict())
        if calibration is not None and any(c.split == "evaluation" for c in cases):
            with journal.step("validation.heldout_coverage", {}) as output:
                coverage = evaluate_coverage(calibration, rows); output.update(coverage)
            _write(temp / "coverage.json", coverage)
        arrays = {k: np.stack([getattr(r, k) for r in rows]) for k in (
            "reference_xy", "predicted_xy", "covariance", "reference_visible", "observed", "error_vectors_px",
            "error_norm_px", "mahalanobis", "rater_count", "rater_disagreement_px")}
        arrays["reference_normalization_span_px"] = np.array([r.normalization_span_px for r in rows])
        np.savez_compressed(temp / "arrays.npz", **arrays)
        _write(temp / "validation.json", report)
        _write(temp / "study.lock.json", {"format": "facemath.validation_study.v4", "domain": domain,
                                         "minimum_raters": minimum_raters, "cases": lock_cases})
        env = environment(); _write(temp / "environment.json", env)
        journal.record("execution.complete", {"report_sha256": report["content_sha256"], "environment": env})
        (temp / "events.jsonl").write_text(journal.jsonl(), encoding="utf-8")
        (temp / "report.html").write_text(_html(report, coverage, domain), encoding="utf-8")
        manifest = {"format": "facemath.validation_bundle.v4", "alpha": alpha,
            "files": {p.name: {"sha256": sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(temp.iterdir())},
            "arrays": {k: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": array_digest(a)} for k, a in arrays.items()},
            "array_semantics": "axis 0 follows validation.cases; point axis follows reference8 names; zeros outside observed masks are missing sentinels",
            "limit": "self-consistency and reproducibility, not external attestation of labels or execution"}
        manifest["manifest_content_sha256"] = object_digest(manifest); _write(temp / "manifest.json", manifest)
        verify_validation_bundle(temp)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temp, destination)
    except Exception as exc:
        from .journal import retain_failed_run
        retain_failed_run(temp, destination, journal, exc)
        raise
    return destination


def verify_validation_bundle(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve(); manifest = _json(root / "manifest.json")
    digest = manifest.pop("manifest_content_sha256")
    if manifest.get("format") != "facemath.validation_bundle.v4" or object_digest(manifest) != digest:
        raise ObservationError("validation manifest mismatch")
    # Unlisted files could masquerade as a second source of truth; reject extras.
    if {p.name for p in root.iterdir()} != set(manifest["files"]) | {"manifest.json"}:
        raise ObservationError("validation bundle has missing/unlisted artifacts")
    for name, spec in manifest["files"].items():
        if Path(name).name != name:
            raise ObservationError("invalid validation manifest path")
        target = _local_path(root, name); content = target.read_bytes()
        if len(content) != spec["bytes"] or sha256(content).hexdigest() != spec["sha256"]:
            raise ObservationError(f"validation artifact mismatch: {name}")
    with np.load(root / "arrays.npz", allow_pickle=False) as data:
        if set(data.files) != set(manifest["arrays"]):
            raise ObservationError("validation array inventory mismatch")
        for name, spec in manifest["arrays"].items():
            a = data[name]
            if a.dtype.str != spec["dtype"] or list(a.shape) != spec["shape"] or array_digest(a) != spec["sha256"]:
                raise ObservationError(f"validation array mismatch: {name}")
    verify_journal([json.loads(x) for x in (root / "events.jsonl").read_text().splitlines()])
    report = _json(root / "validation.json"); expected = report.pop("content_sha256")
    if object_digest(report) != expected:
        raise ObservationError("validation report content digest mismatch")
    return {"verified": True, "kind": "localization_validation", "files": len(manifest["files"]),
            "arrays": len(manifest["arrays"]), "manifest_content_sha256": digest}


def replay_validation_bundle(path: str | Path, destination: str | Path) -> dict[str, Any]:
    root = Path(path); verify_validation_bundle(root)
    env = _json(root / "environment.json")
    if env["engine_code_sha256"] != code_fingerprint() or env["numeric_environment"] != numeric_environment():
        raise ObservationError("validation replay requires matching code and numeric runtime")
    alpha = _json(root / "manifest.json")["alpha"]
    target = run_validation_study(root / "study.lock.json", destination, alpha=alpha)
    previous, actual = _json(root / "validation.json"), _json(target / "validation.json")
    if previous != actual:
        raise ObservationError("validation numerical replay differs")
    for name in ("localization_calibration.json", "coverage.json"):
        if (root / name).exists() and _json(root/name) != _json(target/name):
            raise ObservationError(f"validation replay differs: {name}")
    return {"replay_verified": True, "bundle": str(target), "comparison": "exact localization report/calibration/coverage equality"}
