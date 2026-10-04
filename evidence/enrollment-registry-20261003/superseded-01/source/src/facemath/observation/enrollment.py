"""Immutable full-enrollment receipts and explicitly scoped source reuse checks.

No inference, outcome filtering, identity inference or external attestation.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .contracts import ObservationError, object_digest
from .reference import ReferenceAnnotation, check_digest
from .study import load_study, read_json, write_json_new

RECEIPT_FORMAT = "facemath.enrollment_receipt.v16"
REGISTRY_FORMAT = "facemath.enrollment_registry.v16"
CHECK_FORMAT = "facemath.enrollment_check.v16"
ALIAS_POLICY = "exact_source_hashes_and_shared_group_lineage_ids.v1"
STUDY_FORMATS = {"facemath.prospective_study.v5": "facemath.study_lock_manifest.v5",
                 "facemath.prospective_study.v16": "facemath.study_lock_manifest.v16"}
LIMIT = "declared registered enrollment only; not external authenticity, person independence, training-data coverage or empirical admission"


def _seal(value: dict) -> dict:
    return {**value, "content_sha256": object_digest(value)}


def _require_seal(value: dict, format: str, fields: set[str]) -> None:
    if (not isinstance(value, dict) or set(value) != fields | {"format", "content_sha256"}
            or value.get("format") != format
            or object_digest({k:v for k,v in value.items() if k != "content_sha256"}) != value.get("content_sha256")):
        raise ObservationError("invalid enrollment artifact fields or digest")


def _bound_json(receipt: dict, text: str, name: str) -> dict:
    if not isinstance(text, str):
        raise ObservationError("enrollment artifact must retain exact JSON bytes")
    data = text.encode("utf-8")
    files = receipt["manifest"].get("files")
    if not isinstance(files, dict):
        raise ObservationError("enrollment manifest requires a file inventory")
    pin = files.get(name, {})
    if pin != {"bytes": len(data), "sha256": sha256(data).hexdigest()}:
        raise ObservationError("enrollment artifact binding mismatch: " + name)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ObservationError("enrollment artifact JSON must be an object")
    return value


def enrollment_row(case: dict, refs: tuple[ReferenceAnnotation, ...]) -> dict:
    from .transfer import _image_tokens
    if (any(not isinstance(case.get(k), str) or not case[k].strip() for k in ("case_id", "group_id"))
            or case.get("split") not in ("development", "calibration", "evaluation")
            or case.get("eligibility") not in ("included", "excluded")
            or not isinstance(case.get("lineage_ids"), list)
            or any(not isinstance(v, str) or not v.strip() for v in case["lineage_ids"])
            or len(case["lineage_ids"]) != len(set(case["lineage_ids"]))):
        raise ObservationError("invalid full enrollment case")
    for key in ("image_sha256", "original_encoded_sha256"):
        check_digest(case.get(key), key)
    for ref in refs:
        if ref.image_sha256 != case["image_sha256"] or list(ref.image_shape) != case["image_shape"]:
            raise ObservationError("enrollment reference/image mismatch")
    return {"case_id": case["case_id"], "split": case["split"], "eligibility": case["eligibility"],
            "group_id": case["group_id"], "lineage_ids": list(case["lineage_ids"]),
            "image_tokens": sorted(_image_tokens(case, refs)),
            "identity_tokens": sorted({case["group_id"], *case["lineage_ids"]})}


def load_receipt(receipt: dict) -> list[dict]:
    _require_seal(receipt, RECEIPT_FORMAT, {"manifest", "study_json", "references_json"})
    manifest = receipt["manifest"]
    if (not isinstance(manifest, dict)
            or object_digest({k:v for k,v in manifest.items() if k != "content_sha256"}) != manifest.get("content_sha256")):
        raise ObservationError("enrollment study manifest digest mismatch")
    study = _bound_json(receipt, receipt["study_json"], "study.lock.json")
    if (study.get("format") not in STUDY_FORMATS
            or manifest.get("format") != STUDY_FORMATS[study["format"]]
            or object_digest(study) != manifest.get("study_sha256")):
        raise ObservationError("enrollment study manifest binding mismatch")
    cases = study.get("cases")
    if (not isinstance(cases, list) or not cases or any(not isinstance(c, dict) for c in cases)
            or len({c["case_id"] for c in cases}) != len(cases)):
        raise ObservationError("enrollment receipt requires unique complete cases")
    expected = {name for case in cases for name in case["references"]}
    if not isinstance(receipt["references_json"], dict) or set(receipt["references_json"]) != expected:
        raise ObservationError("enrollment receipt lost complete reference enrollment")
    refs = {name: ReferenceAnnotation.from_dict(_bound_json(receipt, text, name))
            for name, text in receipt["references_json"].items()}
    return [{"study_sha256": manifest["study_sha256"],
             **enrollment_row(case, tuple(refs[n] for n in case["references"]))} for case in cases]


def export_enrollment(study_path: str | Path, destination: str | Path) -> dict:
    """Export all cases from a verified lock, independent of session outcomes."""
    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    study = load_study(study_path)
    root = study.root
    receipt = _seal({"format": RECEIPT_FORMAT, "manifest": read_json(root / "manifest.json"),
        "study_json": (root / "study.lock.json").read_bytes().decode("utf-8"),
        "references_json": {n: (root / n).read_bytes().decode("utf-8")
                            for c in study.cases for n in c["references"]}})
    rows = load_receipt(receipt)
    write_json_new(target, receipt)
    return {"receipt": str(target), "study_sha256": study.digest, "enrolled": len(rows),
            "content_sha256": receipt["content_sha256"], "inference_executed": False, "limit": LIMIT}


def load_registry(registry: dict) -> list[dict]:
    """Check every retained ancestor; an extension cannot remove older entries."""
    if not isinstance(registry, dict):
        raise ObservationError("enrollment registry must be a snapshot object")
    cursor, scope, rows, seen, depth = registry, None, [], set(), 0
    while cursor is not None:
        depth += 1
        if depth > 32:
            raise ObservationError("enrollment registry exceeds 32 snapshot generations")
        _require_seal(cursor, REGISTRY_FORMAT, {"scope", "alias_policy", "prior", "entries"})
        if (not isinstance(cursor["scope"], str) or not cursor["scope"].strip()
                or cursor["alias_policy"] != ALIAS_POLICY):
            raise ObservationError("registry requires an explicit scope and fixed alias policy")
        if scope is not None and cursor["scope"] != scope:
            raise ObservationError("registry extension cannot change the prior scope")
        scope = cursor["scope"]
        entries = cursor["entries"]
        if not isinstance(entries, list) or not entries:
            raise ObservationError("registry requires new full enrollment receipts")
        digests = []
        for entry in entries:
            enrolled = load_receipt(entry)
            digest = entry["manifest"]["study_sha256"]
            if digest in seen:
                raise ObservationError("duplicate registered study")
            seen.add(digest); digests.append(digest); rows.extend(enrolled)
        if digests != sorted(digests):
            raise ObservationError("registry entries must be ordered by study digest")
        cursor = cursor["prior"]
    return sorted(rows, key=lambda r: (r["study_sha256"], r["case_id"]))


def build_registry(receipts: list[dict], scope: str, *, prior: dict | None = None) -> dict:
    for receipt in receipts:
        load_receipt(receipt)
    registry = _seal({"format": REGISTRY_FORMAT, "scope": scope, "alias_policy": ALIAS_POLICY,
        "prior": prior, "entries": sorted(receipts, key=lambda r:r["manifest"]["study_sha256"])})
    load_registry(registry)
    return registry


def check_enrollment(registry: dict, cases_and_references: list | tuple) -> dict:
    prior = load_registry(registry)
    cases = [enrollment_row(case, refs) for case, refs in cases_and_references]
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ObservationError("registry check requires unique complete target enrollment")
    # Index all prior records; repeated enrollment remains attributable to each study.
    indices = {"image_tokens": {}, "identity_tokens": {}}
    for i, row in enumerate(prior):
        for kind, index in indices.items():
            for token in row[kind]:
                index.setdefault(token, set()).add(i)
    overlaps = []
    for case in cases:
        hits = set()
        for kind, index in indices.items():
            for token in case[kind]:
                hits.update(index.get(token, ()))
        if hits:
            overlaps.append({"case_id": case["case_id"], "split": case["split"],
                "eligibility": case["eligibility"], "matches": [{
                    **{k:prior[i][k] for k in ("study_sha256", "case_id", "split", "eligibility")},
                    "image_aliases": sorted(set(case["image_tokens"]) & set(prior[i]["image_tokens"])),
                    "identity_aliases": sorted(set(case["identity_tokens"]) & set(prior[i]["identity_tokens"]))}
                    for i in sorted(hits)]})
    return {"format": CHECK_FORMAT, "registry_sha256": registry["content_sha256"],
        "scope": registry["scope"], "alias_policy": ALIAS_POLICY,
        "registered_studies": len({r["study_sha256"] for r in prior}), "registered_cases": len(prior),
        "target_enrollment_sha256": object_digest(cases), "enrolled": len(cases),
        "eligible": sum(c["eligibility"] == "included" for c in cases),
        "prospectively_excluded": sum(c["eligibility"] == "excluded" for c in cases),
        "splits": {s:sum(c["split"] == s for c in cases) for s in ("development", "calibration", "evaluation")},
        "disjoint": not overlaps, "overlapping_cases": len(overlaps), "overlaps": overlaps, "limit": LIMIT}


def require_disjoint(registry: dict, cases_and_references: list | tuple) -> dict:
    report = check_enrollment(registry, cases_and_references)
    if not report["disjoint"]:
        raise ObservationError(f"registry overlaps full prior enrollment: {report['overlapping_cases']} of "
                               f"{report['enrolled']} enrolled cases; no sources were replaced")
    return report


def require_query_disjoint(registry: dict, groups: tuple) -> None:
    rows = load_registry(registry)
    images = {s for row in rows for s in row["image_tokens"]}
    identities = {s for row in rows for s in row["identity_tokens"]}
    for group in groups:
        if ({group.group_id, *group.lineage_ids} & identities
                or any(source.image_sha256 in images for source in group.sources)):
            raise ObservationError("query/context leaks registered prior enrollment")
