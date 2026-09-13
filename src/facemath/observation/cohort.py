"""Per-metric reference cohorts with exact recipe matching and leakage checks.

No mean of percentiles is presented as a percentile of a joint statistic.
Synthetic baselines remain explicitly synthetic. Reusing a face across pairs
invalidates IID bootstrap, so connected source-image components are resampled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping
from types import MappingProxyType
import numpy as np
from scipy.stats import spearmanr

from .contracts import ObservationError, object_digest


@dataclass(frozen=True)
class CohortRow:
    pair_id: str
    image_a_sha256: str
    image_b_sha256: str
    values: Mapping[str, float]
    # Optional declared subject grouping prevents same-person multi-image leakage.
    subject_a: str | None = None
    subject_b: str | None = None
    source_manifest_sha256: str | None = None


    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ObservationError("cohort pair ID must be nonempty")
        for digest in (self.image_a_sha256, self.image_b_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ObservationError("cohort image provenance requires lowercase SHA-256")
        if self.source_manifest_sha256 is not None:
            digest = self.source_manifest_sha256
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ObservationError("invalid source bundle manifest SHA-256")
        values = {str(k): float(v) for k, v in self.values.items()}
        if not values or not all(np.isfinite(v) for v in values.values()):
            raise ObservationError("cohort values must be finite and nonempty")
        object.__setattr__(self, "values", MappingProxyType(values))

    def to_dict(self) -> dict:
        return {"pair_id": self.pair_id, "image_a_sha256": self.image_a_sha256,
                "image_b_sha256": self.image_b_sha256, "values": dict(self.values),
                "subject_a": self.subject_a, "subject_b": self.subject_b,
                "source_manifest_sha256": self.source_manifest_sha256}


@dataclass(frozen=True, init=False)
class Cohort:
    name: str
    recipe_sha256: str
    rows: tuple[CohortRow, ...]
    kind: str
    fingerprint: str

    def __init__(self, name: str, recipe_sha256: str, rows: list[CohortRow],
                 kind: Literal["synthetic", "observational"] = "observational") -> None:
        if not name or kind not in ("synthetic", "observational") or len(rows) < 3:
            raise ObservationError("named cohort requires >=3 rows and a declared kind")
        if len(recipe_sha256) != 64 or any(c not in "0123456789abcdef" for c in recipe_sha256):
            raise ObservationError("cohort recipe must be lowercase SHA-256")
        if len({r.pair_id for r in rows}) != len(rows):
            raise ObservationError("duplicate pair IDs")
        pairs = [tuple(sorted((r.image_a_sha256, r.image_b_sha256))) for r in rows]
        if len(set(pairs)) != len(pairs):
            raise ObservationError("duplicate unordered image pairs")
        for r in rows:
            if r.image_a_sha256 == r.image_b_sha256 or not all(np.isfinite(v) for v in r.values.values()):
                raise ObservationError("invalid self-pair or nonfinite metric")
        for key, value in (("name", name), ("recipe_sha256", recipe_sha256), ("rows", tuple(rows)), ("kind", kind)):
            object.__setattr__(self, key, value)
        object.__setattr__(self, "fingerprint", object_digest(self.to_dict()))

    def to_dict(self) -> dict:
        return {"format": "facemath.cohort.v3", "name": self.name, "recipe_sha256": self.recipe_sha256,
                "kind": self.kind, "rows": [r.to_dict() for r in self.rows]}

    @classmethod
    def from_dict(cls, obj: dict) -> Cohort:
        if obj.get("format") != "facemath.cohort.v3":
            raise ObservationError("unknown cohort format")
        return cls(obj["name"], obj["recipe_sha256"], [CohortRow(**r) for r in obj["rows"]], obj["kind"])

    def _clusters(self) -> list[list[int]]:
        parent = list(range(len(self.rows)))
        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        owner = {}
        for i, row in enumerate(self.rows):
            tokens = ["image:" + row.image_a_sha256, "image:" + row.image_b_sha256]
            tokens.extend("subject:" + s for s in (row.subject_a, row.subject_b) if s is not None)
            for key in tokens:
                if key in owner:
                    parent[root(i)] = root(owner[key])
                owner[key] = i
        groups = {}
        for i in range(len(parent)):
            groups.setdefault(root(i), []).append(i)
        return list(groups.values())

    def calibrate(self, metric: str, value: float, recipe_sha256: str,
                  query_images: tuple[str, str], *, lower_is_closer: bool = True,
                  query_subjects: tuple[str, ...] = (), bootstrap_draws: int = 500, seed: int = 0) -> dict:
        if recipe_sha256 != self.recipe_sha256:
            raise ObservationError("cohort recipe differs from query; recalibration required")
        if not np.isfinite(value) or bootstrap_draws < 10:
            raise ObservationError("finite value and >=10 bootstrap draws required")
        for row in self.rows:
            if set(query_images) & {row.image_a_sha256, row.image_b_sha256}:
                raise ObservationError("query image leaks into calibration cohort")
            if set(query_subjects) & {row.subject_a, row.subject_b}:
                raise ObservationError("query subject leaks into calibration cohort")
        if any(metric not in row.values for row in self.rows):
            raise ObservationError("metric missing in cohort; no silent row filtering")
        values = np.asarray([row.values[metric] for row in self.rows], float)
        wins = (values > value).astype(float) if lower_is_closer else (values < value).astype(float)
        wins += .5 * (values == value)
        groups = self._clusters()
        interval = None
        if len(groups) >= 3:
            rng = np.random.default_rng(seed)
            draws = [float(wins[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean() * 100)
                     for _ in range(bootstrap_draws)]
            interval = np.quantile(draws, [.025, .975]).tolist()
        return {"metric": metric, "query_value": value, "recipe_sha256": recipe_sha256,
                "lower_is_closer": lower_is_closer, "query_images": list(query_images),
                "query_subjects": list(query_subjects), "bootstrap_draws": bootstrap_draws, "seed": seed,
                "percentile": float(wins.mean() * 100), "interval": interval,
                "interval_kind": "cluster_bootstrap_percentile_95" if interval is not None else None,
                "interval_unavailable_reason": "fewer_than_three_independent_components" if interval is None else None,
                "cohort_name": self.name, "cohort_kind": self.kind, "cohort_sha256": self.fingerprint,
                "independent_components": len(groups), "row_count": len(values),
                "interpretation": "reference-cohort rank; not probability of identity or universal facial similarity"}

    def dependence(self) -> dict:
        metrics = sorted(set.intersection(*(set(r.values) for r in self.rows)))
        matrix: list[list[float | None]] = []
        for a in metrics:
            row = []
            x = np.array([r.values[a] for r in self.rows])
            for b in metrics:
                y = np.array([r.values[b] for r in self.rows])
                if np.std(x) == 0 or np.std(y) == 0:
                    row.append(None)
                else:
                    row.append(float(spearmanr(x, y).statistic))
            matrix.append(row)
        return {"metrics": metrics, "spearman": matrix,
                "warning": "correlated metrics do not provide independent confirmations; no fused percentile is computed"}


def cohort_from_bundles(name: str, bundle_paths: list[str], *, kind: str,
                        metrics: tuple[str, ...] = ("geometry.all.rms",),
                        subjects_by_image: Mapping[str, str] | None = None) -> Cohort:
    """Ingest verified instrument runs without inventing or recomputing measurements.

    Exact shared recipes are required. Missing metrics reject the ingestion,
    rather than dropping inconvenient rows. Synthetic fixtures cannot silently
    become an observational population. Subject identifiers, when available, are
    caller-supplied provenance, never inferred from facial identity.
    """
    from pathlib import Path
    import json
    from .artifacts import verify_bundle

    if len(bundle_paths) < 3 or not metrics or len(set(metrics)) != len(metrics):
        raise ObservationError("cohort ingestion needs >=3 bundles and unique explicit metrics")
    subjects = subjects_by_image or {}
    rows: list[CohortRow] = []
    recipe = None
    for path in bundle_paths:
        root = Path(path)
        verified = verify_bundle(root)
        evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
        observations = json.loads((root / "observations.json").read_text(encoding="utf-8"))
        if recipe is None:
            recipe = evidence["recipe_sha256"]
        elif recipe != evidence["recipe_sha256"]:
            raise ObservationError("cohort bundles have incompatible recipes; do not pool them")
        for observation in observations.values():
            if "synthetic" in observation["provider"] and kind != "synthetic":
                raise ObservationError("synthetic observations require an explicitly synthetic cohort")
        available = {m["name"]: m["value"] for m in evidence["measurement_profile"]}
        if any(available.get(metric) is None for metric in metrics):
            raise ObservationError(f"missing/withheld requested cohort metric in {root}")
        a, b = (observations[key]["image_sha256"] for key in ("a", "b"))
        rows.append(CohortRow(
            pair_id=object_digest(sorted((a, b))),
            image_a_sha256=a,
            image_b_sha256=b,
            values={metric: available[metric] for metric in metrics},
            subject_a=subjects.get(a),
            subject_b=subjects.get(b),
            source_manifest_sha256=verified["manifest_content_sha256"],
        ))
    assert recipe is not None
    return Cohort(name, recipe, rows, kind=kind)
