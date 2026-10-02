"""A qualified study enters the same pair instrument with its raw provenance.

The portable snapshot contains source rasters, reference annotations, acquisition
receipts and observations, not just an approval string. Loading recomputes the
existing qualification pipeline and compares exact results. This does not rerun
a native model, prove human independence, or attest an external timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
import json
import stat
import zipfile

from .contracts import ObservationError, canonical_json, object_digest
from .study import read_json

MAX_FILES = 30000
MAX_BYTES = 2 * 1024**3


def pack_snapshot(root: Path) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        paths = sorted(root.rglob("*"))
        if len(paths) > MAX_FILES:
            raise ObservationError("qualification snapshot exceeds file budget")
        total = 0
        for path in paths:
            if path.is_symlink():
                raise ObservationError("qualification snapshot cannot contain symlinks")
            if not path.is_file():
                continue
            total += path.stat().st_size
            if total > MAX_BYTES:
                raise ObservationError("qualification snapshot exceeds byte budget")
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes())
    return output.getvalue()


def unpack_snapshot(data: bytes, destination: Path) -> None:
    """Bounded flat-file inventory extraction; never trust archive filenames."""
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise ObservationError("qualification extraction requires an empty private directory")
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise ObservationError("invalid qualification snapshot bytes")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES or sum(i.file_size for i in entries) > MAX_BYTES:
                raise ObservationError("qualification snapshot exceeds extraction budget")
            seen = set()
            for info in entries:
                # ZipInfo truncates NULs and normalizes backslashes on Windows.
                # Validate the original decoded name before using any alias.
                name = info.orig_filename
                path = PurePosixPath(name)
                if (name != info.filename or not name or "\\" in name or "\x00" in name or ":" in name
                        or path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))
                        or name.casefold() in seen
                        or any(p.rstrip(" .") != p for p in path.parts)
                        or any(p.split(".")[0].upper() in {"CON","PRN","AUX","NUL",*("COM"+str(i) for i in range(1,10)),*("LPT"+str(i) for i in range(1,10))} for p in path.parts)
                        or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ObservationError("unsafe or duplicate qualification snapshot path")
                seen.add(name.casefold())
            # A file cannot also be a directory, including case aliases on
            # Windows. Validate the entire inventory before publishing bytes.
            for info in entries:
                path = PurePosixPath(info.filename)
                if any(parent.as_posix().casefold() in seen for parent in path.parents if parent != PurePosixPath('.')):
                    raise ObservationError("unsafe qualification snapshot file/directory collision")
            for info in entries:
                path = PurePosixPath(info.filename)
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(archive.read(info))
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise ObservationError(f"cannot read qualification snapshot: {exc}") from exc


@dataclass(frozen=True, init=False)
class ValidatedQualification:
    """Construct only by remeasurement/replay of retained raw study artifacts."""
    snapshot: bytes
    report_json: str
    study_json: str
    calibration: Any

    @classmethod
    def from_directory(cls, root: str | Path) -> ValidatedQualification:
        from .study_io import verify_qualification_bundle
        path = Path(root).resolve()
        verify_qualification_bundle(path)
        return cls.from_bytes(pack_snapshot(path))

    @classmethod
    def from_bytes(cls, data: bytes) -> ValidatedQualification:
        from .study_io import replay_qualification_bundle
        from .bias import load_localization_calibration
        with TemporaryDirectory(prefix="facemath-admission-") as directory:
            root = Path(directory) / "qualification"
            root.mkdir()
            unpack_snapshot(data, root)
            # Uses existing source-coordinate evaluator, development model fit,
            # calibration, acceptance computation, and all saved observations.
            replay_qualification_bundle(root, Path(directory) / "recomputed")
            report = read_json(root / "qualification.json")
            study = read_json(root / "acquisition/study/study.lock.json")
            calibration = load_localization_calibration(read_json(root / "localization_calibration.json"))
        obj = object.__new__(cls)
        object.__setattr__(obj, "snapshot", data)
        object.__setattr__(obj, "report_json", canonical_json(report))
        object.__setattr__(obj, "study_json", canonical_json(study))
        object.__setattr__(obj, "calibration", calibration)
        return obj

    @property
    def report(self) -> dict[str, Any]:
        return json.loads(self.report_json)

    @property
    def study(self) -> dict[str, Any]:
        return json.loads(self.study_json)

    @property
    def summary(self) -> dict[str, Any]:
        report = self.report
        admission = report.get("empirical_admission", {"status": "not_prespecified", "passed": False})
        return {"format": "facemath.validated_qualification_link.v10",
            "snapshot_sha256": sha256(self.snapshot).hexdigest(),
            "qualification_sha256": report["content_sha256"], "study_sha256": report["study_sha256"],
            "calibration_sha256": self.calibration.digest,
            "admission": admission, "raw_qualification_recomputed": True,
            "observer_reexecuted": False,
            "scope": "synthetic control only" if report["execution_kind"] == "synthetic_fixture" else "declared native reference study",
            "limit": "exact numerical replay, not externally attested source truth or independent human provenance"}

    def require_admitted(self) -> None:
        admission = self.report.get("empirical_admission", {})
        if not admission.get("passed"):
            reasons = [f"{c['category']}: {', '.join(c['reasons'])}" for c in admission.get("categories", []) if not c["passes"]]
            raise ObservationError("empirical admission withheld: " + ("; ".join(reasons) or "no prespecified held-out acceptance policy"))
        if admission.get("calibration_sha256") != self.calibration.digest:
            raise ObservationError("admission belongs to another calibration")

    def require_query(self, groups: tuple[Any, Any], calibration: Any) -> None:
        self.require_admitted()
        model = getattr(calibration, "model", calibration)
        if model.digest != self.calibration.digest:
            raise ObservationError("query calibration differs from replayed admitted study")
        if groups is None or len(groups) != 2:
            raise ObservationError("empirically admitted analysis requires two complete measured source groups")
        admission = self.report["empirical_admission"]
        permitted = set(admission["policy"]["required_categories"])
        cases = self.study["cases"]
        enrolled_images = {c["image_sha256"] for c in cases}
        enrolled_groups = {c["group_id"] for c in cases}
        enrolled_lineages = {v for c in cases for v in c["lineage_ids"]}
        for group in groups:
            if group.category not in permitted:
                raise ObservationError("query category is outside the frozen admitted scope")
            if group.group_id in enrolled_groups or set(group.lineage_ids) & enrolled_lineages:
                raise ObservationError("query group/lineage leaks enrolled development, calibration or evaluation evidence")
            if any(s.image_sha256 in enrolled_images for s in group.sources):
                raise ObservationError("query context image leaks enrolled development, calibration or evaluation evidence")


def load_snapshot(path: str | Path) -> ValidatedQualification:
    return ValidatedQualification.from_bytes(Path(path).read_bytes())
