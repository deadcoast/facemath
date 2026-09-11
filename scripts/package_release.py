"""Package source, tests, docs, built wheel and executed evidence; exclude caches."""

import json
import zipfile
from hashlib import sha256
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root.parent / "facemath_wave_03.zip"
exclude_parts = {"__pycache__", ".pytest_cache", "build", ".git"}
with zipfile.ZipFile(
    output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
) as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part in exclude_parts or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.suffix == ".pyc" or path.name == ".coverage":
            continue
        archive.write(path, Path(root.name) / relative)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
    file_count = len(archive.infolist())
record = {
    "artifact": output.name,
    "bytes": output.stat().st_size,
    "sha256": sha256(output.read_bytes()).hexdigest(),
    "files": file_count,
    "zip_crc_check": "passed",
}
(root.parent / "facemath_wave_03_release.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
