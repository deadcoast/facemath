"""Execute release checks and retain the actual command, output and exit record.

Run from any directory. Outputs are create-only. This script does not download
models and does not mark a skipped live detector test as passed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "wave5"


def run(label: str, argv: list[str]) -> None:
    record_path = OUT / f"{label}.json"
    if record_path.exists():
        raise FileExistsError(record_path)
    start = datetime.now(timezone.utc).isoformat()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PIP_DISABLE_PIP_VERSION_CHECK="1")
    result = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True)
    record = {"command": argv, "cwd": str(ROOT), "started_utc": start,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    (OUT / f"{label}.txt").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run("final-doctor", [sys.executable, "-m", "facemath", "doctor"])
    run("final-source-tests", [sys.executable, "-m", "pytest", "-q",
        f"--junitxml={OUT / 'final-source-tests.xml'}"])
    run("final-laboratory", [sys.executable, "-m", "facemath", "study-lab",
                            "--out", str(OUT / "laboratory")])
    run("final-study-replay", [sys.executable, "-m", "facemath", "replay",
        str(OUT / "laboratory/qualification"), "--out", str(OUT / "source-study-replay")])
    run("final-pair-replay", [sys.executable, "-m", "facemath", "replay",
        str(OUT / "laboratory/qualified_pair"), "--out", str(OUT / "source-pair-replay")])
    run("final-wheel-build", [sys.executable, "-m", "pip", "wheel", "--no-deps",
        "--no-build-isolation", "--no-index", "--wheel-dir", str(ROOT / "dist"), str(ROOT)])
    run("final-wheel-validation", [sys.executable, str(ROOT / "scripts/check_wheel.py"),
        str(ROOT / "dist/facemath-0.5.0-py3-none-any.whl"), "--tests", str(ROOT / "tests"),
        "--bundle", str(OUT / "laboratory/qualification"),
        "--pair-bundle", str(OUT / "laboratory/qualified_pair"), "--out", str(OUT / "wheel-check")])


if __name__ == "__main__":
    main()
