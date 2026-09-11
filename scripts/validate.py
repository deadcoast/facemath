"""Run the local verification suite and retain actual command outputs.

This script does not manufacture a historical trace. It executes each command
now and records that command's stdout/stderr/return code. Use a new output path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    commands = [
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={out / 'pytest.xml'}"],
        [
            sys.executable,
            "-m",
            "facemath",
            "laboratory",
            "--out",
            str(out / "laboratory.json"),
        ],
        [sys.executable, "-m", "facemath", "demo", "--out", str(out / "demo")],
        [sys.executable, "-m", "facemath", "verify", str(out / "demo")],
        [
            sys.executable,
            "-m",
            "facemath",
            "replay",
            str(out / "demo"),
            "--out",
            str(out / "replay"),
        ],
        [
            sys.executable,
            "-m",
            "facemath",
            "capabilities",
            "--out",
            str(out / "capabilities.json"),
        ],
    ]
    for i, command in enumerate(commands):
        start = datetime.now(UTC).isoformat()
        result = subprocess.run(
            command, cwd=root, env=env, text=True, capture_output=True
        )
        record = {
            "command": command,
            "cwd": str(root),
            "started_utc": start,
            "finished_utc": datetime.now(UTC).isoformat(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        (out / f"command-{i:02}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
