"""Install the built wheel away from the source tree, test it, and replay evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel")
    parser.add_argument("--tests", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    wheel = Path(args.wheel).resolve()
    tests = Path(args.tests).resolve()
    bundle = Path(args.bundle).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="facemath-installed-") as workspace:
        work = Path(workspace)
        target = work / "installed"
        config = work / "pytest.ini"
        config.write_text(
            "[pytest]\nmarkers = live_observer: explicit local model integration\n"
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(target),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        commands = [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            [
                sys.executable,
                "-c",
                "import facemath,json; from facemath.observation.journal import code_fingerprint; "
                'print(json.dumps({"file":facemath.__file__,"version":facemath.__version__,"code_sha256":code_fingerprint()}))',
            ],
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(tests),
                "-c",
                str(config),
                f"--junitxml={out / 'pytest.xml'}",
            ],
            [
                sys.executable,
                "-m",
                "facemath",
                "replay",
                str(bundle),
                "--out",
                str(out / "replayed"),
            ],
        ]
        for index, command in enumerate(commands):
            start = datetime.now(UTC).isoformat()
            result = subprocess.run(
                command, cwd=work, env=env, text=True, capture_output=True
            )
            record = {
                "command": command,
                "cwd": str(work),
                "started_utc": start,
                "finished_utc": datetime.now(UTC).isoformat(),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            (out / f"command-{index:02}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            if result.returncode:
                raise SystemExit(result.returncode)
            if index == 1:
                imported = json.loads(result.stdout)
                assert Path(imported["file"]).is_relative_to(target), (
                    "source-tree shadowing detected"
                )
                original = json.loads((bundle / "environment.json").read_text())
                assert imported["code_sha256"] == original["engine_code_sha256"]


if __name__ == "__main__":
    main()
