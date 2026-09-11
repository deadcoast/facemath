"""Actual execution events, not model self-explanations or reconstructed history."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from .contracts import canonical_json, object_digest


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    records = [
        (p.relative_to(root).as_posix(), sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*.py"))
    ]
    return object_digest(records)


def numeric_environment() -> dict[str, str]:
    """Versions of numerical kernels that may change measurement semantics."""
    import cv2
    import numpy as np
    import scipy

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }


def environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "opencv-python", "mediapipe", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "numeric_environment": numeric_environment(),
        "engine_code_sha256": code_fingerprint(),
    }


class Journal:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, operation: str, payload: dict[str, Any]) -> None:
        entry = {
            "sequence": len(self.records),
            "operation": operation,
            "utc": datetime.now(UTC).isoformat(),
            "payload": payload,
            "previous_sha256": self.records[-1]["sha256"] if self.records else "0" * 64,
        }
        entry["sha256"] = object_digest(entry)
        self.records.append(entry)

    @contextmanager
    def step(self, operation: str, inputs: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.record(f"{operation}.begin", inputs)
        start = perf_counter()
        outputs: dict[str, Any] = {}
        try:
            yield outputs
        except Exception as exc:
            self.record(
                f"{operation}.error",
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_seconds": perf_counter() - start,
                },
            )
            raise
        else:
            self.record(
                f"{operation}.end",
                {"elapsed_seconds": perf_counter() - start, **outputs},
            )

    def jsonl(self) -> str:
        return "".join(canonical_json(entry) + "\n" for entry in self.records)


def verify_journal(records: list[dict[str, Any]]) -> None:
    previous = "0" * 64
    for sequence, entry in enumerate(records):
        payload = dict(entry)
        digest = payload.pop("sha256")
        if (
            payload["sequence"] != sequence
            or payload["previous_sha256"] != previous
            or object_digest(payload) != digest
        ):
            raise ValueError(f"journal chain mismatch at record {sequence}")
        previous = digest
