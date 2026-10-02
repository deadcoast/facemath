"""Necessary runtime compatibility for exact replay; never a tolerance policy."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from typing import Any

import cv2
import numpy as np
import scipy
import scipy.linalg  # Load LAPACK/BLAS before inspecting the process.
from threadpoolctl import threadpool_info

from .contracts import ObservationError, object_digest
from .runtime import OPENCV_DISTRIBUTIONS

FORMAT = "facemath.numeric_environment.v12"
DISPATCH_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE", "GOTO_NUM_THREADS",
    "OMP_NUM_THREADS", "OMP_DYNAMIC", "MKL_NUM_THREADS", "MKL_DYNAMIC",
    "MKL_CBWR", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES",
)


@lru_cache(maxsize=512)
def _binary(path: str) -> dict[str, Any]:
    # Loaded libraries and their installation are immutable for this process.
    # An in-place package replacement requires a restart, not a cache refresh.
    source = Path(path)
    digest = sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": source.stat().st_size, "sha256": digest.hexdigest()}


def _compiled(path: str) -> bool:
    name = path.lower()
    return name.endswith((".pyd", ".dll", ".dylib", ".so")) or ".so." in name


@lru_cache(maxsize=1)
def _installation() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "pillow", *OPENCV_DISTRIBUTIONS):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        binaries = {str(p).replace("\\", "/"): _binary(str(dist.locate_file(p).resolve()))
                    for p in dist.files or () if _compiled(str(p))}
        if not binaries:
            raise ObservationError(f"cannot identify compiled numerical distribution: {name}")
        packages[name] = {"version": dist.version, "binaries": dict(sorted(binaries.items()))}
    if not {"numpy", "scipy"} <= packages.keys() or not set(OPENCV_DISTRIBUTIONS) & packages.keys():
        raise ObservationError("numerical runtime requires identifiable NumPy, SciPy and OpenCV builds")
    core = importlib.import_module("numpy._core._multiarray_umath" if np.__version__.split('.')[0] != '1'
                                   else "numpy.core._multiarray_umath")
    features = getattr(core, "__cpu_features__", None)
    if not isinstance(features, dict) or not features:
        raise ObservationError("NumPy CPU dispatch information is unavailable")
    return {
        "interpreter": {"implementation": platform.python_implementation(), "version": platform.python_version(),
                        "compiler": platform.python_compiler(), "build": list(platform.python_build()),
                        "byteorder": sys.byteorder, "float_mant_dig": sys.float_info.mant_dig,
                        "float_rounds": sys.float_info.rounds},
        "platform": {"system": platform.system(), "machine": platform.machine(), "libc": list(platform.libc_ver())},
        "packages": packages,
        "numpy_cpu": {"features": {k: bool(v) for k, v in sorted(features.items())},
                      "baseline": list(getattr(core, "__cpu_baseline__", ())),
                      "dispatch": list(getattr(core, "__cpu_dispatch__", ()))},
        "opencv_build_sha256": sha256(cv2.getBuildInformation().encode()).hexdigest(),
    }


def numeric_environment() -> dict[str, Any]:
    """Path-independent build identity plus current numerical execution settings.

    A match is necessary, not sufficient: replay must still compare its outputs.
    This does not change thread counts, kernels, observations or fitted models.
    """
    installation = _installation()
    pools = []
    for pool in threadpool_info():
        if pool["user_api"] != "blas":
            continue
        identity = {k: v for k, v in pool.items() if k not in {"filepath", "prefix"}}
        identity["binary"] = dict(_binary(str(Path(pool["filepath"]).resolve())))
        pools.append(identity)
    result = {"format": FORMAT, "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__,
              "installation": deepcopy(installation),
              "execution": {"blas": sorted(pools, key=object_digest),
                            "opencv_cpu": cv2.getCPUFeaturesLine(), "opencv_optimized": cv2.useOptimized(),
                            "opencv_threads": cv2.getNumThreads(),
                            "dispatch_environment": {key: os.environ.get(key) for key in DISPATCH_VARIABLES}}}
    result["content_sha256"] = object_digest(result)
    return result


def compare_numeric_environments(expected: dict, actual: dict | None = None) -> dict[str, Any]:
    """Explain incompatibility without changing evidence or executing a replay."""
    actual = numeric_environment() if actual is None else actual
    differences = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in left or key not in right:
                    differences.append({"path": child, "kind": "missing", "expected_present": key in left,
                                        "actual_present": key in right})
                else:
                    walk(left[key], right[key], child)
        elif type(left) is not type(right) or left != right:
            differences.append({"path": path, "kind": "value", "expected": left, "actual": right})

    def valid(value: dict) -> bool:
        if not isinstance(value, dict) or value.get("format") != FORMAT:
            return False
        unsigned = {k: v for k, v in value.items() if k != "content_sha256"}
        return value.get("content_sha256") == object_digest(unsigned)

    walk(expected, actual, "")
    expected_valid, actual_valid = valid(expected), valid(actual)
    return {"compatible": expected_valid and actual_valid and not differences,
            "expected_profile_valid": expected_valid, "actual_profile_valid": actual_valid,
            "differences": differences, "exact_replay_verified": False,
            "meaning": "necessary runtime match only; byte verification and exact output comparison are separate gates"}


def require_numeric_environment(expected: dict, context: str) -> None:
    comparison = compare_numeric_environments(expected)
    if not comparison["compatible"]:
        paths = [d["path"] for d in comparison["differences"] if d["path"] != "/content_sha256"]
        detail = ", ".join(paths[:6]) or "invalid or legacy runtime profile"
        raise ObservationError(f"{context} requires matching numerical runtime: {detail}; "
                               "use runtime-profile --compare with the retained environment.json; "
                               "preserve this run and use its original runtime or a distinct new study")
