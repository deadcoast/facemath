"""Native worker entry point. No reference labels or split metadata are received."""
from __future__ import annotations

from pathlib import Path
import sys
import traceback

from .contracts import ObservationError, ProviderUnavailable
from .journal import code_fingerprint, numeric_environment
from .provider import MediaPipeProvider, DetectionMiss
from .study import ObserverDeclaration, load_rgb, read_json, write_json_new


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m facemath.observation.worker REQUEST RESPONSE", file=sys.stderr)
        return 2
    request_path, response_path = map(Path, args)
    response = {"format": "facemath.worker_response.v5", "observation": None}
    exit_code = 0
    try:
        request = read_json(request_path)
        if request.get("format") != "facemath.worker_request.v5":
            raise ObservationError("unsupported worker request")
        if request["engine_code_sha256"] != code_fingerprint() or request["numeric_environment"] != numeric_environment():
            raise ObservationError("worker code/numeric environment differs from parent")
        declaration = ObserverDeclaration.from_dict(request["observer"])
        if declaration.execution_kind != "native_mediapipe":
            raise ObservationError("native worker rejects synthetic declarations")
        image = load_rgb(request["image"])
        with MediaPipeProvider(request["model"], declaration.model_sha256,
                               noise_floor_px=declaration.noise_floor_px,
                               use_augmentations=declaration.attempted_samples == 7) as provider:
            if provider.version != declaration.runtime_version:
                raise ObservationError("worker observer runtime mismatch")
            observation = provider.observe(image)
        declaration.require_observation(observation, image)
        response.update(status="observed", observation=observation.to_dict(), detail="")
    except DetectionMiss as exc:
        response.update(status=exc.kind, detail=str(exc))
    except ObservationError as exc:
        response.update(status="contract_error", detail=str(exc)); exit_code = 2
    except Exception as exc:
        traceback.print_exc()
        response.update(status="runtime_error", detail=f"{type(exc).__name__}: {exc}"); exit_code = 3
    write_json_new(response_path, response)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
