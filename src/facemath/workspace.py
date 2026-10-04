"""Loopback image-pair application; the existing analyze command owns all maths."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from urllib.parse import urlsplit
import uuid
import zipfile

import cv2
import numpy as np

from .observation.contracts import ObservationError

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * ((MAX_IMAGE_BYTES + 2) // 3 * 4) + 65536
RUN_ID = re.compile(r"[0-9a-f]{32}\Z")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Workspace:
    """One native worker at a time, create-only inputs/results, durable status."""

    def __init__(self, root: str | Path, model: str | Path,
                 model_sha256: str | None = None, *, timeout: float = 1200):
        self.root = Path(root).resolve()
        self.model = Path(model).resolve()
        if not self.model.is_file():
            raise ObservationError("Choose an existing local Face Landmarker .task model.")
        self.model_hash = sha256(self.model.read_bytes()).hexdigest()
        if model_sha256 and self.model_hash != model_sha256:
            raise ObservationError("Local model bytes do not match --model-sha256.")
        self.root.mkdir(parents=True, exist_ok=True)
        self._file_lock = (self.root / "workspace.lock").open("a+b")
        self._file_lock.seek(0)
        self._file_lock.write(b"0")
        self._file_lock.flush()
        self._file_lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._file_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file_lock.close()
            raise ObservationError("This comparison directory is already open in another workspace.") from None
        self.timeout = timeout
        self.lock = threading.RLock()
        self.active: str | None = None
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        # An interrupted process never becomes a completed comparison on restart.
        for path in self.root.glob("*/status.json"):
            if RUN_ID.fullmatch(path.parent.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] == "running":
                    self._save(path.parent, {**record, "status": "interrupted",
                        "finished": now(), "message": "Application stopped before recording completion. Start a new comparison; the original files are retained."})

    @staticmethod
    def _save(directory: Path, record: dict) -> None:
        temp = directory / "status.tmp"
        temp.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(directory / "status.json")

    def status(self, run_id: str) -> dict:
        if not RUN_ID.fullmatch(run_id):
            raise FileNotFoundError("Comparison not found.")
        with self.lock:
            return json.loads((self.root / run_id / "status.json").read_text(encoding="utf-8"))

    def history(self) -> list[dict]:
        with self.lock:
            rows = [self.status(p.name) for p in self.root.iterdir()
                    if RUN_ID.fullmatch(p.name) and (p / "status.json").is_file()]
        return sorted(rows, key=lambda row: row["created"], reverse=True)

    @staticmethod
    def _image(item: dict) -> tuple[str, bytes, np.ndarray]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Each image needs its filename and encoded contents.")
        value = item.get("data")
        if not isinstance(value, str) or len(value) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise ValueError("Each image must be at most 16 MiB.")
        raw = base64.b64decode(value, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Each image must be at most 16 MiB.")
        if not (raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8\xff")):
            raise ValueError("Choose a PNG or JPEG image.")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or min(image.shape[:2]) < 8:
            raise ValueError("The image could not be decoded. Choose a valid PNG or JPEG.")
        if image.shape[0] * image.shape[1] > 30_000_000:
            raise ValueError("Images must contain at most 30 million pixels.")
        name = item["name"].replace("\\", "/").split("/")[-1][:180]
        return name, raw, image

    def submit(self, request: dict) -> dict:
        with self.lock:
            if self.active:
                raise BlockingIOError("A comparison is already running. Wait for it or cancel it first.")
            images = [self._image(request.get(key)) for key in ("a", "b")]
            tolerance = request.get("tolerance", .05)
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not .001 <= tolerance <= 1:
                raise ValueError("Tolerance must be between 0.001 and 1 eye span.")
            run_id = uuid.uuid4().hex
            directory = self.root / run_id
            directory.mkdir()
            inputs = []
            for key, (name, raw, decoded) in zip(("a", "b"), images):
                source = directory / f"input_{key}.image"
                source.write_bytes(raw)
                # Browser previews use the same stored-raster orientation as analysis.
                ok, preview = cv2.imencode(".png", decoded)
                if not ok:
                    raise OSError("Could not write the source preview.")
                (directory / f"preview_{key}.png").write_bytes(preview.tobytes())
                inputs.append({"name": name, "sha256": sha256(raw).hexdigest(),
                               "width": decoded.shape[1], "height": decoded.shape[0]})
            command = [sys.executable, "-m", "facemath", "analyze",
                str(directory / "input_a.image"), str(directory / "input_b.image"),
                "--model", str(self.model), "--model-sha256", self.model_hash,
                "--tolerance", str(tolerance), "--out", str(directory / "result")]
            record = {"id": run_id, "status": "running", "created": now(),
                "inputs": inputs, "tolerance": tolerance, "model_sha256": self.model_hash,
                "message": "Observing both images and computing the full comparison.",
                "directory": str(directory), "returncode": None}
            (directory / "request.json").write_text(json.dumps({k: record[k] for k in
                ("id", "created", "inputs", "tolerance", "model_sha256")}, indent=2), encoding="utf-8")
            (directory / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            self._save(directory, record)
            self.active = run_id
            try:
                env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLCONFIGDIR": str(directory / "runtime-cache")}
                with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
                    self.process = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except OSError as exc:
                record.update(status="failed", message=str(exc), finished=now())
                self._save(directory, record)
                self.active = None
                return record
            self.thread = threading.Thread(target=self._finish, args=(run_id, self.process), daemon=True)
            self.thread.start()
            return record

    def _finish(self, run_id: str, process: subprocess.Popen) -> None:
        directory = self.root / run_id
        timed_out = False
        try:
            process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        with self.lock:
            record = self.status(run_id)
            record.update(returncode=process.returncode, finished=now())
            try:
                if record["status"] == "cancelled":
                    pass
                elif timed_out:
                    record.update(status="failed", message="The comparison exceeded its time limit. Logs and inputs are retained; try a new pair.")
                elif process.returncode:
                    errors = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
                    message = next((s.removeprefix("FACEMATH: ") for s in reversed(errors.splitlines())
                                    if s.startswith("FACEMATH: ")), "Analysis failed. Open the execution log for details.")
                    record.update(status="failed", message=message)
                else:
                    from .observation.artifacts import verify_bundle
                    verify_bundle(directory / "result")
                    with zipfile.ZipFile(directory / "comparison.zip", "x", zipfile.ZIP_DEFLATED) as archive:
                        for path in sorted((directory / "result").iterdir()):
                            if path.is_file():
                                archive.write(path, "result/" + path.name)
                        for name in ("input_a.image", "input_b.image", "request.json", "command.json", "stdout.log", "stderr.log"):
                            archive.write(directory / name, name)
                    record.update(status="complete", message="Comparison ready. Inspect measured differences and their available support.")
            except Exception as exc:
                record.update(status="failed", message=f"Could not publish the comparison: {exc}")
            finally:
                self._save(directory, record)
                self.active = None
                self.process = None

    def cancel(self, run_id: str) -> dict:
        with self.lock:
            record = self.status(run_id)
            if self.active == run_id and self.process is not None:
                record.update(status="cancelled", message="Comparison cancelled. Inputs and execution logs are retained.")
                self._save(self.root / run_id, record)
                if self.process.poll() is None:
                    self.process.terminate()
            return record

    def close(self) -> None:
        with self.lock:
            if self.active:
                self.cancel(self.active)
        if self.thread:
            self.thread.join(timeout=15)
        self._file_lock.close()


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, workspace: Workspace, port: int = 0):
        self.workspace = workspace
        self.prefix = "/" + secrets.token_urlsafe(32) + "/"
        super().__init__(("127.0.0.1", port), Handler)
        self.origin = f"http://127.0.0.1:{self.server_port}"
        self.url = self.origin + self.prefix


class Handler(BaseHTTPRequestHandler):
    server: WorkspaceServer

    def log_message(self, *args) -> None:
        pass  # Never print capability URLs or image names to HTTP access logs.

    def _route(self) -> str | None:
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self._json(403, {"error": "Use the application's local address."})
            return None
        if self.headers.get("Origin") not in (None, self.server.origin):
            self._json(403, {"error": "Cross-origin requests are not accepted."})
            return None
        path = urlsplit(self.path).path
        if not path.startswith(self.server.prefix):
            self._json(404, {"error": "Application address not found."})
            return None
        return path[len(self.server.prefix):]

    def _send(self, code: int, body: bytes, mime: str, *, attachment: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        try:
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            # Closing a tab must not relabel a successfully started job as failed.
            self.close_connection = True

    def _json(self, code: int, data: dict | list) -> None:
        self._send(code, json.dumps(data, allow_nan=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        route = self._route()
        if route is None:
            return
        work = self.server.workspace
        try:
            if route == "":
                from .workspace_ui import page
                self._send(200, page().encode(), "text/html; charset=utf-8")
            elif route == "api/jobs":
                self._json(200, work.history())
            elif re.fullmatch(r"api/jobs/[0-9a-f]{32}", route):
                self._json(200, work.status(route.split("/")[-1]))
            elif re.fullmatch(r"runs/[0-9a-f]{32}/(?:result/)?[A-Za-z0-9_.-]+", route):
                parts = route.split("/")
                record = work.status(parts[1])
                if (work.root / parts[1]).is_symlink():
                    raise FileNotFoundError("Comparison not found.")
                relative = "/".join(parts[2:])
                if relative.startswith("result/"):
                    if record["status"] != "complete":
                        raise FileNotFoundError("The report is not ready.")
                    path = work.root / parts[1] / relative
                elif relative == "readout.html" and record["status"] == "complete":
                    from .observation.readout import render_pair_readout
                    evidence = json.loads((work.root / parts[1] / "result/evidence.json").read_text(encoding="utf-8"))
                    html = render_pair_readout(evidence)
                    self._send(200, html.encode(), "text/html; charset=utf-8")
                    return
                elif relative in {"preview_a.png", "preview_b.png", "stderr.log", "stdout.log", "command.json", "comparison.zip"}:
                    path = work.root / parts[1] / relative
                else:
                    raise FileNotFoundError("File not found.")
                if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(work.root):
                    raise FileNotFoundError("File not found.")
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if path.suffix == ".log":
                    mime = "text/plain; charset=utf-8"
                self._send(200, path.read_bytes(), mime, attachment=path.name if path.suffix in (".zip", ".npz") else None)
            else:
                raise FileNotFoundError("Page not found.")
        except (FileNotFoundError, ValueError):
            self._json(404, {"error": "Comparison or file not found."})

    def do_POST(self) -> None:
        route = self._route()
        if route is None:
            return
        try:
            if self.headers.get("Content-Type") != "application/json":
                raise ValueError("Send an application/json request.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                self.close_connection = True
                self._json(413, {"error": "Upload exceeds the two-image limit of 16 MiB per image."})
                return
            self.connection.settimeout(30)
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected an image-pair request.")
            if route == "api/jobs":
                self._json(202, self.server.workspace.submit(data))
            elif re.fullmatch(r"api/jobs/[0-9a-f]{32}/cancel", route):
                self._json(200, self.server.workspace.cancel(route.split("/")[-2]))
            else:
                self._json(404, {"error": "Action not found."})
        except BlockingIOError as exc:
            self._json(409, {"error": str(exc)})
        except (ValueError, TypeError, cv2.error) as exc:
            self._json(400, {"error": str(exc)})
        except FileNotFoundError:
            self._json(404, {"error": "Comparison not found."})
        except OSError as exc:
            self._json(500, {"error": f"Could not save or start the comparison: {exc}"})


def serve(args) -> None:
    if not 0 <= args.port <= 65535:
        raise ObservationError("Port must be between 0 and 65535; use 0 for an available port.")
    workspace = Workspace(args.out, args.model, args.model_sha256)
    try:
        server = WorkspaceServer(workspace, args.port)
    except OSError:
        workspace.close()
        raise
    print(f"FACEMATH workspace: {server.url}", flush=True)
    print(f"Comparisons are saved in {workspace.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        workspace.close()
        server.server_close()
