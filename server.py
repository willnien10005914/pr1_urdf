#!/usr/bin/env python3
"""PR1 Studio web server. Python 3.11+, standard library only."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pr1store.ballet_steps import load_document, replay_motion, save_step
from pr1store.catalog import public_catalog
from pr1store.csv_motion import CsvMotionError, motion_from_hhtools_csv
from pr1store.jobs import JobStore
from pr1store.progress import format_duration
from pr1store.train_cmd import clamp_config
from pr1store.worker import Worker, hhtools_python, train_python
from pr1store.youtube import require_youtube

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
ASSETS = ROOT / "assets"
MOTIONS = ASSETS / "motions"
MAX_BODY = 20 * 1024 * 1024


def gpu_name() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return ""
    line = output.strip().splitlines()
    return line[0] if line else ""


def capabilities() -> dict:
    smpl = Path.home() / ".cache" / "hhtools" / "body_models" / "smpl" / "SMPL_NEUTRAL.npz"
    smplx = Path.home() / ".cache" / "hhtools" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    python = train_python()
    hhtools = hhtools_python()
    return {
        "yt_dlp": bool(subprocess_which("yt-dlp")),
        "ffmpeg": bool(subprocess_which("ffmpeg")),
        "hhtools": hhtools is not None,
        "smpl": smpl.exists(),
        "smplx": smplx.exists(),
        "train_python": str(python) if python else "",
        "gpu": gpu_name(),
    }


def subprocess_which(name: str) -> str | None:
    from shutil import which
    return which(name)


def public_job(job: dict, store: JobStore) -> dict:
    payload = dict(job)
    payload["eta"] = format_duration(job.get("eta_s"))
    payload["log_tail"] = store.log_tail(job["id"], 40)
    payload["has_motion"] = (store.directory(job["id"]) / "motion.json").is_file()
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "PR1Studio/0.1"

    @property
    def store(self) -> JobStore:
        return self.server.store

    def log_message(self, fmt: str, *args) -> None:
        if str(args[1:2]) == "('200',)" or (args and str(args[1]) == "200"):
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        path = unquote(urlsplit(self.path).path)
        try:
            if path == "/api/catalog":
                return self._json(public_catalog())
            if path == "/api/capabilities":
                return self._json(capabilities())
            if path == "/api/motions":
                index = json.loads((MOTIONS / "index.json").read_text(encoding="utf-8"))
                return self._json(index)
            if path.startswith("/api/motions/"):
                return self._motion(path.removeprefix("/api/motions/"))
            if path == "/api/ballet-steps":
                return self._json(load_document(self.store.data_dir))
            if path == "/api/ballet-steps/replay":
                return self._json(replay_motion(load_document(self.store.data_dir)))
            if path == "/api/jobs":
                return self._json({"jobs": [public_job(job, self.store) for job in self.store.list_jobs()]})
            if path.startswith("/api/jobs/"):
                return self._job(path.removeprefix("/api/jobs/"))
            return self._static(path)
        except KeyError:
            self._error(404, "Not found")
        except FileNotFoundError:
            self._error(404, "Not found")
        except Exception as error:  # surface a message, not a stack, to the browser
            self._error(500, str(error))

    def do_POST(self) -> None:
        path = unquote(urlsplit(self.path).path)
        try:
            if path.startswith("/api/ballet-steps/"):
                step_id = path.removeprefix("/api/ballet-steps/").strip("/")
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > MAX_BODY:
                    raise ValueError("JSON body is missing or too large")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                return self._json(save_step(
                    self.store.data_dir, step_id, payload.get("joints") or {}, payload.get("com"),
                ))
            if path == "/api/jobs":
                return self._create_job()
            if path.endswith("/cancel") and path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                job = self.store.update(job_id, state="cancelled", stage_label="Cancelled")
                worker: Worker = self.server.worker
                if worker.process is not None:
                    worker._kill()
                return self._json(public_job(job, self.store))
            self._error(404, "Not found")
        except KeyError:
            self._error(404, "Not found")
        except ValueError as error:
            self._error(400, str(error))
        except CsvMotionError as error:
            self._error(400, str(error))
        except Exception as error:
            self._error(500, str(error))

    def _create_job(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("JSON body is missing or too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        kind = payload.get("kind")
        if kind == "retarget":
            url = require_youtube(str(payload.get("youtube_url") or ""))
            start = float(payload.get("start"))
            end = float(payload.get("end"))
            if not 0.5 <= end - start <= 60:
                raise ValueError("Clip length must be between 0.5 and 60 seconds.")
            reference = payload.get("reference") or "smplx"
            if reference not in ("smpl", "smplx"):
                raise ValueError("reference must be smpl or smplx")
            title = str(payload.get("title") or f"YouTube {start:g}–{end:g}s")[:80]
            job = self.store.create("retarget", title, {
                "youtube_url": str(payload["youtube_url"]).strip(),
                "video_id": url,
                "start": start,
                "end": end,
                "reference": reference,
            })
            return self._json(public_job(job, self.store), status=202)
        if kind == "train":
            config = clamp_config(payload.get("config") or {})
            title = str(payload.get("title") or config["task"])[:80]
            job = self.store.create("train", title, config)
            return self._json(public_job(job, self.store), status=202)
        if kind == "csv":
            text = str(payload.get("csv") or "")
            if len(text) < 20:
                raise ValueError("Paste an HHTools robot CSV.")
            title = str(payload.get("title") or "HHTools CSV")[:80]
            motion = motion_from_hhtools_csv(text, title=title)
            job = self.store.create("csv", title, {"source": "upload"}, state="complete")
            motion["id"] = job["id"]
            (self.store.directory(job["id"]) / "motion.json").write_text(json.dumps(motion), encoding="utf-8")
            job = self.store.update(job["id"], stage_label="CSV ready", percent=1, eta_s=0)
            return self._json({"job": public_job(job, self.store), "motion": motion})
        raise ValueError("kind must be retarget, train, or csv")

    def _job(self, rest: str) -> None:
        job_id, _, extra = rest.partition("/")
        if extra == "motion":
            path = self.store.directory(job_id) / "motion.json"
            if not path.is_file():
                raise KeyError(job_id)
            self._bytes(path.read_bytes(), "application/json")
            return
        if extra == "preview.gif":
            path = self.store.directory(job_id) / "preview.gif"
            if not path.is_file():
                raise KeyError(job_id)
            self._bytes(path.read_bytes(), "image/gif", cache="no-store")
            return
        if extra == "preview.json":
            path = self.store.directory(job_id) / "preview.json"
            if not path.is_file():
                raise KeyError(job_id)
            self._bytes(path.read_bytes(), "application/json", cache="no-store")
            return
        if extra:
            raise KeyError(extra)
        self._json(public_job(self.store.get(job_id), self.store))

    def _motion(self, motion_id: str) -> None:
        if not motion_id.replace("_", "").isalnum():
            raise KeyError(motion_id)
        path = MOTIONS / f"{motion_id}.json"
        if not path.is_file():
            raise KeyError(motion_id)
        self._bytes(path.read_bytes(), "application/json")

    def _static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        if path.startswith("/assets/"):
            base, relative = ASSETS, path.removeprefix("/assets/")
        else:
            base, relative = PUBLIC, path.lstrip("/")
        base_resolved = base.resolve()
        target = (base / relative).resolve()
        if target != base_resolved and base_resolved not in target.parents or not target.is_file():
            self._error(404, "Not found")
            return
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._bytes(target.read_bytes(), mime)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body: bytes, mime: str, cache: str = "") -> None:
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--no-worker", action="store_true")
    args = parser.parse_args()
    data = Path(args.data_dir) if args.data_dir else ROOT / ".data"
    store = JobStore(ROOT, data)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = store
    server.worker = Worker(store)
    if not args.no_worker:
        thread = threading.Thread(target=server.worker.run_forever, name="pr1-worker", daemon=True)
        thread.start()
    print(f"PR1 Studio http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.worker.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
