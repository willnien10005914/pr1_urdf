"""JSON job records under .data/jobs."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


class JobStore:
    def __init__(self, root: Path, data_dir: Path | None = None, clock=time.time):
        self.root = Path(root)
        self.data_dir = Path(data_dir) if data_dir else self.root / ".data"
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.jobs_dir / "index.json"
        self.clock = clock
        self._lock_path = self.jobs_dir / ".lock"

    def _lock(self):
        handle = self._lock_path.open("a+")
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _read_index(self) -> list[dict]:
        if not self.index_path.is_file():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, rows: list[dict]) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        os.replace(temporary, self.index_path)

    def create(self, kind: str, title: str, config: dict, *, state: str = "queued") -> dict:
        job_id = uuid.uuid4().hex
        now = self.clock()
        job = {
            "id": job_id,
            "kind": kind,
            "title": title,
            "state": state,
            "stage_label": "Queued",
            "percent": 0.0,
            "eta_s": None,
            "config": config,
            "metrics": {},
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
        directory = self.jobs_dir / job_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "events.jsonl").write_text("", encoding="utf-8")
        (directory / "log.txt").write_text("", encoding="utf-8")
        handle = self._lock()
        try:
            rows = self._read_index()
            rows.append(job)
            self._write_index(rows)
        finally:
            handle.close()
        return job

    def list_jobs(self) -> list[dict]:
        handle = self._lock()
        try:
            return list(reversed(self._read_index()))
        finally:
            handle.close()

    def get(self, job_id: str) -> dict:
        handle = self._lock()
        try:
            for job in self._read_index():
                if job["id"] == job_id:
                    return job
        finally:
            handle.close()
        raise KeyError(job_id)

    def update(self, job_id: str, **fields) -> dict:
        handle = self._lock()
        try:
            rows = self._read_index()
            found = None
            for job in rows:
                if job["id"] == job_id:
                    if job.get("state") == "cancelled" and fields.get("state") not in (None, "cancelled"):
                        found = job
                        break
                    job.update(fields)
                    job["updated_at"] = self.clock()
                    found = job
                    break
            if found is None:
                raise KeyError(job_id)
            self._write_index(rows)
            return found
        finally:
            handle.close()

    def directory(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def event(self, job_id: str, message: str, **extra) -> None:
        line = json.dumps({"time": self.clock(), "message": message, **extra}, ensure_ascii=False)
        with (self.directory(job_id) / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def append_log(self, job_id: str, text: str) -> None:
        with (self.directory(job_id) / "log.txt").open("a", encoding="utf-8") as stream:
            stream.write(text)

    def log_tail(self, job_id: str, limit: int = 80) -> str:
        path = self.directory(job_id) / "log.txt"
        if not path.is_file():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-limit:])
