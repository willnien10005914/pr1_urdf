"""Background worker for retarget and Isaac Lab jobs."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .csv_motion import motion_from_hhtools_csv
from .jobs import JobStore
from .progress import eta_seconds, parse_metrics, train_percent
from .train_cmd import train_argv

STAGES = {
    "queued": "Queued",
    "download": "Downloading the clip",
    "pose": "Estimating the body pose",
    "retarget": "Retargeting onto RP1",
    "starting_isaac": "Starting Isaac Lab",
    "training": "Training",
    "complete": "Finished",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "blocked": "Waiting for a local environment",
}


def hhtools_python() -> Path | None:
    root = Path(os.environ.get("PR1_HHTOOLS_ROOT", Path.home() / "projects" / "human-humanoid-tools"))
    python = root / ".venv" / "bin" / "python"
    return python if python.is_file() else None


def train_python() -> Path | None:
    override = os.environ.get("PR1_TRAIN_PYTHON")
    candidates = [
        Path(override) if override else None,
        Path.home() / "projects" / "roboparty_train" / ".venv" / "bin" / "python",
        Path.home() / "projects" / "roboparty_train" / ".venv-physx" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def train_root() -> Path:
    return Path(os.environ.get("PR1_ROBOPARTY_TRAIN_ROOT", Path.home() / "projects" / "roboparty_train"))


class Worker:
    def __init__(self, store: JobStore):
        self.store = store
        self.process: subprocess.Popen | None = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True
        self._kill()

    def _kill(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()

    def recover(self) -> None:
        running = {"download", "pose", "retarget", "starting_isaac", "training"}
        for job in self.store.list_jobs():
            if job.get("state") in running:
                self.store.update(
                    job["id"], state="failed", stage_label="Interrupted",
                    error="The site restarted before this job finished. Start it again.",
                )

    def run_forever(self, poll_s: float = 2.0) -> None:
        import fcntl
        lock_path = self.store.data_dir / "worker.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = lock_path.open("a")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("PR1 worker is already running.", flush=True)
            return
        self.recover()
        while not self._stop:
            if not self.run_once():
                time.sleep(poll_s)

    def run_once(self) -> bool:
        pending = [job for job in reversed(self.store.list_jobs()) if job.get("state") == "queued"]
        if not pending:
            return False
        job = pending[0]
        if job["kind"] == "retarget":
            self.retarget(job)
        elif job["kind"] == "train":
            self.train(job)
        else:
            self.store.update(job["id"], state="failed", error=f"Unknown job kind {job['kind']}", stage_label="Failed")
        return True

    def _cancelled(self, job_id: str) -> bool:
        return self.store.get(job_id).get("state") == "cancelled"

    def _stream(self, job_id: str, cmd: list[str], *, cwd: Path, env: dict, on_line) -> int:
        self.store.append_log(job_id, "$ " + " ".join(cmd) + "\n")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.process = process
        assert process.stdout is not None
        for line in process.stdout:
            self.store.append_log(job_id, line)
            if self._cancelled(job_id):
                self._kill()
                break
            on_line(line)
        code = process.wait()
        self.process = None
        return code

    def retarget(self, job: dict) -> None:
        job_id = job["id"]
        python = hhtools_python()
        if python is None:
            self.store.update(
                job_id, state="blocked", stage_label=STAGES["blocked"],
                error="human-humanoid-tools .venv was not found. Set PR1_HHTOOLS_ROOT.",
            )
            return
        config = job["config"]
        out = self.store.directory(job_id)
        script = self.store.root / "scripts" / "youtube_seconds_to_hhtools.py"
        cmd = [
            str(python), str(script),
            "--youtube-url", config["youtube_url"],
            "--start", str(config["start"]),
            "--end", str(config["end"]),
            "--reference", config.get("reference") or "smplx",
            "--out", str(out),
        ]
        env = os.environ.copy()
        env["PR1_HHTOOLS_ROOT"] = str(python.parents[2])

        def on_line(line: str) -> None:
            if not line.startswith("PR1_STATUS "):
                return
            try:
                payload = json.loads(line[len("PR1_STATUS "):])
            except json.JSONDecodeError:
                return
            stage = payload.get("stage") or "retarget"
            self.store.update(
                job_id,
                state=stage if stage != "ready" else "retarget",
                stage_label=payload.get("message") or STAGES.get(stage, stage),
                percent=float(payload.get("percent") or 0),
            )

        code = self._stream(job_id, cmd, cwd=self.store.root, env=env, on_line=on_line)
        if self._cancelled(job_id):
            return
        csv_path = out / "rp1.csv"
        if code != 0 or not csv_path.is_file():
            tail = self.store.log_tail(job_id, 20)
            self.store.update(
                job_id, state="failed", stage_label="Failed",
                error=tail.splitlines()[-1] if tail else f"Retarget exited {code}",
            )
            return
        motion = motion_from_hhtools_csv(csv_path.read_text(encoding="utf-8"), motion_id=job_id, title=job["title"])
        (out / "motion.json").write_text(json.dumps(motion), encoding="utf-8")
        self.store.update(job_id, state="complete", stage_label="Retarget ready", percent=1.0, eta_s=0, error="")
        self.store.event(job_id, "retarget complete", frames=motion["frames"])

    def train(self, job: dict) -> None:
        job_id = job["id"]
        python = train_python()
        root = train_root()
        if python is None or not (root / "robolab").is_dir():
            self.store.update(
                job_id, state="blocked", stage_label=STAGES["blocked"], percent=0, eta_s=None,
                error=(
                    "Isaac Lab training is not installed for RP1. From ~/projects/roboparty_train run "
                    "`uv sync --locked` (MJWarp) or the PhysX extra in that repo's README, then restart this site. "
                    "Reference clips in the viewer stay available."
                ),
            )
            return
        config = job["config"]
        overrides = {
            "learning_rate": config["learning_rate"],
            "entropy_coef": config["entropy_coef"],
            "clip_param": config["clip_param"],
            "gamma": config["gamma"],
            "num_steps_per_env": config["num_steps_per_env"],
        }
        script = self.store.root / "scripts" / "train_pr1_job.py"
        cmd = [str(python), str(script), *train_argv(config)]
        env = os.environ.copy()
        env["PR1_PPO_OVERRIDES"] = json.dumps(overrides)
        origin = self.store.clock()
        self.store.update(job_id, state="starting_isaac", stage_label=STAGES["starting_isaac"], percent=0.08)
        latest = {"iteration": 0}

        def on_line(line: str) -> None:
            if line.startswith("PR1_STATUS "):
                return
            metrics = parse_metrics(line)
            if "iteration" not in metrics:
                return
            iteration = int(metrics["iteration"])
            total = int(metrics.get("iteration_total") or config["max_iterations"])
            elapsed = self.store.clock() - origin
            eta = eta_seconds(iteration, total, elapsed)
            latest["iteration"] = iteration
            self.store.update(
                job_id,
                state="training",
                stage_label=f"Learning iteration {iteration}/{total}",
                percent=train_percent(iteration, total, started=True),
                eta_s=eta,
                metrics={**metrics, "elapsed_s": round(elapsed, 1)},
            )

        code = self._stream(job_id, cmd, cwd=root, env=env, on_line=on_line)
        if self._cancelled(job_id):
            return
        if code != 0:
            tail = self.store.log_tail(job_id, 15)
            self.store.update(
                job_id, state="failed", stage_label="Failed",
                error=tail.splitlines()[-1] if tail else f"Training exited {code}",
            )
            return
        self.store.update(job_id, state="complete", stage_label="Training finished", percent=1, eta_s=0, error="")
