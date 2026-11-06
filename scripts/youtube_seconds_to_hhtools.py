#!/usr/bin/env python3
"""YouTube interval → GVHMR → HHTools retarget CSV for RP1.

Run with the human-humanoid-tools virtualenv so ``hhtools`` imports resolve.
Progress lines start with ``PR1_STATUS`` and are JSON.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pr1store.youtube import require_youtube  # noqa: E402


def status(stage: str, percent: float, message: str) -> None:
    print("PR1_STATUS " + json.dumps({"stage": stage, "percent": percent, "message": message}), flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def download(url: str, start: float, end: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 1024:
        status("download", 0.25, f"Reusing {target.name}")
        return
    yt_dlp = shutil.which("yt-dlp") or "yt-dlp"
    section = f"*{start}-{end}"
    try:
        run([
            yt_dlp, "-f", "mp4/best",
            "--download-sections", section,
            "--force-keyframes-at-cuts",
            "-o", str(target),
            url,
        ])
    except subprocess.CalledProcessError:
        raw = target.with_suffix(".raw.mp4")
        run([yt_dlp, "-f", "mp4/best", "-o", str(raw), url])
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        run([
            ffmpeg, "-y", "-ss", str(start), "-i", str(raw),
            "-t", str(max(0.1, end - start)), "-c", "copy", str(target),
        ])
    if not target.is_file() or target.stat().st_size < 1024:
        raise SystemExit("yt-dlp did not write a video file. The clip may be private or blocked.")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube-url", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--reference", choices=("smpl", "smplx"), default="smplx")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.5 <= args.end - args.start <= 60:
        raise SystemExit("Clip length must be between 0.5 and 60 seconds.")
    video_id = require_youtube(args.youtube_url)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    video = out / "clip.mp4"
    status("download", 0.1, "Downloading the YouTube interval")
    download(args.youtube_url, args.start, args.end, video)

    hhtools = Path(os.environ.get("PR1_HHTOOLS_ROOT", Path.home() / "projects" / "human-humanoid-tools"))
    status("pose", 0.35, f"Estimating {args.reference.upper()} pose with GVHMR")
    sys.path.insert(0, str(hhtools))
    os.environ.setdefault("HHTOOLS_GVHMR_ROOT", str(Path.home() / "projects" / "GVHMR"))
    os.environ.setdefault("HHTOOLS_GVHMR_PYTHON", str(Path(os.environ["HHTOOLS_GVHMR_ROOT"]) / ".venv/bin/python"))
    from hhtools.integrations.gvhmr import gvhmr_status, run_gvhmr

    ready = gvhmr_status(probe_runtime=True)
    if not ready.get("ready"):
        missing = ready.get("missing") or ["GVHMR runtime"]
        raise SystemExit("GVHMR is not ready:\n- " + "\n- ".join(map(str, missing)))
    gvhmr_dir = out / "gvhmr"
    gvhmr_dir.mkdir(parents=True, exist_ok=True)
    staged = gvhmr_dir / video.name
    if not staged.is_file() or staged.stat().st_size != video.stat().st_size:
        shutil.copy2(video, staged)
    result = run_gvhmr(staged, gvhmr_dir, static_cam=True)
    result_path = Path(result)

    status("retarget", 0.7, f"Retargeting onto RP1 with HHTools ({args.reference})")
    motions = out / "motions"
    motions.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv") or "uv"
    results = result_path if result_path.is_file() else next(iter(sorted(result_path.rglob("hmr4d_results.pt"))), None)
    if results is None:
        raise SystemExit(f"No hmr4d_results.pt under {result_path}")
    run([
        uv, "run", "hhtools", "import", "run",
        "--dataset", "gvhmr",
        "--root", str(results.parent),
        "--sequence", results.stem,
        "--out", str(motions),
    ], cwd=hhtools)
    npzs = sorted(motions.rglob("*.npz"))
    if not npzs:
        raise SystemExit(f"HHTools import wrote no NPZ under {motions}")
    csv_path = out / "rp1.csv"
    run([
        uv, "run", "python", str(ROOT / "scripts" / "bootstrap_rp1_preset.py"),
    ], cwd=hhtools)
    run([
        uv, "run", "hhtools", "retarget", "run",
        str(npzs[0]),
        "--robot", "rp1",
        "-o", str(csv_path),
        "--calibration-reference", args.reference,
    ], cwd=hhtools)
    manifest = {
        "youtube_url": args.youtube_url,
        "video_id": video_id,
        "start": args.start,
        "end": args.end,
        "reference": args.reference,
        "csv": str(csv_path),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    status("ready", 1.0, f"Retarget CSV ready ({csv_path.name})")
    print("CSV", csv_path, flush=True)


if __name__ == "__main__":
    main()
