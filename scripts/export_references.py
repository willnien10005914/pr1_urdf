#!/usr/bin/env python3
"""Downsample roboparty_train RP1 references into browser motion JSON.

The source files stay in the local roboparty_train checkout. This script writes
compact 30 Hz clips the URDF viewer can play without Isaac Sim.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "motions"
TRAIN = Path.home() / "projects" / "roboparty_train" / "robolab" / "data" / "motions"
TARGET_FPS = 30.0


def _step(fps: float) -> int:
    return max(1, int(round(float(fps) / TARGET_FPS)))


def _as_wxyz(quat, order: str):
    quat = np.asarray(quat, dtype=np.float32)
    key = order.lower()
    if key == "wxyz":
        return quat
    if key == "xyzw":
        return quat[:, [3, 0, 1, 2]]
    raise SystemExit(f"unsupported quaternion order {order}")


def _pack(name: str, meta: dict, joint_names: list[str], fps: float, q, root_pos, root_quat) -> dict:
    step = _step(fps)
    index = np.arange(0, len(q), step)
    quat = np.asarray(root_quat, dtype=np.float32)[index]
    pos = np.asarray(root_pos, dtype=np.float32)[index]
    joints = np.asarray(q, dtype=np.float32)[index]
    payload = {
        "id": name,
        "fps": round(float(fps) / step, 4),
        "source_fps": float(fps),
        "frames": int(joints.shape[0]),
        "joint_names": list(joint_names),
        "q": np.round(joints, 5).tolist(),
        "root": np.round(np.concatenate([pos, quat], axis=1), 5).tolist(),
        **meta,
    }
    return payload


def from_amp(path: Path, name: str, meta: dict) -> dict:
    with path.open("rb") as handle:
        clip = pickle.load(handle)
    order = str(clip.get("quaternion_order", "wxyz"))
    quat = _as_wxyz(clip["root_rot"], order)
    return _pack(name, meta, list(clip["joint_names"]), float(clip["fps"]), clip["dof_pos"], clip["root_pos"], quat)


def from_bm(path: Path, name: str, meta: dict) -> dict:
    clip = np.load(path, allow_pickle=False)
    names = [str(item) for item in clip["joint_names"].tolist()]
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    # body 0 is the floating base in the BeyondMimic export.
    pos = np.asarray(clip["body_pos_w"][:, 0, :], dtype=np.float32)
    order = str(np.asarray(clip["quaternion_order"]).reshape(-1)[0])
    quat = _as_wxyz(clip["body_quat_w"][:, 0, :], order)
    meta = {
        **meta,
        "body_name": str(clip["body_names"][0]),
        "source_sha256": str(np.asarray(clip["source_sha256"]).reshape(-1)[0]),
    }
    return _pack(name, meta, names, fps, clip["joint_pos"], pos, quat)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [
        from_amp(
            TRAIN / "rp1_amp" / "neutral_walk_ff_180_R_002__A534.pkl",
            "walk",
            {
                "title": "Walk",
                "task": "RP1-AMP",
                "kind": "amp_reference",
                "summary": "Neutral forward walk used as an AMP demonstration. This is the reference trajectory, not a saved PPO checkpoint.",
                "source_file": "robolab/data/motions/rp1_amp/neutral_walk_ff_180_R_002__A534.pkl",
            },
        ),
        from_amp(
            TRAIN / "rp1_amp" / "stairs_climbing_up_start_R_001__A300.pkl",
            "climb",
            {
                "title": "Climb stairs",
                "task": "RP1-AMP",
                "kind": "amp_reference",
                "summary": "Stair-climb start clip from the AMP motion pack. Parkour training (RP1-Parkour) learns terrain; this clip is the kinematic reference.",
                "source_file": "robolab/data/motions/rp1_amp/stairs_climbing_up_start_R_001__A300.pkl",
            },
        ),
        from_bm(
            TRAIN / "rp1_bm" / "wave_R_001__A428.npz",
            "wave",
            {
                "title": "Wave",
                "task": "RP1-BeyondMimic",
                "kind": "beyondmimic_reference",
                "summary": "BeyondMimic wave clip shipped with roboparty_train. Tracking this motion is what RP1-BeyondMimic optimizes.",
                "source_file": "robolab/data/motions/rp1_bm/wave_R_001__A428.npz",
            },
        ),
        from_bm(
            TRAIN / "rp1_bm" / "dance1_subject2_smooth6hz.npz",
            "dance",
            {
                "title": "Dance",
                "task": "RP1-BeyondMimic",
                "kind": "beyondmimic_reference",
                "summary": "Smoothed dance reference at the BeyondMimic export rate. A backflip is not in this open motion pack.",
                "source_file": "robolab/data/motions/rp1_bm/dance1_subject2_smooth6hz.npz",
            },
        ),
    ]
    index = []
    for clip in clips:
        target = OUT / f"{clip['id']}.json"
        target.write_text(json.dumps(clip, separators=(",", ":")), encoding="utf-8")
        index.append({key: clip[key] for key in ("id", "title", "task", "kind", "summary", "fps", "frames", "source_file")})
        print(f"{clip['id']}: {clip['frames']} frames @ {clip['fps']} Hz, {target.stat().st_size} bytes")
    (OUT / "index.json").write_text(json.dumps({"motions": index}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
