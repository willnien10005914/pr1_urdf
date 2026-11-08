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


def _seed_legs_down(pipeline, *, right_knee: float = 0.35) -> None:
    """Start IK from a mild crouch so the knees do not flip over the hips."""
    import numpy as np

    ctx = pipeline.ctx
    model = ctx.model
    labels = list(ctx.body_labels)
    children = model.joint_child.numpy()
    starts = model.joint_q_start.numpy()
    seeds = {
        "left_hip_pitch_link": 0.2,
        "right_hip_pitch_link": -0.2,
        "left_knee_link": -0.35,
        "right_knee_link": right_knee,
    }
    current = model.joint_q.numpy()
    updated = np.array(current, copy=True)
    flat = updated.reshape(-1)
    for joint_index, child in enumerate(children):
        child_index = int(child)
        if child_index < 0 or child_index >= len(labels):
            continue
        value = seeds.get(labels[child_index])
        if value is None:
            continue
        flat[int(starts[joint_index])] = value
    model.joint_q.assign(updated)


def _keep_feet_with_body(pipeline, *, ballet: bool = False) -> None:
    """Stop hip yaw from turning the toes away from the body.

    Ankle position can be reached with the thigh yaw near zero. Leaving that
    coordinate free lets each foot settle pointing sideways or backward.
    """
    import numpy as np

    model = pipeline.ctx.model
    labels = list(pipeline.ctx.body_labels)
    children = model.joint_child.numpy()
    dof_starts = model.joint_qd_start.numpy()
    lower = np.array(model.joint_limit_lower.numpy(), copy=True)
    upper = np.array(model.joint_limit_upper.numpy(), copy=True)
    bounds = {
        # Legs hang off the pelvis. Waist yaw turns only the chest, so a
        # large waist yaw leaves the feet facing a different way from the body.
        "waist_yaw_link": (-0.35, 0.35),
        "left_hip_yaw_link": (-0.4, 0.4),
        "right_hip_yaw_link": (-0.4, 0.4),
    }
    if ballet:
        # A collapsed elbow or a twisted shoulder keeps the hands out of the
        # overhead reach. These limits leave the raise, the side reach, and
        # the palm-forward wrist free.
        bounds.update({
            "left_elbow_link": (-0.85, 1.24),
            "right_elbow_link": (-1.24, 0.85),
            "left_shoulder_yaw_link": (-0.8, 0.8),
            "right_shoulder_yaw_link": (-0.8, 0.8),
        })
    for joint_index, child in enumerate(children):
        child_index = int(child)
        if child_index < 0 or child_index >= len(labels):
            continue
        pair = bounds.get(labels[child_index])
        if pair is None:
            continue
        dof = int(dof_starts[joint_index])
        lower[dof] = pair[0]
        upper[dof] = pair[1]
    model.joint_limit_lower.assign(lower)
    model.joint_limit_upper.assign(upper)


def _wrap_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def facing_flip_mask(headings_deg) -> object:
    """Mark frames that face the opposite way from the start of the clip.

    GVHMR sometimes swaps front and back for a run of frames, including a
    few in-between headings, so a one-frame continuity check still follows
    the swap. Each frame keeps the yaw closer to the opening facing.
    """
    import numpy as np

    headings = np.asarray(headings_deg, dtype=np.float64).reshape(-1)
    flip = np.zeros(headings.shape[0], dtype=bool)
    if headings.size == 0:
        return flip
    origin = float(headings[0])
    for index, heading in enumerate(headings):
        stay = abs(_wrap_deg(float(heading) - origin))
        turned = abs(_wrap_deg(float(heading) + 180.0 - origin))
        flip[index] = turned < stay
    return flip


def heading_spike_deltas(headings_deg, window: int = 7, limit_deg: float = 45.0):
    """Rotation, in degrees, that pulls a one-frame facing spike back to its neighbors."""
    import numpy as np

    headings = np.asarray(headings_deg, dtype=np.float64).reshape(-1)
    deltas = np.zeros(headings.shape[0], dtype=np.float64)
    half = max(int(window) // 2, 1)
    for index in range(headings.shape[0]):
        lo = max(0, index - half)
        hi = min(headings.shape[0], index + half + 1)
        sample = np.concatenate([headings[lo:index], headings[index + 1:hi]])
        if sample.size == 0:
            continue
        ang = np.deg2rad(sample)
        base = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
        shifted = (ang - base + np.pi) % (2.0 * np.pi) - np.pi
        center = base + float(np.median(shifted))
        delta = _wrap_deg(float(headings[index]) - float(np.rad2deg(center)))
        if abs(delta) > limit_deg:
            deltas[index] = -delta
    return deltas


def _rotate_frame_about_pelvis(positions, quaternions, index, pelvis, degrees, Q) -> None:
    import numpy as np

    half = np.deg2rad(degrees) * 0.5
    yaw = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)
    origin = positions[index, pelvis].copy()
    delta = positions[index] - origin
    positions[index] = origin + Q.rotate(np.broadcast_to(yaw, (delta.shape[0], 4)), delta)
    quaternions[index] = Q.multiply(np.broadcast_to(yaw, quaternions[index].shape), quaternions[index])


def stabilize_facing(motion):
    """Keep the body facing continuous by undoing 180° estimator flips."""
    from dataclasses import replace

    import numpy as np

    from hhtools.core.coord import to_up_axis
    from hhtools.core.math import quaternion as Q

    if str(motion.up_axis).upper() != "Z":
        motion = to_up_axis(motion, "Z")
    names = list(motion.bone_names)
    if not {"left_hip", "right_hip", "pelvis"}.issubset(names):
        return motion
    positions = np.array(motion.positions, copy=True)
    quaternions = np.array(motion.quaternions, copy=True)
    left = names.index("left_hip")
    right = names.index("right_hip")
    pelvis = names.index("pelvis")
    across = positions[:, left, :2] - positions[:, right, :2]
    headings = np.degrees(np.arctan2(-across[:, 0], across[:, 1]))
    mask = facing_flip_mask(headings)
    for index in np.flatnonzero(mask):
        _rotate_frame_about_pelvis(positions, quaternions, int(index), pelvis, 180.0, Q)
    across = positions[:, left, :2] - positions[:, right, :2]
    headings = np.degrees(np.arctan2(-across[:, 0], across[:, 1]))
    for index, degrees in enumerate(heading_spike_deltas(headings)):
        if degrees:
            _rotate_frame_about_pelvis(positions, quaternions, index, pelvis, float(degrees), Q)
    across = positions[:, left, :2] - positions[:, right, :2]
    headings = np.degrees(np.arctan2(-across[:, 0], across[:, 1]))
    # The pelvis quaternion's own forward axis does not stay on this hip
    # line, and that is what turns the robot around. Replace it with a yaw
    # the retargeter reads as this heading.
    theta = np.deg2rad(headings)
    half = 0.5 * theta
    rz = np.zeros((len(theta), 4), dtype=np.float32)
    rz[:, 2] = np.sin(half)
    rz[:, 3] = np.cos(half)
    ry = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)], dtype=np.float32)
    quaternions[:, pelvis] = Q.multiply(rz, np.broadcast_to(ry, rz.shape))
    print(f"stabilized facing on {int(mask.sum())} / {len(mask)} frames", flush=True)
    return replace(motion, positions=positions, quaternions=quaternions, up_axis="Z")


def _clear_position_offsets(scaler):
    """Keep calibration rotations, drop the rest-pose position offsets.

    Those offsets are about a metre long. After GVHMR is rotated Y-up to
    Z-up they no longer point down the limb, so the foot targets sit in
    front of the hips and the legs fold over the waist. The scaled joint
    positions already put the feet under the pelvis.
    """
    from dataclasses import replace

    return replace(
        scaler,
        joint_offsets={
            name: ((0.0, 0.0, 0.0), quat)
            for name, (translation, quat) in scaler.joint_offsets.items()
        },
    )


def ballet_phrase_targets(times):
    """Body-frame targets for the ballet phrase, in metres from the pelvis.

    Forward is the direction the chest faces, left is the dancer's left, and
    up is vertical. Times are seconds. The phrase matches the clip: a small
    left-front step, the right knee lifting backward while both hands rise
    overhead, the right foot stepping out as the hands lower, then the right
    hand opening to the upper right while the left arm and left foot reach
    out, a hook of the left foot, and a return to standing.
    """
    import numpy as np

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    stand = {
        "left_ankle": (0.02, 0.11, -0.95),
        "right_ankle": (0.02, -0.11, -0.95),
        "left_knee": (0.04, 0.12, -0.50),
        "right_knee": (0.04, -0.12, -0.50),
        "left_foot": (0.16, 0.12, -1.00),
        "right_foot": (0.16, -0.12, -1.00),
        "left_wrist": (0.06, 0.24, -0.08),
        "right_wrist": (0.06, -0.24, -0.08),
    }
    poses = [
        (0.00, stand),
        (0.28, stand),
        (0.85, {
            **stand,
            "left_ankle": (0.16, 0.28, -0.94),
            "left_knee": (0.10, 0.20, -0.48),
            "left_foot": (0.30, 0.30, -0.99),
        }),
        (1.35, {
            **stand,
            "left_ankle": (0.08, 0.16, -0.95),
            "left_knee": (0.06, 0.14, -0.50),
            "left_foot": (0.20, 0.17, -1.00),
            "right_ankle": (-0.12, -0.12, -0.62),
            "right_knee": (-0.22, -0.12, -0.28),
            "right_foot": (-0.08, -0.12, -0.74),
            "left_wrist": (0.24, 0.30, 0.16),
            "right_wrist": (0.24, -0.30, 0.16),
        }),
        (2.15, {
            **stand,
            "right_ankle": (-0.22, -0.10, -0.48),
            "right_knee": (-0.38, -0.10, -0.12),
            "right_foot": (-0.16, -0.10, -0.60),
            "left_wrist": (0.12, 0.20, 0.64),
            "right_wrist": (0.12, -0.20, 0.64),
        }),
        (2.85, {
            **stand,
            "right_ankle": (0.18, -0.28, -0.94),
            "right_knee": (0.10, -0.20, -0.48),
            "right_foot": (0.30, -0.30, -0.99),
            "left_wrist": (0.16, 0.28, 0.02),
            "right_wrist": (0.16, -0.28, 0.02),
        }),
        (3.05, {
            **stand,
            "right_ankle": (0.08, -0.18, -0.95),
            "right_knee": (0.06, -0.16, -0.50),
            "right_foot": (0.20, -0.18, -1.00),
            "left_wrist": (0.06, 0.30, -0.02),
            "right_wrist": (0.06, -0.30, -0.02),
        }),
        (3.40, {
            **stand,
            "left_ankle": (0.06, 0.38, -0.70),
            "left_knee": (0.04, 0.24, -0.40),
            "left_foot": (0.14, 0.42, -0.66),
            "right_ankle": (0.02, -0.12, -0.95),
            "right_knee": (0.04, -0.12, -0.50),
            "right_foot": (0.14, -0.12, -1.00),
            "left_wrist": (0.08, 0.52, 0.20),
            "right_wrist": (0.08, -0.42, 0.58),
        }),
        (3.62, {
            **stand,
            "left_ankle": (0.06, 0.36, -0.72),
            "left_knee": (0.04, 0.22, -0.42),
            "left_foot": (0.00, 0.34, -0.58),
            "left_wrist": (0.06, 0.40, 0.08),
            "right_wrist": (0.04, -0.28, 0.20),
        }),
        (3.85, stand),
        (4.00, stand),
    ]
    names = list(stand)
    curves = {name: [] for name in names}
    for stamp, pose in poses:
        for name in names:
            curves[name].append((stamp, *pose[name]))
    result = {}
    for name, samples in curves.items():
        sample_t = np.array([row[0] for row in samples], dtype=np.float64)
        values = np.array([row[1:] for row in samples], dtype=np.float64)
        result[name] = np.column_stack([
            np.interp(times, sample_t, values[:, axis]) for axis in range(3)
        ])
    return result


def shape_ballet(motion):
    """Move the estimated limbs onto the ballet phrase before IK."""
    from dataclasses import replace

    import numpy as np

    names = list(motion.bone_names)
    needed = {
        "pelvis", "left_hip", "right_hip",
        "left_ankle", "right_ankle", "left_knee", "right_knee",
        "left_foot", "right_foot", "left_wrist", "right_wrist",
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    }
    if not needed.issubset(names):
        return motion
    positions = np.array(motion.positions, copy=True)
    count = positions.shape[0]
    rate = float(motion.framerate or 30.0)
    times = np.arange(count, dtype=np.float64) / rate
    targets = ballet_phrase_targets(times)
    index = {name: names.index(name) for name in needed}
    for frame in range(count):
        pelvis = positions[frame, index["pelvis"]]
        # The solved RP1 root stays near identity for this clip, so the
        # phrase axes are the robot axes: +X forward, +Y the dancer's left.
        forward = np.array([1.0, 0.0])
        left = np.array([0.0, 1.0])
        for name, curve in targets.items():
            forward_m, left_m, up_m = curve[frame]
            positions[frame, index[name], 0] = pelvis[0] + forward[0] * forward_m + left[0] * left_m
            positions[frame, index[name], 1] = pelvis[1] + forward[1] * forward_m + left[1] * left_m
            positions[frame, index[name], 2] = pelvis[2] + up_m
        for side in ("left", "right"):
            shoulder = positions[frame, index[f"{side}_shoulder"]]
            wrist = positions[frame, index[f"{side}_wrist"]]
            elbow = shoulder + 0.55 * (wrist - shoulder)
            elbow[0] -= forward[0] * 0.05
            elbow[1] -= forward[1] * 0.05
            positions[frame, index[f"{side}_elbow"]] = elbow
    print(f"shaped ballet phrase across {count} frames", flush=True)
    return replace(motion, positions=positions)


def _finish_ballet_csv(csv_path: Path) -> None:
    """Keep the raised palms from twisting, and hook the left foot at the end."""
    import csv

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    notes = [line for line in lines if line.startswith("#")]
    table = [line for line in lines if line and not line.startswith("#")]
    reader = csv.DictReader(table)
    rows = list(reader)
    if not rows or reader.fieldnames is None:
        return
    header = list(reader.fieldnames)

    def blend(row, column, target, amount):
        if column not in row or amount <= 0:
            return
        current = float(row[column])
        row[column] = f"{current + (target - current) * amount:.5f}"

    def window(time, start, peak0, peak1, end):
        if time <= start or time >= end:
            return 0.0
        if time < peak0:
            return (time - start) / max(peak0 - start, 1e-6)
        if time > peak1:
            return (end - time) / max(end - peak1, 1e-6)
        return 1.0

    standing = dict(rows[0])
    for row in rows:
        time = float(row["time"])
        raise_amount = max(0.0, min(1.0, (time - 1.1) / 0.3, (2.7 - time) / 0.3))
        blend(row, "dof_left_wrist_joint", 0.0, raise_amount)
        blend(row, "dof_right_wrist_joint", 0.0, raise_amount)
        # The position solve reaches backward with a straight leg. Bend the
        # right hip and knee so the knee itself lifts behind the body.
        lift = window(time, 1.20, 1.65, 2.25, 2.65)
        blend(row, "dof_right_hip_pitch_joint", -1.35, lift)
        blend(row, "dof_right_knee_joint", -1.55, lift)
        side = window(time, 3.15, 3.35, 3.52, 3.68)
        blend(row, "dof_right_shoulder_pitch_joint", -2.2, side)
        blend(row, "dof_right_shoulder_roll_joint", -0.35, side)
        blend(row, "dof_right_elbow_joint", 0.65, side)
        settle = window(time, 3.72, 3.92, 4.20, 4.40)
        for column in (
            "dof_left_shoulder_pitch_joint", "dof_left_shoulder_roll_joint", "dof_left_shoulder_yaw_joint",
            "dof_left_elbow_joint", "dof_left_wrist_joint",
            "dof_right_shoulder_pitch_joint", "dof_right_shoulder_roll_joint", "dof_right_shoulder_yaw_joint",
            "dof_right_elbow_joint", "dof_right_wrist_joint",
            "dof_right_hip_pitch_joint", "dof_right_knee_joint",
            "dof_left_hip_pitch_joint", "dof_left_knee_joint",
        ):
            blend(row, column, float(standing[column]), settle)
        if 3.48 <= time <= 3.78:
            if time <= 3.64:
                amount = min(1.0, (time - 3.48) / 0.08)
            else:
                amount = max(0.0, (3.78 - time) / 0.14)
            blend(row, "dof_left_ankle_pitch_joint", -0.70, amount)
    body = [",".join(header)]
    body.extend(",".join(row[column] for column in header) for row in rows)
    csv_path.write_text("\n".join([*notes, *body]) + "\n", encoding="utf-8")


def retarget_upright(npz_path: Path, csv_path: Path, *, reference: str, phrase: str = "") -> None:
    """Retarget one clip while keeping the RP1 pelvis upright.

    GVHMR is Y-up. Copying that root quaternion onto the Z-up URDF lays
    the robot on its side, so the pelvis target is reduced to yaw. The
    calibration position offsets are cleared because they point the feet
    forward once the clip is converted to Z-up.
    """
    from dataclasses import replace

    from hhtools.io import load_motion
    from hhtools.io.export_bundle import bake_export_root_z
    from hhtools.io.robot_csv import save_robot_csv
    from hhtools.retarget.calibration import load_calibration, resolve_preset_calibration_file
    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.retarget_profile import (
        build_pipeline_config_for_preset,
        build_scaler_config_for_robot,
    )

    configure_warp_cache()
    preset = get_preset("rp1")
    robot = load_robot(preset)
    calibration_path = resolve_preset_calibration_file(preset, reference)
    if calibration_path is None:
        raise SystemExit(f"No RP1 calibration for reference {reference}")
    motion = stabilize_facing(load_motion(npz_path))
    if phrase == "ballet":
        motion = shape_ballet(motion)
    calibration = load_calibration(calibration_path)
    scaler = _clear_position_offsets(
        build_scaler_config_for_robot(calibration, robot, motion, human_height=1.7)
    )
    # The chest is pulled toward the pelvis so it faces the same way as the
    # feet. Hip yaw and ankle roll stay near the middle of their limits.
    masks = {
        "waist_yaw_link": 1.0,
        "waist_roll_link": 0.4,
        "left_hip_yaw_link": 1.0,
        "right_hip_yaw_link": 1.0,
        "left_ankle_roll_link": 1.0,
        "right_ankle_roll_link": 1.0,
    }
    if phrase == "ballet":
        masks.update({
            "left_elbow_link": 0.55,
            "right_elbow_link": 0.55,
            "left_shoulder_yaw_link": 0.7,
            "right_shoulder_yaw_link": 0.7,
        })
    pipeline_cfg = replace(
        build_pipeline_config_for_preset(preset, reference, ik_iterations=24),
        disable_rotation_objectives=True,
        pelvis_yaw_only_rotation_target=True,
        smooth_joint_filter_masks=masks,
    )
    if phrase == "ballet":
        # Foot planting pulls the raised knee back to the floor. This phrase
        # lifts the right knee and, later, the left foot.
        pipeline_cfg = replace(
            pipeline_cfg,
            apply_feet_stabilizer=False,
            post_ik_foot_clamps=False,
        )
    pipeline = NewtonBasicPipeline(
        robot,
        scaler_config=scaler,
        pipeline_config=pipeline_cfg,
        human_height=1.7,
        configure_warp=False,
    )
    _seed_legs_down(pipeline, right_knee=-0.35 if phrase == "ballet" else 0.35)
    _keep_feet_with_body(pipeline, ballet=phrase == "ballet")
    retargeted = pipeline.run(motion)
    joint_q, lift = bake_export_root_z(robot, retargeted, source_motion=motion)
    meta = {
        "source": str(npz_path.resolve()),
        "source_format": motion.meta.get("source_format", ""),
        "ik_iterations": "24",
        "pelvis": "yaw_only",
        **{key: str(value) for key, value in retargeted.meta.items()},
    }
    if abs(lift) > 1e-12:
        meta["playback_mesh_z_lift"] = f"{lift:.6f}"
    save_robot_csv(
        csv_path,
        robot=robot,
        joint_q=joint_q,
        sample_rate=retargeted.sample_rate,
        meta=meta,
    )
    if phrase == "ballet":
        _finish_ballet_csv(csv_path)
    print(f"retarget upright {retargeted.num_frames} frames -> {csv_path}", flush=True)


def gvhmr_sequence_id(results: Path) -> str:
    """HHTools looks up ``root / sequence`` as a file, so the id keeps ``.pt``."""
    return results.name


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
    # GVHMR sequence ids are the result filename, including ``.pt``.
    # The stem alone makes HHTools look for a file that does not exist.
    run([
        uv, "run", "hhtools", "import", "run",
        "--dataset", "gvhmr",
        "--root", str(results.parent),
        "--sequence", gvhmr_sequence_id(results),
        "--out", str(motions),
    ], cwd=hhtools)
    npzs = sorted(motions.rglob("*.npz"))
    if not npzs:
        raise SystemExit(f"HHTools import wrote no NPZ under {motions}")
    csv_path = out / "rp1.csv"
    run([
        uv, "run", "python", str(ROOT / "scripts" / "bootstrap_rp1_preset.py"),
    ], cwd=hhtools)
    phrase = "ballet" if video_id == "a5R0Xf46hyg" else ""
    retarget_upright(npzs[0], csv_path, reference=args.reference, phrase=phrase)
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
