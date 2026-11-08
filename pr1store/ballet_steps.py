"""Editable ballet poses and a replay built from them."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from pr1store.catalog import joint_table

# Middle of each held pose in the replay, from the opening stand through the close.
PHRASE_HOLD_SECONDS = (0.12, 0.83, 1.57, 2.30, 3.03, 3.77, 4.50, 5.23)
PHRASE_DURATION = 5.366667


def phrase_hold_frames(total_frames: int) -> list[int]:
    """Frame index of each saved pose inside a clip of `total_frames` samples."""
    count = len(PHRASE_HOLD_SECONDS)
    if total_frames < 2:
        return [0] * count
    last = total_frames - 1
    frames = []
    previous = -1
    for stamp in PHRASE_HOLD_SECONDS:
        frame = int(round(stamp / PHRASE_DURATION * last))
        frame = min(last, max(previous + 1, frame))
        frames.append(frame)
        previous = frame
    return frames


STEPS = (
    {"id": "stand-start", "title": "始站", "time": 0.0, "keep": "stand",
     "note": "開始前的站立，雙手放下。"},
    {"id": "left-step", "title": "1 左腳左前", "time": 0.80, "keep": "all",
     "note": "左腳向左前方踏出一小步。"},
    {"id": "right-knee", "title": "2 右膝後抬", "time": 2.20, "keep": "legs",
     "note": "右膝向後抬起，小腿垂在身後。"},
    {"id": "arms-up", "title": "3 雙手過頭", "time": 2.20, "keep": "arms",
     "note": "雙手向前再向上舉過頭頂。"},
    {"id": "right-plant", "title": "4 右腳踏下", "time": 2.80, "keep": "all",
     "note": "右腳向右前方踏下，雙手順著放下。"},
    {"id": "sides", "title": "5 向兩側展開", "time": 3.40, "keep": "all",
     "note": "右手到頭的右上方，左手和左腳向左，右腳站穩。"},
    {"id": "hook", "title": "6 左腳勾起", "time": 3.63, "keep": "all",
     "note": "左腳掌勾起。"},
    {"id": "finish", "title": "7 收尾", "time": 4.0, "keep": "stand",
     "note": "收尾回到站立，雙手放下。"},
)

ARM_PREFIXES = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
LEG_PREFIXES = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
BALLET_JOB = "c212cca38f844173aa36e14c5d54bec4"


def _limits() -> dict[str, tuple[float, float]]:
    found = {}
    for row in joint_table("urdf/rp1_24dof.urdf"):
        if row["lower"] is None or row["upper"] is None:
            continue
        found[row["name"]] = (float(row["lower"]), float(row["upper"]))
    return found


def _clamp_pose(joints: dict) -> dict[str, float]:
    limits = _limits()
    if not isinstance(joints, dict):
        raise ValueError("joints must be an object of joint name to radians")
    pose = {}
    for name, pair in limits.items():
        if name not in joints:
            continue
        value = float(joints[name])
        pose[name] = round(min(pair[1], max(pair[0], value)), 5)
    if not pose:
        raise ValueError("No RP1 joint values were included.")
    return pose


def _read_csv_poses(csv_path: Path) -> tuple[dict[str, float], dict[float, dict[str, float]]]:
    lines = [line for line in csv_path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(lines)
    rows = list(reader)
    if not rows or reader.fieldnames is None:
        return {}, {}
    names = [column.removeprefix("dof_") for column in reader.fieldnames if column.startswith("dof_")]

    def pose_at(row: dict) -> dict[str, float]:
        return {name: round(float(row[f"dof_{name}"]), 5) for name in names}

    by_time = {round(float(row["time"]), 2): pose_at(row) for row in rows}
    return pose_at(rows[0]), by_time


def _nearest(by_time: dict[float, dict[str, float]], stamp: float) -> dict[str, float]:
    if not by_time:
        return {}
    key = min(by_time, key=lambda item: abs(item - stamp))
    return dict(by_time[key])


def _masked(pose: dict[str, float], standing: dict[str, float], keep: str) -> dict[str, float]:
    if keep == "all" or not standing:
        return pose
    prefixes = LEG_PREFIXES if keep == "legs" else ARM_PREFIXES
    mixed = dict(standing)
    for name, value in pose.items():
        if name.startswith(prefixes):
            mixed[name] = value
    return mixed


def arms_down_stand() -> dict[str, float]:
    """Straight stand with the arms at the URDF rest pose, beside the body."""
    return _clamp_pose({name: 0.0 for name in _limits()})


def default_steps(data_dir: Path) -> tuple[dict[str, float], list[dict]]:
    csv_path = data_dir / "jobs" / BALLET_JOB / "rp1.csv"
    standing, by_time = ({}, {})
    if csv_path.is_file():
        standing, by_time = _read_csv_poses(csv_path)
    steps = []
    for spec in STEPS:
        if spec["keep"] == "stand":
            pose = arms_down_stand()
        elif by_time:
            pose = _clamp_pose(_masked(_nearest(by_time, spec["time"]), standing, spec["keep"]))
        else:
            pose = {}
        steps.append({
            "id": spec["id"],
            "title": spec["title"],
            "note": spec["note"],
            "saved": False,
            "joints": pose,
            "com": {"x": 0.0, "y": 0.0},
        })
    standing = _clamp_pose(standing) if standing else {}
    return standing, steps


def _store_path(data_dir: Path) -> Path:
    return data_dir / "ballet_steps.json"


def _clamp_com(com: dict | None) -> dict[str, float]:
    raw = com if isinstance(com, dict) else {}

    def axis(name: str) -> float:
        try:
            value = float(raw.get(name, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        return round(min(0.4, max(-0.4, value)), 5)

    return {"x": axis("x"), "y": axis("y")}


def _unpack_saved(pose: dict) -> tuple[dict, dict]:
    """Accept the current record and the earlier flat joint map."""
    joints = pose.get("joints") if isinstance(pose.get("joints"), dict) else pose
    com = pose.get("com") if isinstance(pose.get("com"), dict) else {}
    return joints, com


def load_document(data_dir: Path) -> dict:
    standing, steps = default_steps(data_dir)
    path = _store_path(data_dir)
    saved: dict = {}
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8")).get("steps") or {}
    for step in steps:
        pose = saved.get(step["id"])
        if isinstance(pose, dict) and pose:
            joints, com = _unpack_saved(pose)
            step["joints"] = _clamp_pose(joints)
            step["com"] = _clamp_com(com)
            step["saved"] = True
    return {"stand": standing, "steps": steps}


def save_step(data_dir: Path, step_id: str, joints: dict, com: dict | None = None) -> dict:
    known = {spec["id"] for spec in STEPS}
    if step_id not in known:
        raise KeyError(step_id)
    pose = _clamp_pose(joints)
    path = _store_path(data_dir)
    current = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    steps = current.get("steps") if isinstance(current.get("steps"), dict) else {}
    steps[step_id] = {"joints": pose, "com": _clamp_com(com)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"steps": steps}, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    document = load_document(data_dir)
    return next(step for step in document["steps"] if step["id"] == step_id)


def _blend(start: dict[str, float], end: dict[str, float], amount: float) -> dict[str, float]:
    names = list(dict.fromkeys([*start, *end]))
    curved = amount * amount * (3 - 2 * amount)
    return {
        name: start.get(name, 0.0) + (end.get(name, 0.0) - start.get(name, 0.0)) * curved
        for name in names
    }


def replay_motion(document: dict) -> dict:
    """Play the opening stand, the six phrases, and the closing stand."""
    standing = document.get("stand") or {}
    sequence = []
    centers = []
    for step in document["steps"]:
        sequence.append(step.get("joints") or standing)
        com = step.get("com") or {}
        centers.append((float(com.get("x", 0.0)), float(com.get("y", 0.0))))
    names = list(standing or sequence[1] or _limits())
    hold = 8
    blend = 14
    frames: list[dict[str, float]] = []
    roots: list[list[float]] = []
    for index, pose in enumerate(sequence):
        frames.extend([pose] * hold)
        roots.extend([_root(centers[index])] * hold)
        if index + 1 < len(sequence):
            for step_index in range(1, blend + 1):
                amount = step_index / blend
                frames.append(_blend(pose, sequence[index + 1], amount))
                start = centers[index]
                end = centers[index + 1]
                curved = amount * amount * (3 - 2 * amount)
                roots.append(_root((
                    start[0] + (end[0] - start[0]) * curved,
                    start[1] + (end[1] - start[1]) * curved,
                )))
    return {
        "id": "ballet-steps",
        "title": "芭蕾",
        "kind": "posed",
        "plant_feet": True,
        "fps": 30,
        "frames": len(frames),
        "joint_names": names,
        "q": [[round(frame.get(name, 0.0), 5) for name in names] for frame in frames],
        "root": roots,
        "summary": "Six poses you saved, played in order and returned to standing.",
    }


def _root(com: tuple[float, float]) -> list[float]:
    return [round(com[0], 5), round(com[1], 5), 0.0, 1.0, 0.0, 0.0, 0.0]


def hhtools_csv(document: dict) -> str:
    """HHTools CSV of the saved pose sequence. Root height is filled in later by MuJoCo."""
    motion = replay_motion(document)
    fps = float(motion["fps"])
    names = motion["joint_names"]
    header = [
        "time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw",
        *[f"dof_{name}" for name in names],
    ]
    lines = [f"# sample_rate: {fps:g}", ",".join(header)]
    for index, joints in enumerate(motion["q"]):
        root = motion["root"][index]
        values = [
            f"{index / fps:.6f}",
            f"{root[0]:.6f}", f"{root[1]:.6f}", f"{root[2]:.6f}",
            f"{root[4]:.6f}", f"{root[5]:.6f}", f"{root[6]:.6f}", f"{root[3]:.6f}",
            *[f"{value:.6f}" for value in joints],
        ]
        lines.append(",".join(values))
    return "\n".join(lines) + "\n"
