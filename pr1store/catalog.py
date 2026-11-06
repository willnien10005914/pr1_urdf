"""Robot models, official training recipes, and the PR1 learning stack."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = ROOT / "assets" / "rp1_description"
UPSTREAM = "https://github.com/Roboparty/rp1_description"

ROBOTS = (
    {
        "id": "rp1_24dof",
        "file": "urdf/rp1_24dof.urdf",
        "title": "24-DoF body",
        "summary": "The configuration roboparty_train actually trains: 6 DoF per leg, 2-DoF waist, 5-DoF arms.",
    },
    {
        "id": "rp1",
        "file": "urdf/rp1.urdf",
        "title": "28-DoF body",
        "summary": "Same legs and waist, with a 3-axis wrist on each arm. Deployment config exists; the default policy pack is still 24-DoF.",
    },
    {
        "id": "rp1_hand",
        "file": "urdf/rp1_hand.urdf",
        "title": "Body + RP Hand",
        "summary": "Movable hand joints from rp1_hand.urdf. Finger motion is outside the locomotion PPO tasks.",
    },
    {
        "id": "rp1_hand_fixed",
        "file": "urdf/rp1_hand_fixed.urdf",
        "title": "Hands fixed",
        "summary": "Hand links present, finger joints locked. Useful for checking the arm chain without driving every finger.",
    },
)

# Defaults copied from roboparty_train agent configs (Isaac Lab 3 / RSL-RL).
# Practice presets shorten max_iterations and num_envs so a laptop GPU can
# show a real log without committing to the full published budget.
TASKS = (
    {
        "id": "RP1-Flat",
        "title": "Flat walk",
        "family": "velocity PPO",
        "learns": "Commanded walking and standing on flat ground.",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 9001,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-Rough",
        "title": "Rough ground",
        "family": "velocity PPO",
        "learns": "The same velocity policy on uneven terrain.",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 9001,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-AMP",
        "title": "AMP style",
        "family": "adversarial motion prior",
        "learns": "A discriminator reward that pulls the policy toward the AMP clips (walk, stairs, turns).",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 5000,
            "learning_rate": 1.0e-4,
            "entropy_coef": 0.01,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-BeyondMimic",
        "title": "BeyondMimic",
        "family": "motion tracking",
        "learns": "Whole-body tracking of one retargeted clip, such as wave or dance.",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 8000,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-Parkour",
        "title": "Parkour climb",
        "family": "perceptive locomotion",
        "learns": "Stairs, gaps, and obstacles from depth. This is the climb-up task, not a single canned flip.",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 30000,
            "learning_rate": 1.0e-4,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
)

STACK = [
    {
        "step": 1,
        "title": "Video in, body model out",
        "body": "A YouTube interval is cut with yt-dlp and ffmpeg. GVHMR estimates an SMPL or SMPL-X pose sequence. Those body-model weights stay on this machine; they are licensed separately and are not part of this repository.",
    },
    {
        "step": 2,
        "title": "HHTools retarget",
        "body": "human-humanoid-tools runs Newton IK onto the RP1 URDF and writes a robot CSV (time, root pose, dof_*). SMPL-X is the default calibration reference; SMPL is the other shipped choice.",
    },
    {
        "step": 3,
        "title": "Training-format motion",
        "body": "roboparty_train converts that CSV with convert_motion.py: AMP demonstrations at 120 Hz (.pkl) or BeyondMimic tracking clips at 50 Hz (.npz).",
    },
    {
        "step": 4,
        "title": "Isaac Lab 3 PPO",
        "body": "RSL-RL trains RP1-Flat, RP1-Rough, RP1-AMP, RP1-BeyondMimic, or RP1-Parkour. The published commands use MJWarp by default and PhysX when you select it. The web bar reads Learning iteration from the live log.",
    },
    {
        "step": 5,
        "title": "What you can play today",
        "body": "This checkout has the official reference clips (walk, stair climb, wave, dance) and no released model_*.pt. The viewer plays those references on the URDF. A new checkpoint appears only after a worker run finishes.",
    },
]


def upstream_commit() -> str:
    path = DESCRIPTION / "UPSTREAM_COMMIT.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def joint_table(urdf_file: str) -> list[dict]:
    tree = ET.parse(DESCRIPTION / urdf_file)
    rows = []
    for joint in tree.getroot().findall("joint"):
        kind = joint.get("type") or ""
        if kind == "fixed":
            continue
        limit = joint.find("limit")
        lower = upper = None
        if limit is not None:
            if limit.get("lower") is not None:
                lower = float(limit.get("lower"))
            if limit.get("upper") is not None:
                upper = float(limit.get("upper"))
        rows.append({
            "name": joint.get("name"),
            "type": kind,
            "lower": lower,
            "upper": upper,
        })
    return rows


def robot_catalog() -> list[dict]:
    robots = []
    for spec in ROBOTS:
        joints = joint_table(spec["file"])
        robots.append({
            **spec,
            "urdf_url": f"/assets/rp1_description/{spec['file']}",
            "joints": len(joints),
            "joint_names": [row["name"] for row in joints],
        })
    return robots


def task_by_id(task_id: str) -> dict:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)


def public_catalog() -> dict:
    return {
        "robots": [{key: robot[key] for key in ("id", "title", "summary", "urdf_url", "joints")} for robot in robot_catalog()],
        "tasks": [
            {
                "id": task["id"],
                "title": task["title"],
                "family": task["family"],
                "learns": task["learns"],
                "defaults": task["defaults"],
                "practice": {**task["defaults"], **task["practice"]},
            }
            for task in TASKS
        ],
        "stack": STACK,
        "description": {
            "upstream": UPSTREAM,
            "commit": upstream_commit(),
            "note": "24-DoF is the trained body. 30-DoF and 42-DoF in the RP1 overview count a 2-DoF head and two 6-axis hands; those extra modules are not separate URDFs in this snapshot.",
        },
        "flip": {
            "available": False,
            "title": "Backflip / front flip",
            "summary": "RoboParty publishes a front-flip clip on the RP1 overview. The open motion pack in roboparty_train does not include a flip trajectory or a flip checkpoint. Retarget a flip video, then train RP1-BeyondMimic on that CSV.",
        },
    }
