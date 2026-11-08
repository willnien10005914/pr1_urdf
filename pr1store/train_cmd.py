"""Build the Isaac Lab command for one studio job.

Official flags (--task, --num_envs, --max_iterations, --seed, --logger, --viz)
match Isaac Lab 3's RSL-RL entrypoint. Learning rate and the other PPO fields
are applied by scripts/train_pr1_job.py, which patches the loaded agent config
before training starts. They are not passed as unchecked shell text.
"""
from __future__ import annotations

from .catalog import task_by_id

PHYSICS = {
    "newton_mjwarp": [],
    "isaacsim_physx": ["physics=isaacsim_physx"],
}


def clamp_config(raw: dict) -> dict:
    task = str(raw.get("task") or "RP1-Flat")
    try:
        spec = task_by_id(task)
    except KeyError as error:
        raise ValueError(f"Unknown task {task}") from error
    merged = {**spec["defaults"], **(raw or {})}
    merged["task"] = task

    def integer(name: str, low: int, high: int) -> int:
        value = int(merged[name])
        if value < low or value > high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    def real(name: str, low: float, high: float) -> float:
        value = float(merged[name])
        if not (low <= value <= high):
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    physics = str(merged.get("physics") or "newton_mjwarp")
    if physics not in PHYSICS:
        raise ValueError("physics must be newton_mjwarp or isaacsim_physx")
    clip_id = str(raw.get("clip_id") or "")
    if clip_id and not clip_id.isalnum():
        raise ValueError("clip_id must be a job id")
    return {
        "task": task,
        "physics": physics,
        "num_envs": integer("num_envs", 1, 8192),
        "max_iterations": integer("max_iterations", 1, 30000),
        "learning_rate": real("learning_rate", 1e-6, 1e-1),
        "entropy_coef": real("entropy_coef", 0.0, 0.1),
        "num_steps_per_env": integer("num_steps_per_env", 1, 256),
        "clip_param": real("clip_param", 0.01, 0.5),
        "gamma": real("gamma", 0.9, 0.999),
        "seed": integer("seed", 0, 2**31 - 1),
        "practice": bool(raw.get("practice")),
        "clip_id": clip_id,
    }


def train_argv(config: dict) -> list[str]:
    """Arguments after the interpreter and scripts/train_pr1_job.py."""
    args = [
        f"--task={config['task']}",
        "--viz=none",
        "--logger=tensorboard",
        f"--num_envs={config['num_envs']}",
        f"--max_iterations={config['max_iterations']}",
        f"--seed={config['seed']}",
        *PHYSICS[config["physics"]],
    ]
    return args
