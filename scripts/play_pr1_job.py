#!/usr/bin/env python3
"""Roll out one trained RP1 policy and save the motion the studio viewer plays.

Run with the roboparty_train interpreter. PR1_MOTION_FILE is the BeyondMimic
clip, and PR1_PLAY_OUTPUT is the motion.json path. Remaining arguments are the
Isaac Lab play flags, including --checkpoint.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _as_tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _snapshot(robot, names: list[str]) -> tuple[list[float], list[float]]:
    from pr1store.view_math import viewer_root

    position = _as_tensor(robot.data.root_pos_w)[0].detach().cpu().tolist()
    quaternion = _as_tensor(robot.data.root_quat_w)[0].detach().cpu().tolist()
    joints = _as_tensor(robot.data.joint_pos)[0].detach().cpu().tolist()
    joint_names = list(getattr(robot, "joint_names", names))
    by_name = {name: round(float(value), 5) for name, value in zip(joint_names, joints)}
    return viewer_root(position, quaternion), [by_name.get(name, 0.0) for name in names]


def _install_motion_file() -> None:
    import importlib.util

    path = ROOT / "scripts" / "train_pr1_job.py"
    spec = importlib.util.spec_from_file_location("train_pr1_job", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_motion_file()


def main() -> None:
    output = Path(os.environ["PR1_PLAY_OUTPUT"])
    _install_motion_file()
    import robolab.tasks  # noqa: F401
    from pr1store.ballet_task import register

    register()

    import torch
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.utils.seed import configure_seed
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
    from isaaclab_rl.entrypoints.common import (
        add_common_play_args,
        apply_env_overrides,
        close_env,
        create_isaaclab_env,
        set_hydra_args,
    )
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
    import argparse

    parser = argparse.ArgumentParser(description="Record one PPO rollout for the PR1 viewer.")
    add_common_play_args(parser, agent_default="rsl_rl_cfg_entry_point", agent_help="RSL-RL agent configuration.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args_cli, remaining = setup_preset_cli(parser, sys.argv[1:])
    set_hydra_args(remaining)
    installed = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    # Keep the policy running for the whole reference clip. Training ends an
    # episode when a foot or elbow leaves the reference, which would otherwise
    # cut the viewer clip off after the first stumble.
    for name in ("ee_body_pos", "anchor_pos", "anchor_ori"):
        if getattr(env_cfg.terminations, name, None) is not None:
            setattr(env_cfg.terminations, name, None)
    for name in (
        "randomize_push_robot", "randomize_rigid_body_material", "randomize_joint_default_pos",
        "add_base_mass", "base_com", "scale_link_mass", "scale_actuator_gains", "scale_joint_parameters",
    ):
        if getattr(env_cfg.events, name, None) is not None:
            setattr(env_cfg.events, name, None)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed)
        apply_env_overrides(args_cli, env_cfg)
        configure_seed(env_cfg.seed if env_cfg.seed is not None else 42, torch_deterministic=True)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(wrapped, agent_cfg)
            runner.load(args_cli.checkpoint)
            policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
            robot = wrapped.unwrapped.scene["robot"]
            names = list(getattr(robot, "joint_names"))
            roots: list[list[float]] = []
            rows: list[list[float]] = []
            obs = wrapped.get_observations()
            root, joints = _snapshot(robot, names)
            roots.append(root)
            rows.append(joints)
            step_dt = float(wrapped.unwrapped.step_dt)
            limit = max(2, int(round(12.0 / step_dt)))
            for _step in range(limit):
                with torch.inference_mode():
                    obs, _reward, dones, _extras = wrapped.step(policy(obs))
                    policy.reset(dones)
                root, joints = _snapshot(robot, names)
                roots.append(root)
                rows.append(joints)
                if bool(dones[0].item()):
                    break
            motion = {
                "id": output.parent.name,
                "title": "芭蕾 PPO",
                "kind": "ppo_rollout",
                "fps": round(1.0 / step_dt, 4),
                "frames": len(rows),
                "joint_names": names,
                "q": rows,
                "root": roots,
                "summary": "Isaac Lab rollout of the trained BeyondMimic PPO policy on the saved ballet poses.",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(motion), encoding="utf-8")
            print(f"Wrote {output}: {len(rows)} frames at {motion['fps']} Hz", flush=True)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
