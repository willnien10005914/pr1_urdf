#!/usr/bin/env python3
"""Train one RP1 task, applying studio PPO overrides to the loaded agent config.

Run this with the roboparty_train environment. The parent process passes
PR1_PPO_OVERRIDES as JSON. Isaac Lab still receives only its own flags.
"""
from __future__ import annotations

import json
import os
import sys


def install_overrides() -> None:
    raw = os.environ.get("PR1_PPO_OVERRIDES", "")
    if not raw:
        return
    overrides = json.loads(raw)
    import isaaclab_rl.entrypoints.backends.cli_args_rsl_rl as cli
    import isaaclab_rl.entrypoints.backends.train_rsl_rl as trainer

    original = cli.update_rsl_rl_cfg

    def update(agent_cfg, args_cli):
        agent_cfg = original(agent_cfg, args_cli)
        algorithm = agent_cfg.algorithm
        if "learning_rate" in overrides:
            algorithm.learning_rate = float(overrides["learning_rate"])
        if "entropy_coef" in overrides:
            algorithm.entropy_coef = float(overrides["entropy_coef"])
        if "clip_param" in overrides:
            algorithm.clip_param = float(overrides["clip_param"])
        if "gamma" in overrides:
            algorithm.gamma = float(overrides["gamma"])
        if "num_steps_per_env" in overrides:
            agent_cfg.num_steps_per_env = int(overrides["num_steps_per_env"])
        print(
            "PR1_STATUS "
            + json.dumps({
                "stage": "training",
                "message": (
                    f"PPO learning_rate={algorithm.learning_rate} "
                    f"entropy_coef={algorithm.entropy_coef} "
                    f"clip_param={algorithm.clip_param} gamma={algorithm.gamma} "
                    f"steps_per_env={agent_cfg.num_steps_per_env}"
                ),
            }),
            flush=True,
        )
        return agent_cfg

    cli.update_rsl_rl_cfg = update
    trainer.cli_args.update_rsl_rl_cfg = update


if __name__ == "__main__":
    install_overrides()
    import robolab.tasks  # noqa: F401
    from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

    run(sys.argv[1:])
