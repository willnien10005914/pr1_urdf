"""Ballet PPO: learn the eight saved poses, starting from the opening stand.

The action is a small residual around the reference joint pose. The reference
waits on a pose while the body is settling, then blends to the next saved pose.
Standing on the current pose opens the next one early. If a pose is never
balanced, the clock still moves on, so later poses cannot be skipped by camping
on the first step. Support is measured on the robot's own feet.
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from robolab.tasks.manager_based.beyondmimic.agents.rp1_beyondmimic_agent_cfg import (
    RP1BeyondMimicPPORunnerCfg,
)
from robolab.tasks.manager_based.beyondmimic import mdp
from robolab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand
from robolab.tasks.manager_based.beyondmimic.rp1_beyondmimic_env_cfg import (
    RP1BeyondMimicEnvCfg,
    RP1BeyondMimicObservationsCfg,
)

from pr1store.ballet_steps import phrase_hold_frames
from pr1store.preview_gif import write_learning_gif

HOLD_READY_STEPS = 20  # 0.40 s balanced, then start the next pose
HOLD_LIMIT_STEPS = 80  # 1.60 s, then show the next pose even if it is not settled
BLEND_STEPS = 50  # 1.00 s smoothstep between saved poses

STICK = (
    "base_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_link", "right_wrist_link",
)
PATHS = ("left_wrist_link", "right_wrist_link", "left_ankle_roll_link", "right_ankle_roll_link")


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _segment_distance(point, start, end) -> torch.Tensor:
    delta = end - start
    length = delta.pow(2).sum(dim=-1).clamp_min(1e-8)
    weight = ((point - start) * delta).sum(dim=-1) / length
    closest = start + weight.clamp(0, 1).unsqueeze(-1) * delta
    return torch.linalg.norm(point - closest, dim=-1)


class BalletMotionCommand(MotionCommand):
    """Walk the eight saved poses in order, from the opening stand to the close."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        count = self.num_envs
        self.holding = torch.ones(count, dtype=torch.bool, device=self.device)
        self.ready = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.phase = torch.zeros(count, dtype=torch.long, device=self.device)
        self.blending = torch.zeros(count, dtype=torch.bool, device=self.device)
        self.timer = torch.zeros(count, dtype=torch.long, device=self.device)
        self.settled = torch.zeros(count, dtype=torch.long, device=self.device)
        self.phrase_fraction = torch.zeros(count, device=self.device)
        self._pose_frames = torch.tensor(
            phrase_hold_frames(int(self.motion.time_step_total)), dtype=torch.long, device=self.device,
        )
        self.metrics["phrase_progress"] = torch.zeros(count, device=self.device)
        self.metrics["pose_index"] = torch.zeros(count, device=self.device)
        self._preview_frames: deque = deque(maxlen=28)
        self._control_steps = 0
        self._foot_ids: tuple[int, int] | None = None
        self._robot_foot_ids: tuple[int, int] | None = None

    def _velocities(self, value: torch.Tensor) -> torch.Tensor:
        holding = getattr(self, "holding", None)
        if holding is None:
            return value
        shape = (holding.shape[0],) + (1,) * (value.ndim - 1)
        return torch.where(holding.view(shape), torch.zeros_like(value), value)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._velocities(super().joint_vel)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._velocities(super().body_lin_vel_w)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._velocities(super().body_ang_vel_w)

    def _adaptive_sampling(self, env_ids):
        self.phase[env_ids] = 0
        self.blending[env_ids] = False
        self.timer[env_ids] = 0
        self.settled[env_ids] = 0
        self.phrase_fraction[env_ids] = 0
        self.time_steps[env_ids] = int(self._pose_frames[0])
        self.metrics["sampling_entropy"][:] = 0
        self.metrics["sampling_top1_prob"][:] = 1
        self.metrics["sampling_top1_bin"][:] = 0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if not isinstance(env_ids, slice) and len(env_ids) == 0:
            return
        self.holding[env_ids] = True
        self._follow_reference()

    def _foot_indexes(self) -> tuple[int, int]:
        if self._foot_ids is None:
            sensor = self._env.scene.sensors["contact_forces"]
            indexes, _names = sensor.find_bodies(
                ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True,
            )
            self._foot_ids = (int(indexes[0]), int(indexes[1]))
        return self._foot_ids

    def _robot_feet(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._robot_foot_ids is None:
            names = list(self.robot.body_names)
            self._robot_foot_ids = (
                names.index("left_ankle_roll_link"),
                names.index("right_ankle_roll_link"),
            )
        left_i, right_i = self._robot_foot_ids
        positions = self.robot.data.body_pos_w.torch
        return positions[:, left_i], positions[:, right_i]

    def _balance(self) -> dict[str, torch.Tensor]:
        left_i = self.cfg.body_names.index("left_ankle_roll_link")
        right_i = self.cfg.body_names.index("right_ankle_roll_link")
        # Which foot the current pose asks to lift comes from the reference.
        # Where the pelvis sits is measured on the robot's own feet.
        ref_left_z = self.body_pos_w[:, left_i, 2]
        ref_right_z = self.body_pos_w[:, right_i, 2]
        lifted_left = ref_left_z > ref_right_z + 0.04
        lifted_right = ref_right_z > ref_left_z + 0.04
        left_pos, right_pos = self._robot_feet()
        sensor = self._env.scene.sensors["contact_forces"]
        left_id, right_id = self._foot_indexes()
        normal = sensor.data.net_normal_forces_w_history.torch.norm(dim=-1).amax(dim=1)
        left_contact = normal[:, left_id] > 10.0
        right_contact = normal[:, right_id] > 10.0
        contact_ok = (
            (lifted_left & right_contact)
            | (lifted_right & left_contact)
            | (~lifted_left & ~lifted_right & (left_contact | right_contact))
        )
        pelvis = self.robot.data.root_pos_w.torch[:, :2]
        both = ~lifted_left & ~lifted_right
        distance = torch.where(
            both,
            _segment_distance(pelvis, left_pos[:, :2], right_pos[:, :2]),
            torch.where(
                lifted_left,
                torch.linalg.norm(pelvis - right_pos[:, :2], dim=-1),
                torch.linalg.norm(pelvis - left_pos[:, :2], dim=-1),
            ),
        )
        supported = contact_ok & (distance < torch.where(both, 0.22, 0.18))
        gravity = self.robot.data.projected_gravity_b.torch
        upright = gravity[:, 2] < -0.85
        joint_error = torch.mean(torch.abs(self.robot.data.joint_pos.torch - self.joint_pos), dim=-1)
        heading = torch.abs(_wrap(
            _yaw_xyzw(self.robot.data.root_quat_w.torch) - _yaw_xyzw(self.anchor_quat_w)
        ))
        facing = heading < 0.55
        arrived = joint_error < 0.35
        ready = upright & facing & arrived & supported
        return {
            "ready": ready,
            "upright": upright.float(),
            "facing": facing.float(),
            "supported": supported.float(),
            "distance": distance,
            "joint_error": joint_error,
            "heading": heading,
            "left_stance": (~lifted_left).float(),
            "right_stance": (~lifted_right).float(),
            "left_contact": left_contact.float(),
            "right_contact": right_contact.float(),
        }

    def support_observation(self) -> torch.Tensor:
        state = self._balance()
        return torch.stack((
            state["left_stance"], state["right_stance"],
            state["distance"], state["heading"],
            state["left_contact"], state["right_contact"],
        ), dim=-1)

    def _follow_reference(self) -> None:
        manager = getattr(self._env, "action_manager", None)
        if manager is None:
            return
        term = manager.get_term("joint_pos")
        reference = self.joint_pos
        ids = term._joint_ids
        if not isinstance(ids, slice):
            reference = reference[:, ids]
        term._offset[:] = reference

    def _remember_preview(self) -> None:
        count = min(3, self.num_envs)
        origins = self._env.scene.env_origins[:count]
        positions = self.robot.data.body_pos_w.torch[:count] - origins[:, None, :]
        names = list(self.robot.body_names)
        indexes = [names.index(name) for name in STICK if name in names]
        chosen = [name for name in STICK if name in names]
        snapshot = []
        cpu = positions[:, indexes].detach().cpu()
        for robot in range(count):
            snapshot.append({name: tuple(cpu[robot, column].tolist()) for column, name in enumerate(chosen)})
        self._preview_frames.append(snapshot)

    def _target_paths(self) -> list[dict[str, list[tuple[float, float, float]]]]:
        count = min(3, self.num_envs)
        names = list(self.cfg.body_names)
        columns = [names.index(name) for name in PATHS if name in names]
        labels = [name for name in PATHS if name in names]
        paths = []
        for robot in range(count):
            start = int(self.time_steps[robot])
            stop = min(self.motion.time_step_total, start + 50)
            future = self.motion.body_pos_w[start:stop][:, columns].detach().cpu()
            pose = {}
            for column, name in enumerate(labels):
                pose[name] = [tuple(future[frame, column].tolist()) for frame in range(len(future))]
            paths.append(pose)
        return paths

    def _write_preview(self) -> None:
        directory = os.environ.get("PR1_PREVIEW_DIR", "")
        if not directory or len(self._preview_frames) < 4:
            return
        total = int(os.environ.get("PR1_MAX_ITERATIONS", "0"))
        steps = int(os.environ.get("PR1_STEPS_PER_ENV", "24"))
        iteration = self._control_steps // max(steps, 1)
        write_learning_gif(
            Path(directory) / "preview.gif",
            list(self._preview_frames),
            self._target_paths(),
            iteration=iteration,
            total=total,
        )

    def _update_command(self):
        state = self._balance()
        self.ready = state["ready"]
        last = int(self._pose_frames.shape[0]) - 1
        was_holding = ~self.blending
        not_last = self.phase < last
        self.timer = torch.where(not_last, self.timer + 1, self.timer)
        self.settled = torch.where(
            was_holding & self.ready & not_last, self.settled + 1,
            torch.where(was_holding, torch.zeros_like(self.settled), self.settled),
        )
        start_blend = was_holding & not_last & (
            (self.settled >= HOLD_READY_STEPS) | (self.timer >= HOLD_LIMIT_STEPS)
        )
        self.blending = torch.where(start_blend, torch.ones_like(self.blending), self.blending)
        self.timer = torch.where(start_blend, torch.zeros_like(self.timer), self.timer)
        self.settled = torch.where(start_blend, torch.zeros_like(self.settled), self.settled)
        end_blend = self.blending & ~start_blend & (self.timer >= BLEND_STEPS)
        self.phase = torch.where(end_blend, (self.phase + 1).clamp(max=last), self.phase)
        self.blending = torch.where(end_blend, torch.zeros_like(self.blending), self.blending)
        self.timer = torch.where(end_blend, torch.zeros_like(self.timer), self.timer)
        self.settled = torch.where(end_blend, torch.zeros_like(self.settled), self.settled)
        start = self._pose_frames[self.phase]
        nxt = self._pose_frames[(self.phase + 1).clamp(max=last)]
        alpha = torch.where(
            self.blending,
            (self.timer.float() / BLEND_STEPS).clamp(0, 1),
            torch.zeros(self.num_envs, device=self.device),
        )
        curved = alpha * alpha * (3 - 2 * alpha)
        frame = (start.float() + (nxt.float() - start.float()) * curved).round().long()
        self.time_steps = frame.clamp(0, int(self.motion.time_step_total) - 1)
        self.holding = ~self.blending
        self.phrase_fraction = (self.phase.float() + curved) / last
        self.motion_ended = self.phase >= last
        self.metrics["phrase_progress"][:] = self.phrase_fraction
        self.metrics["pose_index"][:] = self.phase.float()
        self.refresh_body_reference()
        self._follow_reference()
        self._control_steps += 1
        try:
            if self._control_steps % 2 == 0:
                self._remember_preview()
            every = int(os.environ.get("PR1_PREVIEW_EVERY", "50"))
            steps = int(os.environ.get("PR1_STEPS_PER_ENV", "24"))
            if every > 0 and self._control_steps % (every * steps) == 0:
                self._write_preview()
        except Exception as error:
            print(f"PR1_PREVIEW {error}", flush=True)


def _command(env, command_name: str) -> BalletMotionCommand:
    command = env.command_manager.get_term(command_name)
    return command


def upright_reward(env, command_name: str) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b.torch[:, 2]
    return torch.exp(-torch.square(gravity + 1.0) / 0.12**2)


def heading_reward(env, command_name: str) -> torch.Tensor:
    state = _command(env, command_name)._balance()
    return torch.exp(-torch.square(state["heading"]) / 0.35**2)


def support_reward(env, command_name: str) -> torch.Tensor:
    state = _command(env, command_name)._balance()
    return state["supported"] * torch.exp(-torch.square(state["distance"]) / 0.12**2)


def pose_when_balanced(env, command_name: str) -> torch.Tensor:
    state = _command(env, command_name)._balance()
    return state["upright"] * torch.exp(-torch.square(state["joint_error"]) / 0.45**2)


def phrase_advance(env, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    state = command._balance()
    matched = state["upright"] * torch.exp(-torch.square(state["joint_error"]) / 0.45**2) * state["supported"]
    return command.phrase_fraction * matched


def support_observation(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).support_observation()


@configclass
class BalletStudentCfg(ObsGroup):
    command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
    support = ObsTerm(func=support_observation, params={"command_name": "motion"})
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1
        self.flatten_history_dim = True


@configclass
class RP1BalletObservationsCfg(RP1BeyondMimicObservationsCfg):
    student: BalletStudentCfg = BalletStudentCfg()


@configclass
class RP1BalletEnvCfg(RP1BeyondMimicEnvCfg):
    observations: RP1BalletObservationsCfg = RP1BalletObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        motion = os.environ.get("PR1_MOTION_FILE", "")
        if motion:
            self.commands.motion.motion_file = motion
        self.commands.motion.class_type = BalletMotionCommand
        self.commands.motion.reset_on_motion_end = False
        self.actions.joint_pos.scale = 0.15
        self.episode_length_s = 26.0
        self.commands.motion.pose_range = {
            "x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.005, 0.005),
            "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.05, 0.05),
        }
        self.commands.motion.velocity_range = {axis: (-0.02, 0.02) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
        self.commands.motion.joint_position_range = (-0.02, 0.02)
        self.events.randomize_push_robot = None
        self.rewards.motion_body_pos.weight = 1.5
        self.rewards.motion_body_ori.weight = 1.5
        self.rewards.motion_body_lin_vel.weight = 0.15
        self.rewards.motion_body_ang_vel.weight = 0.15
        self.rewards.action_rate_l2.weight = -0.4
        self.rewards.joint_torques_l2.weight = -5.0e-5
        self.rewards.joint_velocity_l2.weight = -4.0e-3
        self.rewards.upright = RewTerm(func=upright_reward, weight=5.0, params={"command_name": "motion"})
        self.rewards.heading = RewTerm(func=heading_reward, weight=2.5, params={"command_name": "motion"})
        self.rewards.support = RewTerm(func=support_reward, weight=4.0, params={"command_name": "motion"})
        self.rewards.balanced_pose = RewTerm(func=pose_when_balanced, weight=2.5, params={"command_name": "motion"})
        self.rewards.phrase = RewTerm(func=phrase_advance, weight=3.5, params={"command_name": "motion"})
        self.terminations.ee_body_pos = None
        self.terminations.anchor_pos.params["threshold"] = 0.45
        self.terminations.anchor_ori.params["threshold"] = 1.15


@configclass
class RP1BalletPPORunnerCfg(RP1BeyondMimicPPORunnerCfg):
    max_iterations = 12000
    save_interval = 500
    experiment_name = "rp1_ballet_balance"
    obs_groups = {"actor": ["student"], "critic": ["policy", "critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.12),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.003, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0, normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
    )


def register() -> None:
    try:
        gym.spec("RP1-Ballet")
        return
    except gym.error.NameNotFound:
        pass
    gym.register(
        id="RP1-Ballet",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "pr1store.ballet_task:RP1BalletEnvCfg",
            "rsl_rl_cfg_entry_point": "pr1store.ballet_task:RP1BalletPPORunnerCfg",
        },
    )
