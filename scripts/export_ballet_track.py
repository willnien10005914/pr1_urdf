#!/usr/bin/env python3
"""Write the saved ballet poses as an HHTools CSV with the feet on the ground.

Run this with the roboparty_train interpreter. Root height is solved with the
same MuJoCo asset BeyondMimic uses, so the lowest foot geom sits just above z=0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path.home() / "projects" / "roboparty_train"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN / "robolab" / "scripts" / "tools" / "hhtools"))

from convert_motion import load_motion, load_robot  # noqa: E402
from pr1store.ballet_steps import hhtools_csv, load_document  # noqa: E402


def _lowest_foot(model, data, foot_geoms: list[int]) -> float:
    lowest = min(float(data.geom_xpos[geom, 2] - model.geom_rbound[geom]) for geom in foot_geoms)
    return lowest


def plant(csv_path: Path) -> None:
    times, joint_q, names, _fps = load_motion(csv_path)
    model, joint_names, _bodies = load_robot("rp1")
    missing = set(joint_names) - set(names)
    if missing:
        raise SystemExit(f"Saved poses are missing joints: {sorted(missing)}")
    joint_ids = np.asarray([model.joint(name).id for name in joint_names])
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    root = int(model.jnt_qposadr[free_joints[0]])
    addresses = model.jnt_qposadr[joint_ids]
    columns = [7 + names.index(name) for name in joint_names]
    foot_bodies = {model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")}
    foot_geoms = [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) in foot_bodies]
    if not foot_geoms:
        raise SystemExit("The RP1 asset has no foot geoms to plant.")
    data = mujoco.MjData(model)
    lifts = np.empty(len(joint_q))
    for frame in range(len(joint_q)):
        data.qpos[root : root + 3] = joint_q[frame, :3]
        data.qpos[root + 3 : root + 7] = joint_q[frame, [6, 3, 4, 5]]
        data.qpos[addresses] = joint_q[frame, columns]
        mujoco.mj_kinematics(model, data)
        lifts[frame] = 0.008 - _lowest_foot(model, data, foot_geoms)
    joint_q[:, 2] += lifts
    header = [
        "time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw",
        *[f"dof_{name}" for name in names],
    ]
    lines = ["# sample_rate: 30"]
    lines.append(",".join(header))
    ordered = np.column_stack([times, joint_q])
    for row in ordered:
        lines.append(",".join(f"{value:.6f}" for value in row))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Planted {csv_path}: {len(joint_q)} frames, "
        f"root z {joint_q[:, 2].min():.3f}–{joint_q[:, 2].max():.3f} m"
    )


def main() -> None:
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(hhtools_csv(load_document(ROOT / ".data")), encoding="utf-8")
    plant(destination)


if __name__ == "__main__":
    main()
