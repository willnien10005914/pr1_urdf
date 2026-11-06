"""Point HHTools at the vendored 24-DoF URDF and write a T-pose calibration.

The preset is created in the user robot library the first time a retarget job
runs. Body-model weights are not copied.
"""
from __future__ import annotations

import math
from pathlib import Path

from .catalog import DESCRIPTION, joint_table

PRESET_NAME = "rp1"


def user_preset_dir() -> Path:
    return Path.home() / ".config" / "hhtools" / "robots" / PRESET_NAME


def tpose_joint_q() -> dict[str, float]:
    """Shoulder roll near ±90° so an SMPL T-pose does not leave the arms down."""
    limits = {row["name"]: row for row in joint_table("urdf/rp1_24dof.urdf")}
    targets = {
        "left_shoulder_roll_joint": math.pi / 2,
        "right_shoulder_roll_joint": -math.pi / 2,
    }
    posed = {}
    for name, target in targets.items():
        row = limits.get(name)
        if row is None or row["lower"] is None or row["upper"] is None:
            posed[name] = round(target, 6)
            continue
        posed[name] = round(min(row["upper"], max(row["lower"], target)), 6)
    return posed


def calibration_text(reference: str) -> str:
    lines = [
        f"robot: {PRESET_NAME}",
        f"reference: {reference}",
        "calibrated_joint_q:",
    ]
    for name, value in tpose_joint_q().items():
        lines.append(f"  {name}: {value}")
    lines.append("notes: >-")
    lines.append("  Bootstrap T-pose for the vendored RP1 24-DoF URDF.")
    lines.append("  Shoulder roll is set near a T-pose. Refine this in HHTools before treating the motion as final.")
    return "\n".join(lines) + "\n"


def link_urdf(destination: Path | None = None) -> Path:
    """Symlink the studio URDF into the HHTools user library. Does not import hhtools."""
    destination = destination or user_preset_dir()
    destination.mkdir(parents=True, exist_ok=True)
    urdf = (DESCRIPTION / "urdf" / "rp1_24dof.urdf").resolve()
    link = destination / "rp1_24dof.urdf"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(urdf)
    elif link.resolve() != urdf:
        link.unlink()
        link.symlink_to(urdf)
    return link
