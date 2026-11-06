"""Turn an HHTools robot CSV into a viewer motion."""
from __future__ import annotations

import csv
import io

ROOT_COLUMNS = ("root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz")


class CsvMotionError(ValueError):
    pass


def motion_from_hhtools_csv(text: str, *, motion_id: str = "retarget", title: str = "Retarget") -> dict:
    metadata: dict[str, str] = {}
    rows: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            key, separator, value = line[1:].partition(":")
            if separator:
                metadata[key.strip()] = value.strip()
        elif line.strip():
            rows.append(line)
    if not rows:
        raise CsvMotionError("The CSV has no data rows.")
    reader = csv.DictReader(io.StringIO("\n".join(rows)))
    if reader.fieldnames is None:
        raise CsvMotionError("The CSV has no header.")
    header = list(reader.fieldnames)
    required = {"time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"}
    missing = required.difference(header)
    if missing:
        raise CsvMotionError("CSV is missing " + ", ".join(sorted(missing)))
    dof_names = [column.removeprefix("dof_") for column in header if column.startswith("dof_")]
    if not dof_names:
        raise CsvMotionError("CSV has no joint columns.")
    frames = list(reader)
    if len(frames) < 2:
        raise CsvMotionError("Need at least two frames.")
    q_rows = []
    roots = []
    for frame in frames:
        q_rows.append([round(float(frame[f"dof_{name}"]), 5) for name in dof_names])
        roots.append([
            round(float(frame["root_x"]), 5),
            round(float(frame["root_y"]), 5),
            round(float(frame["root_z"]), 5),
            round(float(frame["root_qw"]), 5),
            round(float(frame["root_qx"]), 5),
            round(float(frame["root_qy"]), 5),
            round(float(frame["root_qz"]), 5),
        ])
    duration = float(frames[-1]["time"]) - float(frames[0]["time"])
    fps = (len(frames) - 1) / duration if duration > 1e-6 else 30.0
    return {
        "id": motion_id,
        "title": title,
        "kind": "hhtools_retarget",
        "fps": round(fps, 4),
        "frames": len(frames),
        "joint_names": dof_names,
        "q": q_rows,
        "root": roots,
        "summary": "Geometric retarget from HHTools. It is not a physics-validated PPO rollout.",
        "sample_rate": metadata.get("sample_rate", ""),
    }
