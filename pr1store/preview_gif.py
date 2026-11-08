"""Small GIF of a few Isaac robots plus the hand and foot paths they are learning.

The picture is drawn from the live simulation state. It is not a realtime
stream: training writes one file every few dozen iterations.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "left_wrist_link": (228, 87, 46),
    "right_wrist_link": (243, 167, 18),
    "left_ankle_roll_link": (46, 134, 171),
    "right_ankle_roll_link": (27, 127, 78),
}
EDGES = (
    ("base_link", "waist_yaw_link"),
    ("waist_yaw_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_link"),
    ("waist_yaw_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_link"),
    ("base_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("base_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
)


def _project(point, origin, scale: float, center: tuple[float, float]) -> tuple[float, float]:
    x = float(point[0]) - origin[0]
    y = float(point[1]) - origin[1]
    z = float(point[2])
    horizontal = (-x + y) * 0.72
    vertical = z + (-x - y) * 0.18
    return center[0] + horizontal * scale, center[1] - vertical * scale


def write_learning_gif(
    path: Path,
    frames: list[list[dict[str, tuple[float, float, float]]]],
    paths: list[dict[str, list[tuple[float, float, float]]]],
    *,
    iteration: int,
    total: int,
) -> None:
    """Write preview.gif and preview.json next to each other."""
    if not frames or not frames[0]:
        raise ValueError("A learning preview needs at least one robot frame.")
    robots = len(frames[0])
    width, height = 280 * robots, 300
    images = []
    for frame in frames:
        image = Image.new("RGB", (width, height), (16, 22, 29))
        draw = ImageDraw.Draw(image)
        for index, pose in enumerate(frame):
            left = 280 * index
            center = (left + 140, 210)
            origin = pose.get("base_link", (0.0, 0.0, 0.0))
            scale = 150.0
            draw.line((left + 24, 250, left + 256, 250), fill=(48, 64, 78), width=2)
            future = paths[index] if index < len(paths) else {}
            for name, color in COLORS.items():
                trail = future.get(name) or []
                points = [_project(point, origin, scale, center) for point in trail]
                if len(points) >= 2:
                    draw.line(points, fill=color, width=3)
                if points:
                    x, y = points[0]
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
            for start, end in EDGES:
                if start not in pose or end not in pose:
                    continue
                draw.line(
                    (_project(pose[start], origin, scale, center), _project(pose[end], origin, scale, center)),
                    fill=(214, 224, 232),
                    width=3,
                )
            draw.text((left + 12, 12), f"robot {index + 1}", fill=(170, 186, 198))
        draw.text((12, height - 22), f"iteration {iteration}/{total}", fill=(214, 224, 232))
        images.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.gif")
    images[0].save(
        temporary, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=True,
    )
    temporary.replace(path)
    summary = path.with_name("preview.json")
    summary.write_text(json.dumps({
        "iteration": iteration,
        "total": total,
        "robots": robots,
        "hands": "left orange, right gold",
        "feet": "left blue, right green",
    }), encoding="utf-8")
