"""Quaternion layout shared by Isaac Lab and the studio viewer.

Isaac Lab 3 stores root_quat_w as xyzw. The viewer stores
[x, y, z, qw, qx, qy, qz]. Treating xyzw as wxyz turns a forward stand
into a robot that faces backward.
"""
from __future__ import annotations


def viewer_root(position, quaternion_xyzw) -> list[float]:
    px, py, pz = (float(value) for value in position)
    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    return [
        round(px, 5), round(py, 5), round(pz, 5),
        round(qw, 5), round(qx, 5), round(qy, 5), round(qz, 5),
    ]
