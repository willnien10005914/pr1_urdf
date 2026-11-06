#!/usr/bin/env python3
"""Register the vendored RP1 URDF with HHTools. Run with the hhtools interpreter."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pr1store.preset import calibration_text, link_urdf, user_preset_dir  # noqa: E402


def main() -> None:
    destination = user_preset_dir()
    link = link_urdf(destination)
    from hhtools.robot.registry import get, refresh
    from hhtools.robot.scaffold import scaffold_yaml_file

    scaffold_yaml_file(link, root_dir=destination)
    for reference in ("smpl", "smplx"):
        path = destination / f"retarget_calibration_{reference}.yaml"
        if not path.is_file():
            path.write_text(calibration_text(reference), encoding="utf-8")
            print(f"wrote {path}")
    refresh()
    preset = get("rp1")
    print(f"preset {preset.name} urdf {preset.urdf_path}")


if __name__ == "__main__":
    main()
