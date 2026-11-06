# PR1 Studio

A local site for the RoboParty RP1 humanoid: inspect the published URDF, search and drive every joint, play the open motion references, retarget a YouTube interval with HHTools, and queue an Isaac Lab PPO run with a live progress bar.

The learning path follows [RP1](https://github.com/Roboparty/RP1) and [roboparty_train](https://github.com/Roboparty/roboparty_train):

1. Cut a YouTube interval.
2. GVHMR estimates SMPL or SMPL-X.
3. [human-humanoid-tools](https://github.com/Roboparty/human-humanoid-tools) retargets onto the 24-DoF body.
4. Isaac Lab 3 and RSL-RL train `RP1-Flat`, `RP1-Rough`, `RP1-AMP`, `RP1-BeyondMimic`, or `RP1-Parkour`.

The viewer plays reference trajectories from `roboparty_train` (walk, stair climb, wave, dance). That checkout does not include a released `model_*.pt`, and the open motion pack does not include a backflip. A front flip is a published RP1 demo, not a file you can load as a policy here. To study one, retarget a clip and train BeyondMimic.

## Run

Python 3.11+ is enough for the viewer, joint search, and the job API. Training and retarget use the checkouts already on this machine.

```bash
bash scripts/start_studio.sh          # http://127.0.0.1:8890/
bash scripts/pinggy.sh                # HTTPS tunnel to port 8890
```

The page binds `0.0.0.0`. Free Pinggy tunnels last about 60 minutes and print an `https://….free.pinggy.net` address. Use the HTTPS link.

`scripts/start_studio.sh` also starts one worker thread. `scripts/worker.sh` is only for a second terminal; it exits when the site already holds the lock.

## What is real

| Action | What happens |
| --- | --- |
| URDF and joint sliders | Loads the vendored RoboParty meshes. Angles stay inside the URDF limits. No physics. |
| Walk, climb, wave, dance | Plays the official AMP / BeyondMimic references at 30 Hz on the 24-DoF body. |
| YouTube start/end | Queues yt-dlp → GVHMR → `hhtools retarget --robot rp1`. SMPL-X is the default; SMPL is the other calibration. |
| Train | Writes a job and, when `~/projects/roboparty_train/.venv` exists, launches Isaac Lab. ETA is computed from finished learning iterations, the same way the Q1 store bar works. Before the first iteration the ETA stays blank while Isaac Lab starts. |
| Practice length | 64 envs and 20 iterations so a log can appear. Published budget restores the numbers from the agent configs (for example 9001 iterations on `RP1-Flat`). |

SMPL and SMPL-X weights are not in this repository. The retarget job reads `~/.cache/hhtools/body_models`. The first retarget also registers the 24-DoF URDF under `~/.config/hhtools/robots/rp1` and writes a bootstrap T-pose calibration. Refine that calibration in HHTools before treating a clip as final.

If the Isaac environment is missing, the job is marked blocked and the page says so. It does not invent a progress bar.

## Layout

- `assets/rp1_description` — URDF, MJCF, and STL snapshot. See `NOTICE.md`.
- `assets/motions` — downsampled references. Regenerate with `python3 scripts/export_references.py` when `~/projects/roboparty_train` is present.
- `public/` — the studio page.
- `.data/jobs/` — local job logs. Not committed.

```bash
python3 -m unittest discover -s tests -v
```

## Sources

- [RoboParty](https://github.com/Roboparty) and the [RP1 overview](https://github.com/Roboparty/RP1)
- [RP1 description](https://github.com/Roboparty/rp1_description)
- [roboparty_train](https://github.com/Roboparty/roboparty_train)
- [human-humanoid-tools](https://github.com/Roboparty/human-humanoid-tools)
- Joint search, YouTube seconds, and the iteration ETA follow the local [q1_store_urdf](https://github.com/willnien10005914/q1_store_urdf) studio.
