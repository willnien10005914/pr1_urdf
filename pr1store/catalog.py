"""Robot models, official training recipes, and the PR1 learning stack."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = ROOT / "assets" / "rp1_description"
UPSTREAM = "https://github.com/Roboparty/rp1_description"

ROBOTS = (
    {
        "id": "rp1_24dof",
        "file": "urdf/rp1_24dof.urdf",
        "title": "24 自由度身體",
        "summary": "roboparty_train 實際在訓的配置：每條腿 6 自由度、腰 2 自由度、手臂各 5 自由度。",
    },
    {
        "id": "rp1",
        "file": "urdf/rp1.urdf",
        "title": "28 自由度身體",
        "summary": "腿和腰相同，每隻手臂多一顆三軸手腕。部署配置存在，預設策略包仍是 24 自由度。",
    },
    {
        "id": "rp1_hand",
        "file": "urdf/rp1_hand.urdf",
        "title": "身體 + RP 手",
        "summary": "rp1_hand.urdf 裡可動的手指關節。手指動作不在行走 PPO 任務裡。",
    },
    {
        "id": "rp1_hand_fixed",
        "file": "urdf/rp1_hand_fixed.urdf",
        "title": "手部固定",
        "summary": "保留手部連桿，手指關節鎖住。用來檢查手臂鏈，不必逐一驅動手指。",
    },
)

# Defaults copied from roboparty_train agent configs (Isaac Lab 3 / RSL-RL).
# Practice presets shorten max_iterations and num_envs so a laptop GPU can
# show a real log without committing to the full published budget.
TASKS = (
    {
        "id": "RP1-Flat",
        "title": "平地行走",
        "family": "速度 PPO",
        "learns": "在平地上依照指令走路和站立。",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 9001,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-Rough",
        "title": "崎嶇地面",
        "family": "速度 PPO",
        "learns": "同一套速度策略，改在不平整的地面上。",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 9001,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-AMP",
        "title": "AMP 風格",
        "family": "對抗式動作先驗",
        "learns": "用判別器報酬把策略拉向 AMP 片段（走路、樓梯、轉身）。",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 5000,
            "learning_rate": 1.0e-4,
            "entropy_coef": 0.01,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-BeyondMimic",
        "title": "BeyondMimic",
        "family": "動作追蹤",
        "learns": "全身追蹤一段重定向後的片段，例如揮手或跳舞。",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 8000,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-Ballet",
        "title": "芭蕾平衡",
        "family": "動作追蹤",
        "learns": "從始站開始學完八個已存的芭蕾姿勢。每個姿勢先站穩再進下一個；若一直站不穩，仍會在時限後進到下一步，所以訓練不會停在第一步。",
        "defaults": {
            "num_envs": 2048,
            "max_iterations": 12000,
            "learning_rate": 1.0e-3,
            "entropy_coef": 0.003,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
    {
        "id": "RP1-Parkour",
        "title": "跑酷攀爬",
        "family": "感知行走",
        "learns": "用深度圖學習樓梯、缺口和障礙。這是向上攀爬的任務，不是一段預錄的空翻。",
        "defaults": {
            "num_envs": 4096,
            "max_iterations": 30000,
            "learning_rate": 1.0e-4,
            "entropy_coef": 0.005,
            "num_steps_per_env": 24,
            "clip_param": 0.2,
            "gamma": 0.99,
            "seed": 42,
        },
        "practice": {"num_envs": 64, "max_iterations": 20},
    },
)

STACK = [
    {
        "step": 1,
        "title": "影片進去，身體模型出來",
        "body": "用 yt-dlp 和 ffmpeg 裁切一段 YouTube。GVHMR 估計 SMPL 或 SMPL-X 的姿勢序列。身體模型權重留在這台電腦上，授權分開，不在這個儲存庫裡。",
    },
    {
        "step": 2,
        "title": "HHTools 重定向",
        "body": "human-humanoid-tools 用 Newton IK 對到 RP1 的 URDF，寫出機器人 CSV（時間、根姿勢、dof_*）。預設對的是 SMPL-X，另一個選項是 SMPL。",
    },
    {
        "step": 3,
        "title": "訓練用的動作格式",
        "body": "roboparty_train 用 convert_motion.py 把 CSV 轉成訓練格式：AMP 示範是 120 Hz 的 .pkl，BeyondMimic 追蹤片段是 50 Hz 的 .npz。",
    },
    {
        "step": 4,
        "title": "Isaac Lab 3 的 PPO",
        "body": "RSL-RL 訓練 RP1-Flat、RP1-Rough、RP1-AMP、RP1-BeyondMimic、RP1-Ballet 或 RP1-Parkour。公開指令預設用 MJWarp，也可以改選 PhysX。網頁上的進度條讀的是即時日誌裡的 Learning iteration。",
    },
    {
        "step": 5,
        "title": "現在可以播什麼",
        "body": "這裡有官方參考片段（走路、爬樓梯、揮手、跳舞），也有訓完的八步芭蕾權重 checkpoints/rp1_ballet_balance/2026-11-08_05-02-50/model_11999.pt。檢視器把參考動作和這份策略的錄影播在 URDF 上。",
    },
]


def upstream_commit() -> str:
    path = DESCRIPTION / "UPSTREAM_COMMIT.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def joint_table(urdf_file: str) -> list[dict]:
    tree = ET.parse(DESCRIPTION / urdf_file)
    rows = []
    for joint in tree.getroot().findall("joint"):
        kind = joint.get("type") or ""
        if kind == "fixed":
            continue
        limit = joint.find("limit")
        lower = upper = None
        if limit is not None:
            if limit.get("lower") is not None:
                lower = float(limit.get("lower"))
            if limit.get("upper") is not None:
                upper = float(limit.get("upper"))
        rows.append({
            "name": joint.get("name"),
            "type": kind,
            "lower": lower,
            "upper": upper,
        })
    return rows


def robot_catalog() -> list[dict]:
    robots = []
    for spec in ROBOTS:
        joints = joint_table(spec["file"])
        robots.append({
            **spec,
            "urdf_url": f"/assets/rp1_description/{spec['file']}",
            "joints": len(joints),
            "joint_names": [row["name"] for row in joints],
        })
    return robots


def task_by_id(task_id: str) -> dict:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise KeyError(task_id)


def public_catalog() -> dict:
    return {
        "robots": [{key: robot[key] for key in ("id", "title", "summary", "urdf_url", "joints")} for robot in robot_catalog()],
        "tasks": [
            {
                "id": task["id"],
                "title": task["title"],
                "family": task["family"],
                "learns": task["learns"],
                "defaults": task["defaults"],
                "practice": {**task["defaults"], **task["practice"]},
            }
            for task in TASKS
        ],
        "stack": STACK,
        "description": {
            "upstream": UPSTREAM,
            "commit": upstream_commit(),
            "note": "實際訓練的是 24 自由度身體。RP1 總覽裡的 30 和 42 自由度還算進 2 自由度的頭和兩隻 6 軸手；那些額外模組在這份快照裡沒有獨立的 URDF。",
        },
        "flip": {
            "available": False,
            "title": "後空翻 / 前空翻",
            "summary": "RoboParty 在 RP1 總覽公開過前空翻片段。roboparty_train 的公開動作包沒有空翻軌跡，也沒有空翻權重。要研究的話，先重定向一段空翻影片，再對那份 CSV 訓練 RP1-BeyondMimic。",
        },
    }
