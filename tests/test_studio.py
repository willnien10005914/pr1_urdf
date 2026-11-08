import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from pr1store.ballet_steps import hhtools_csv, load_document, replay_motion, save_step
from pr1store.preview_gif import write_learning_gif
from pr1store.view_math import viewer_root
from pr1store.catalog import joint_table, public_catalog, task_by_id
from pr1store.csv_motion import CsvMotionError, motion_from_hhtools_csv
from pr1store.jobs import JobStore
from pr1store.preset import tpose_joint_q
from pr1store.progress import eta_seconds, format_duration, parse_metrics, train_percent
from pr1store.train_cmd import clamp_config, train_argv
from pr1store.worker import Worker
from pr1store.youtube import require_youtube
from server import Handler

ROOT = Path(__file__).resolve().parents[1]


class ProgressTests(unittest.TestCase):
    def test_eta_waits_for_one_iteration(self):
        self.assertIsNone(eta_seconds(0, 20, 30))
        self.assertEqual(eta_seconds(5, 20, 10), 30)
        self.assertEqual(eta_seconds(20, 20, 10), 0)

    def test_log_parser_reads_the_latest_iteration(self):
        text = "Learning iteration 2/20\nMean reward: 1.5\nComputation: 4000 steps/s\nLearning iteration 3/20\n"
        metrics = parse_metrics(text)
        self.assertEqual(metrics["iteration"], 3)
        self.assertEqual(metrics["iteration_total"], 20)
        self.assertEqual(metrics["mean_reward"], 1.5)
        self.assertEqual(metrics["steps_per_second"], 4000)
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertAlmostEqual(train_percent(10, 20, started=True), 0.08 + 0.92 * 0.5)


class CatalogTests(unittest.TestCase):
    def test_body_models_match_the_upstream_counts(self):
        self.assertEqual(len(joint_table("urdf/rp1_24dof.urdf")), 24)
        self.assertEqual(len(joint_table("urdf/rp1.urdf")), 28)
        self.assertEqual(len(joint_table("urdf/rp1_hand.urdf")), 50)
        catalog = public_catalog()
        self.assertEqual([task["id"] for task in catalog["tasks"]], [
            "RP1-Flat", "RP1-Rough", "RP1-AMP", "RP1-BeyondMimic", "RP1-Ballet", "RP1-Parkour",
        ])
        self.assertFalse(catalog["flip"]["available"])

    def test_walk_reference_uses_the_trained_joint_names(self):
        path = ROOT / "assets" / "motions" / "walk.json"
        clip = json.loads(path.read_text(encoding="utf-8"))
        names = {row["name"] for row in joint_table("urdf/rp1_24dof.urdf")}
        self.assertTrue(set(clip["joint_names"]).issubset(names))
        self.assertEqual(len(clip["q"][0]), 24)
        self.assertGreater(clip["frames"], 10)


class CommandTests(unittest.TestCase):
    def test_practice_flags_stay_inside_the_isaac_cli(self):
        config = clamp_config({"task": "RP1-Flat", "practice": True, "num_envs": 64, "max_iterations": 20})
        argv = train_argv(config)
        self.assertIn("--task=RP1-Flat", argv)
        self.assertIn("--num_envs=64", argv)
        self.assertIn("--max_iterations=20", argv)
        self.assertNotIn("learning_rate", " ".join(argv))
        self.assertEqual(config["clip_id"], "")
        tracked = clamp_config({"task": "RP1-BeyondMimic", "clip_id": "c212cca38f844173aa36e14c5d54bec4"})
        self.assertEqual(tracked["clip_id"], "c212cca38f844173aa36e14c5d54bec4")
        with self.assertRaises(ValueError):
            clamp_config({"task": "RP1-BeyondMimic", "clip_id": "../secret"})

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(KeyError):
            task_by_id("RP1-Backflip")
        with self.assertRaises(ValueError):
            clamp_config({"task": "RP1-Backflip"})

    def test_gvhmr_import_uses_the_results_filename(self):
        import importlib.util
        path = ROOT / "scripts" / "youtube_seconds_to_hhtools.py"
        spec = importlib.util.spec_from_file_location("youtube_seconds_to_hhtools", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.gvhmr_sequence_id(Path("hmr4d_results.pt")), "hmr4d_results.pt")
        flipped = module.facing_flip_mask([0, 10, 170, 175, 5])
        self.assertEqual(flipped.tolist(), [False, False, True, True, False])
        held = module.facing_flip_mask(list(range(0, 90, 15)))
        self.assertFalse(bool(held.any()))
        spikes = module.heading_spike_deltas([0, 0, 0, 80, 0, 0, 0])
        self.assertLess(spikes[3], -45)
        self.assertTrue((spikes[:3] == 0).all())
        phrase = module.ballet_phrase_targets([0.0, 0.85, 2.15, 2.85, 3.40, 3.62, 4.0])
        self.assertGreater(phrase["left_ankle"][1, 1], phrase["left_ankle"][0, 1] + 0.1)
        self.assertGreater(phrase["right_ankle"][2, 2], phrase["left_ankle"][2, 2] + 0.3)
        self.assertGreater(phrase["left_wrist"][2, 2], 0.5)
        self.assertGreater(phrase["right_wrist"][2, 2], 0.5)
        self.assertLess(phrase["left_wrist"][3, 2], phrase["left_wrist"][2, 2] - 0.4)
        self.assertGreater(phrase["right_ankle"][3, 0], 0.1)
        self.assertLess(phrase["right_ankle"][3, 1], -0.2)
        self.assertGreater(phrase["left_ankle"][4, 1], phrase["right_ankle"][4, 1] + 0.3)
        self.assertGreater(phrase["left_ankle"][4, 2], phrase["right_ankle"][4, 2] + 0.15)
        self.assertGreater(phrase["right_wrist"][4, 2], phrase["left_wrist"][4, 2])
        self.assertLess(phrase["right_wrist"][4, 1], phrase["left_wrist"][4, 1])
        self.assertGreater(phrase["left_foot"][5, 2], phrase["left_ankle"][5, 2])
        self.assertLess(abs(phrase["left_ankle"][6, 1] - 0.11), 0.02)
        self.assertLess(phrase["left_wrist"][6, 2], 0.0)

    def test_youtube_host_is_restricted(self):
        self.assertEqual(require_youtube("https://www.youtube.com/watch?v=a5R0Xf46hyg"), "a5R0Xf46hyg")
        self.assertEqual(require_youtube("https://youtu.be/a5R0Xf46hyg?t=3"), "a5R0Xf46hyg")
        with self.assertRaises(ValueError):
            require_youtube("https://example.com/watch?v=a5R0Xf46hyg")

    def test_tpose_shoulder_is_inside_the_urdf_limit(self):
        limits = {row["name"]: row for row in joint_table("urdf/rp1_24dof.urdf")}
        posed = tpose_joint_q()
        for name, value in posed.items():
            self.assertLessEqual(value, limits[name]["upper"])
            self.assertGreaterEqual(value, limits[name]["lower"])


class CsvTests(unittest.TestCase):
    def test_hhtools_csv_round_trip(self):
        text = "\n".join([
            "# sample_rate: 50",
            "time,root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,dof_left_hip_pitch_joint,dof_right_knee_joint",
            "0,0,0,0.9,0,0,0,1,0.1,-0.2",
            "0.02,0.01,0,0.9,0,0,0,1,0.2,-0.1",
        ])
        motion = motion_from_hhtools_csv(text, title="sample")
        self.assertEqual(motion["joint_names"], ["left_hip_pitch_joint", "right_knee_joint"])
        self.assertEqual(motion["frames"], 2)
        with self.assertRaises(CsvMotionError):
            motion_from_hhtools_csv("time\n0\n")


class BalletStepTests(unittest.TestCase):
    def test_six_poses_save_and_replay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            document = load_document(data)
            self.assertEqual(
                [step["id"] for step in document["steps"]],
                ["stand-start", "left-step", "right-knee", "arms-up", "right-plant", "sides", "hook", "finish"],
            )
            saved = save_step(
                data, "right-knee",
                {"right_knee_joint": 5, "right_hip_pitch_joint": -1.2},
                {"x": 2, "y": -0.1},
            )
            self.assertEqual(saved["joints"]["right_knee_joint"], 2.83)
            self.assertEqual(saved["joints"]["right_hip_pitch_joint"], -1.2)
            self.assertEqual(saved["com"], {"x": 0.4, "y": -0.1})
            self.assertTrue(saved["saved"])
            motion = replay_motion(load_document(data))
            self.assertGreater(motion["frames"], 6)
            self.assertEqual(motion["q"][0], motion["q"][-1])
            column = motion["joint_names"].index("right_hip_pitch_joint")
            self.assertLess(min(row[column] for row in motion["q"]), -0.5)
            self.assertTrue(motion["plant_feet"])
            self.assertEqual(max(row[0] for row in motion["root"]), 0.4)
            (data / "ballet_steps.json").write_text(
                json.dumps({"steps": {"left-step": {"left_knee_joint": -0.4}}}),
                encoding="utf-8",
            )
            legacy = load_document(data)
            left = next(step for step in legacy["steps"] if step["id"] == "left-step")
            self.assertEqual(left["joints"]["left_knee_joint"], -0.4)
            self.assertEqual(left["com"], {"x": 0.0, "y": 0.0})
            opening = next(step for step in legacy["steps"] if step["id"] == "stand-start")
            closing = next(step for step in legacy["steps"] if step["id"] == "finish")
            self.assertEqual(opening["joints"]["left_shoulder_pitch_joint"], 0.0)
            self.assertEqual(opening["joints"]["right_elbow_joint"], 0.0)
            self.assertEqual(closing["joints"], opening["joints"])
            with self.assertRaises(KeyError):
                save_step(data, "missing", {"right_knee_joint": 0})
            from pr1store.ballet_steps import phrase_hold_frames
            frames = phrase_hold_frames(268)
            self.assertEqual(len(frames), 8)
            self.assertEqual(frames, sorted(set(frames)))
            self.assertLess(frames[0], 15)
            self.assertGreater(frames[1], frames[0] + 20)
            self.assertGreater(frames[-1], 250)

    def test_hhtools_csv_keeps_saved_com(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            save_step(data, "left-step", {"left_hip_pitch_joint": -0.4}, {"x": 0.12, "y": -0.08})
            text = hhtools_csv(load_document(data))
            self.assertIn("# sample_rate: 30", text)
            self.assertIn("dof_left_hip_pitch_joint", text.splitlines()[1])
            self.assertIn("0.120000,-0.080000", text)


class PreviewTests(unittest.TestCase):
    def test_identity_quaternion_stays_facing_forward(self):
        root = viewer_root((0.1, -0.2, 0.86), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(root[3:], [1.0, 0.0, 0.0, 0.0])

    def test_learning_gif_has_a_caption_file(self):
        import tempfile
        pose = {
            "base_link": (0.0, 0.0, 0.86),
            "waist_yaw_link": (0.0, 0.0, 1.05),
            "left_wrist_link": (0.2, 0.2, 1.3),
            "right_wrist_link": (0.2, -0.2, 1.3),
            "left_ankle_roll_link": (0.05, 0.08, 0.08),
            "right_ankle_roll_link": (0.0, -0.08, 0.08),
        }
        path = Path(tempfile.mkdtemp()) / "preview.gif"
        write_learning_gif(path, [[pose, pose]], [{
            "left_wrist_link": [(0.2, 0.2, 1.3), (0.3, 0.2, 1.6)],
            "right_wrist_link": [(0.2, -0.2, 1.3), (0.3, -0.2, 1.6)],
            "left_ankle_roll_link": [(0.05, 0.08, 0.08), (0.12, 0.16, 0.08)],
            "right_ankle_roll_link": [(0.0, -0.08, 0.08), (0.0, -0.08, 0.2)],
        }, {}], iteration=50, total=12000)
        self.assertTrue(path.read_bytes().startswith(b"GIF"))
        caption = json.loads(path.with_name("preview.json").read_text(encoding="utf-8"))
        self.assertEqual(caption["iteration"], 50)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.store = JobStore(ROOT, ROOT / ".data" / "test-jobs")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.store = self.store
        self.server.worker = Worker(self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as response:
            return response.status, response.read()

    def test_pages_and_catalog(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"PR1 Studio", body)
        status, body = self.get("/api/catalog")
        payload = json.loads(body)
        self.assertEqual(payload["robots"][0]["joints"], 24)
        status, _ = self.get("/assets/rp1_description/urdf/rp1_24dof.urdf")
        self.assertEqual(status, 200)

    def test_traversal_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/assets/../server.py")


if __name__ == "__main__":
    unittest.main()
