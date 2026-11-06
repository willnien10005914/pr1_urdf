import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

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
            "RP1-Flat", "RP1-Rough", "RP1-AMP", "RP1-BeyondMimic", "RP1-Parkour",
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

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(KeyError):
            task_by_id("RP1-Backflip")
        with self.assertRaises(ValueError):
            clamp_config({"task": "RP1-Backflip"})

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
