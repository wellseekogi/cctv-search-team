import copy
import json
import unittest
from pathlib import Path
from sampling.__main__ import read_jsonl
from sampling.core import InputError, run_sampling

ROOT = Path(__file__).resolve().parents[1]


class UniformTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/sampling/sampling.yaml").read_text(encoding="utf-8"))
        self.config["sampling"]["policy"] = "uniform"
        self.inputs = [read_jsonl(ROOT / "fixtures/event_trace" / name) for name in
                       ("manifest.jsonl", "frames.jsonl", "tracks.jsonl", "video_status.jsonl", "query_exclusions.jsonl")]

    def run_case(self):
        return run_sampling(self.config, *self.inputs)

    def test_1000_2000_5000_and_no_events(self):
        for step, scenes, objects in ((1000, [0,1000,2000,3000], [0,1000,3000]),
                                      (2000, [0,2000], [0]), (5000, [0], [0])):
            with self.subTest(step=step):
                self.config["sampling"]["fixed_step_ms"] = step
                events, samples, _ = self.run_case()
                self.assertEqual(events, [])
                for kind, expected in (("scene", scenes), ("object", objects)):
                    self.assertEqual([s["timestamp_ms"] for s in samples if s["representation"]==kind], expected)
                self.assertTrue(all(s["event_ids"] == [] and s["reasons"] == ["uniform"] for s in samples))

    def test_nearest_tie_prefers_earlier_and_tolerance(self):
        self.inputs[2] = []
        base = self.inputs[1][0]
        self.inputs[1] = [dict(base, frame_index=i, timestamp_ms=t, requested_timestamp_ms=t)
                          for i,t in enumerate([1900,2100])]
        samples = self.run_case()[1]
        self.assertEqual([s["timestamp_ms"] for s in samples], [1900])
        self.inputs[1] = [dict(base, frame_index=0, timestamp_ms=1899, requested_timestamp_ms=1899)]
        self.assertEqual(self.run_case()[1], [])

    def test_exclusion_uses_actual_time_no_replacement(self):
        self.inputs[4] = [{"video_id":"v001", "start_ms":0,"end_ms":2000}]
        samples = self.run_case()[1]
        self.assertEqual([(s["representation"],s["timestamp_ms"]) for s in samples], [("scene",2000)])

    def test_cap_and_intervals(self):
        self.config["sampling"].update(fixed_step_ms=200, max_object_per_track=1, max_scene_per_minute=2)
        _, samples, summary = self.run_case()
        self.assertEqual(summary["counts"], {"events":0,"object":1,"scene":2})
        self.assertIn("CAP", summary["decision_statuses"])
        self.assertIn("INTERVAL", summary["decision_statuses"])

    def test_empty_observations_still_select_scene(self):
        self.inputs[2] = []
        self.assertEqual(self.run_case()[2]["counts"], {"events":0,"object":0,"scene":2})

    def test_failed_frame_skipped(self):
        self.inputs[2] = []
        self.inputs[1][0]["decode_status"] = "failed"
        self.assertEqual([s["timestamp_ms"] for s in self.run_case()[1]], [2000])

    def test_deterministic_and_setting_changes_id(self):
        first = self.run_case()
        self.inputs[1].reverse()
        self.inputs[2].reverse()
        self.assertEqual(first, self.run_case())
        self.config["sampling"]["fixed_step_ms"] = 5000
        self.assertNotEqual(first[1][0]["item_id"], self.run_case()[1][0]["item_id"])

    def test_invalid_step(self):
        self.config["sampling"]["fixed_step_ms"] = 0
        with self.assertRaises(InputError):
            self.run_case()

    def test_partial_video_schedule_anchored_at_zero(self):
        self.inputs[0][0]["available_start_ms"] = 400
        self.inputs[1] = [f for f in self.inputs[1] if f["timestamp_ms"] >= 400]
        self.inputs[2] = [o for o in self.inputs[2] if o["timestamp_ms"] >= 400]
        self.assertEqual([s["timestamp_ms"] for s in self.run_case()[1]], [2000])
