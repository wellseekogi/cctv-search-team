import copy
import json
import math
import unittest
from pathlib import Path

from sampling.__main__ import read_jsonl
from sampling.core import InputError, run_sampling

ROOT = Path(__file__).resolve().parents[1]


class SamplingTests(unittest.TestCase):
    def test_annotation_preview_does_not_invent_completion_or_gap(self):
        self.config["execution_mode"] = "annotation_preview"
        for o in self.inputs[2]:
            o.update(source="gt", confidence=None)
        for f in self.inputs[1]:
            f.update(decode_status="unverified", detector_status="skipped")
        self.inputs[3][0]["video_status"] = "unverified"
        events, samples, summary = self.run_case()
        self.assertTrue(set(e["event_type"] for e in events) <= {"TRACK_START", "POSITION_SHIFT"})
        self.assertFalse(summary["final_evaluation_ready"])
        self.assertTrue(all(s["preview_only"] and not s["timestamp_verified"] for s in samples))
        self.assertEqual(set(summary["track_final_states"].values()), {"UNVERIFIED"})
        del self.config["execution_mode"]
        with self.assertRaises(InputError):
            self.run_case()

    def setUp(self):
        self.config = json.loads((ROOT / "configs/sampling.yaml").read_text(encoding="utf-8"))
        self.inputs = [read_jsonl(ROOT / "fixtures/event_trace" / name) for name in
                       ("manifest.jsonl", "frames.jsonl", "tracks.jsonl", "video_status.jsonl", "query_exclusions.jsonl")]

    def run_case(self):
        return run_sampling(self.config, *self.inputs)

    def pair(self, timestamp=600, ratio=0.03, gain=None):
        first = copy.deepcopy(self.inputs[2][0])
        second = copy.deepcopy(first)
        second.update(timestamp_ms=timestamp, frame_index=1)
        dx = ratio * math.hypot(1920, 1080)
        second["bbox"] = [x+dx if i in (0, 2) else x for i, x in enumerate(first["bbox"])]
        if gain is not None:
            # area term fixed; gain is half the change of blur/(blur+100).
            sharp = .5 + 2*gain
            second["quality"]["blur_var"] = 100*sharp/(1-sharp)
        f0 = copy.deepcopy(self.inputs[1][0])
        f1 = dict(f0, frame_index=1, timestamp_ms=timestamp, requested_timestamp_ms=timestamp)
        self.inputs[1] = [f0, f1]
        self.inputs[2] = [first, second]

    def test_A8_exact_lists_and_links(self):
        events, samples, summary = self.run_case()
        expected = json.loads((ROOT / "fixtures/event_trace/expected.json").read_text())
        self.assertEqual(summary["counts"], expected["counts"])
        self.assertEqual(summary["event_types"], expected["event_types"])
        for representation in ("object", "scene"):
            self.assertEqual([s["timestamp_ms"] for s in samples if s["representation"] == representation],
                             expected[representation + "_timestamps_ms"])
        gap = next(e for e in events if e["event_type"] == "TRACK_GAP")
        self.assertEqual((gap["event_timestamp_ms"], gap["detected_at_ms"], gap["object_status"]), (1200, 2200, "MERGED"))
        sample_map = {s["item_id"]: s for s in samples}
        for ev in events:
            for representation in ("object", "scene"):
                item_id = ev[representation + "_item_id"]
                if item_id:
                    self.assertIn(ev["event_id"], sample_map[item_id]["event_ids"])

    def test_deterministic_even_if_input_order_changes(self):
        first = self.run_case()
        self.inputs[1].reverse()
        self.inputs[2].reverse()
        self.assertEqual(first, self.run_case())

    def test_position_threshold(self):
        for ratio, count in ((.0199, 0), (.02, 1), (.0201, 1)):
            with self.subTest(ratio=ratio):
                self.setUp()
                self.pair(ratio=ratio)
                self.assertEqual(sum(e["event_type"] == "POSITION_SHIFT" for e in self.run_case()[0]), count)

    def test_quality_threshold(self):
        for gain, count in ((.1499, 0), (.15, 1), (.1501, 1)):
            with self.subTest(gain=gain):
                self.setUp()
                self.pair(ratio=0, gain=gain)
                self.assertEqual(sum(e["event_type"] == "QUALITY_GAIN" for e in self.run_case()[0]), count)

    def test_gap_999_and_1000_and_resume(self):
        for timestamp, expected in ((999, ["TRACK_START", "VIDEO_END"]),
                                     (1000, ["TRACK_START", "TRACK_GAP", "TRACK_RESUME", "VIDEO_END"])):
            with self.subTest(timestamp=timestamp):
                self.setUp()
                self.pair(timestamp=timestamp, ratio=0)
                events = self.run_case()[0]
                self.assertEqual([e["event_type"] for e in events], expected)
                if timestamp == 1000:
                    resume = next(e for e in events if e["event_type"] == "TRACK_RESUME")
                    self.assertEqual(resume["segment_id"], 1)

    def test_object_interval(self):
        for timestamp, status in ((499, "INTERVAL"), (500, "ACCEPTED")):
            with self.subTest(timestamp=timestamp):
                self.setUp()
                self.pair(timestamp=timestamp)
                shift = next(e for e in self.run_case()[0] if e["event_type"] == "POSITION_SHIFT")
                self.assertEqual(shift["object_status"], status)

    def test_simultaneous_causes_share_object(self):
        self.pair(gain=.2)
        events, samples, _ = self.run_case()
        both = [e for e in events if e["event_type"] in ("POSITION_SHIFT", "QUALITY_GAIN")]
        self.assertEqual(len(both), 2)
        self.assertEqual(both[0]["object_item_id"], both[1]["object_item_id"])
        self.assertEqual(sum(s["representation"] == "object" and s["timestamp_ms"] == 600 for s in samples), 1)

    def test_exclusion_does_not_remove_state_events(self):
        self.inputs[4] = [{"video_id": "v001", "start_ms": 0, "end_ms": 600}]
        events, samples, _ = self.run_case()
        self.assertTrue(any(e["event_type"] == "TRACK_START" and e["object_status"] == "QUERY_WINDOW" for e in events))
        self.assertTrue(all(s["timestamp_ms"] >= 600 for s in samples))
        self.assertTrue(any(s["timestamp_ms"] == 600 for s in samples))

    def test_missing_quality_keeps_position_and_state(self):
        self.pair()
        for o in self.inputs[2]:
            o["quality"] = None
        events, _, summary = self.run_case()
        self.assertIn("POSITION_SHIFT", [e["event_type"] for e in events])
        self.assertEqual(summary["missing_quality_observations"], 2)
        self.assertTrue(all(e["quality_status"] == "missing" for e in events))

    def test_empty_tracks_success(self):
        self.inputs[2] = []
        events, samples, summary = self.run_case()
        self.assertEqual((events, samples, summary["counts"]), ([], [], {"events": 0, "object": 0, "scene": 0}))

    def test_one_frame_end_merges(self):
        self.inputs[1] = self.inputs[1][:1]
        self.inputs[2] = self.inputs[2][:1]
        events, _, summary = self.run_case()
        self.assertEqual([e["event_type"] for e in events], ["TRACK_START", "VIDEO_END"])
        self.assertEqual(summary["counts"], {"events": 2, "object": 1, "scene": 1})

    def test_lost_at_end_no_end_event(self):
        self.inputs[2] = self.inputs[2][:1]
        events, _, summary = self.run_case()
        self.assertEqual([e["event_type"] for e in events], ["TRACK_START", "TRACK_GAP"])
        self.assertEqual(set(summary["track_final_states"].values()), {"CLOSED"})

    def test_cap_31st_candidate(self):
        first, frame = self.inputs[2][0], self.inputs[1][0]
        self.inputs[0][0].update(duration_ms=16000, available_end_ms=16000)
        self.inputs[3][0]["available_end_ms"] = 16000
        self.inputs[1] = [dict(frame, frame_index=i, timestamp_ms=i*500, requested_timestamp_ms=i*500) for i in range(31)]
        self.inputs[2] = [dict(first, frame_index=i, timestamp_ms=i*500,
                               bbox=[100+100*(i % 2), 200, 300+100*(i % 2), 360]) for i in range(31)]
        events, _, summary = self.run_case()
        self.assertEqual(summary["counts"]["object"], 30)
        self.assertTrue(any(e["event_timestamp_ms"] == 15000 and e["object_status"] == "CAP" for e in events))

    def test_gt_uses_same_function(self):
        for o in self.inputs[2]:
            o.update(source="gt", confidence=None)
        self.assertEqual(self.run_case()[2]["counts"], {"events": 9, "object": 5, "scene": 8})

    def test_invalid_bbox_has_row_number(self):
        self.inputs[2][0]["bbox"][0] = -1
        with self.assertRaisesRegex(InputError, "tracks.jsonl:1: bbox"):
            self.run_case()

    def test_failed_video_not_finalized(self):
        self.inputs[3][0]["video_status"] = "failed"
        with self.assertRaisesRegex(InputError, "not completed"):
            self.run_case()

    def test_gap_decode_failure(self):
        self.inputs[2] = self.inputs[2][:1]
        for frame in self.inputs[1][1:6]:
            frame.update(decode_status="failed", detector_status="skipped")
        gap = next(e for e in self.run_case()[0] if e["event_type"] == "TRACK_GAP")
        self.assertEqual(gap["trigger"]["gap_reason"], "decode_failure")

    def test_scene_interval_independent_of_object(self):
        for timestamp, status in ((199, "INTERVAL"), (200, "ACCEPTED")):
            with self.subTest(timestamp=timestamp):
                self.setUp()
                self.pair(timestamp=timestamp)
                event = next(e for e in self.run_case()[0] if e["event_type"] == "POSITION_SHIFT")
                self.assertEqual(event["object_status"], "INTERVAL")
                self.assertEqual(event["scene_status"], status)

    def test_scene_minute_cap_resets_at_60000(self):
        first, frame = self.inputs[2][0], self.inputs[1][0]
        self.inputs[0][0].update(duration_ms=60200, available_end_ms=60200)
        self.inputs[3][0]["available_end_ms"] = 60200
        self.inputs[1] = [dict(frame, frame_index=i, timestamp_ms=i*200, requested_timestamp_ms=i*200) for i in range(301)]
        # Distinct new tracks guarantee a scene candidate at every time.
        self.inputs[2] = [dict(first, frame_index=i, timestamp_ms=i*200, local_track_id=f"t{i}",
                               track_uid=f"synthetic01:v001:t{i}") for i in range(301)]
        events, samples, _ = self.run_case()
        scenes = [s for s in samples if s["representation"] == "scene"]
        self.assertEqual(len(scenes), 121)
        self.assertEqual(sum(s["timestamp_ms"] < 60000 for s in scenes), 120)
        self.assertEqual(scenes[-1]["timestamp_ms"], 60000)
        self.assertTrue(any(e["scene_status"] == "CAP" for e in events))

    def test_cap_is_shared_across_segments(self):
        self.config["sampling"]["max_object_per_track"] = 1
        events, _, summary = self.run_case()
        resumed = next(e for e in events if e["event_type"] == "TRACK_RESUME")
        self.assertEqual(resumed["object_status"], "CAP")
        self.assertEqual(summary["counts"]["object"], 1)

    def test_run_id_does_not_change_item_or_event_ids(self):
        events, samples, _ = self.run_case()
        again, selected, _ = run_sampling(self.config, *self.inputs, run_id="another-run")
        self.assertEqual([e["event_id"] for e in events], [e["event_id"] for e in again])
        self.assertEqual([s["item_id"] for s in samples], [s["item_id"] for s in selected])


if __name__ == "__main__":
    unittest.main()
