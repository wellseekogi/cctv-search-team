import copy
import json
import unittest
from pathlib import Path
from sampling.__main__ import read_jsonl
from sampling.core import InputError, run_sampling

ROOT = Path(__file__).resolve().parents[1]


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/sampling/sampling.yaml").read_text(encoding="utf-8"))
        self.config["sampling"]["policy"] = "hybrid"
        self.inputs = [read_jsonl(ROOT / "fixtures/event_trace" / n) for n in
                       ("manifest.jsonl", "frames.jsonl", "tracks.jsonl", "video_status.jsonl", "query_exclusions.jsonl")]

    def run_case(self):
        return run_sampling(self.config, *self.inputs)

    def custom(self, frame_times, object_time=None, end=6000):
        base = self.inputs[1][0]
        self.inputs[0][0].update(duration_ms=end, available_end_ms=end)
        self.inputs[3][0]["available_end_ms"] = end
        self.inputs[1] = [dict(base, frame_index=i, timestamp_ms=t, requested_timestamp_ms=t) for i,t in enumerate(frame_times)]
        self.inputs[2] = [] if object_time is None else [dict(self.inputs[2][0], frame_index=frame_times.index(object_time), timestamp_ms=object_time)]

    def test_A8_same_counts_periodic_event_merged(self):
        events, samples, summary = self.run_case()
        self.assertEqual(summary["counts"], {"events":9,"object":5,"scene":8})
        scene = next(s for s in samples if s["representation"]=="scene" and s["timestamp_ms"]==0)
        self.assertEqual(scene["reasons"], ["TRACK_START","periodic_scene"])
        self.assertEqual(next(e for e in events if e["event_type"]=="TRACK_START")["scene_status"], "MERGED")

    def test_no_objects_still_periodic_scenes(self):
        self.custom([0,5000])
        _, samples, summary = self.run_case()
        self.assertEqual(summary["counts"], {"events":0,"object":0,"scene":2})
        self.assertTrue(all(s["event_ids"]==[] and s["reasons"]==["periodic_scene"] for s in samples))

    def test_future_reservation_wins_over_earlier_event(self):
        self.custom([4900,5000],4900)
        events, samples, _ = self.run_case()
        self.assertEqual([s["timestamp_ms"] for s in samples if s["representation"]=="scene"], [5000])
        self.assertTrue(all(e["scene_status"]=="INTERVAL" for e in events))
        self.assertEqual(sum(s["representation"]=="object" for s in samples),1)

    def test_exact_200ms_keeps_both(self):
        self.custom([4800,5000],4800)
        self.assertEqual([s["timestamp_ms"] for s in self.run_case()[1] if s["representation"]=="scene"], [4800,5000])

    def test_excluded_periodic_does_not_block_event(self):
        self.custom([4900,5000],4900)
        self.inputs[4] = [{"video_id":"v001","start_ms":5000,"end_ms":5100}]
        _, samples, summary = self.run_case()
        self.assertEqual([s["timestamp_ms"] for s in samples if s["representation"]=="scene"],[4900])
        self.assertEqual(summary["periodic_statuses"]["QUERY_WINDOW"],1)

    def test_periodic_reservation_consumes_minute_cap_first(self):
        self.custom([0,1000,5000],1000)
        self.config["sampling"]["max_scene_per_minute"] = 2
        events, samples, _ = self.run_case()
        self.assertEqual([s["timestamp_ms"] for s in samples if s["representation"]=="scene"],[0,5000])
        self.assertTrue(all(e["scene_status"]=="CAP" for e in events))

    def test_nearest_tie_and_tolerance_and_failed(self):
        self.custom([4900,5100])
        self.assertEqual([s["timestamp_ms"] for s in self.run_case()[1]],[4900])
        self.inputs[1][0]["decode_status"]="failed"
        self.assertEqual(self.run_case()[1],[])
        self.custom([4899])
        self.assertEqual(self.run_case()[1],[])

    def test_object_choices_unchanged_and_deterministic(self):
        first = self.run_case()
        self.inputs[1].reverse()
        self.inputs[2].reverse()
        self.assertEqual(first,self.run_case())
        self.config["sampling"]["policy"]="object_event"
        baseline=self.run_case()
        select=lambda r:[(s["timestamp_ms"],s["bbox"],s["track_uid"]) for s in r[1] if s["representation"]=="object"]
        self.assertEqual(select(first),select(baseline))

    def test_invalid_period(self):
        self.config["sampling"]["periodic_scene_ms"]=0
        with self.assertRaises(InputError):
            self.run_case()
